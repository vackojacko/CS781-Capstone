# MTL BiLSTM-CRF (Malik et al 2022) on top of my phase 1 models
# role branch: bilstm over sentence vectors (+ position, domain), shift branch: bilstm over [v_i ; v_i+1]
import torch
import torch.nn as nn

from backbone import Backbone, WINDOW
from data import LABELS, make_windows
from peft_methods import add_peft

HIDDEN = 256
DROPOUT = 0.2
BACKBONE_DROPOUT = 0.1


# lstm from my A2
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2g = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.h2g = nn.Linear(hidden_size, 4 * hidden_size, bias=True)  # takes a vector memory and vectorizes the computation of the 4 gates

    def forward(self, x_t, state):
        h_prev, c_prev = state[0], state[1]
        gates = self.x2g(x_t) + self.h2g(h_prev)  # pre-activations means the values before activation functions are applied on them
        f_pre, i_pre, g_pre, o_pre = gates.chunk(4, dim=-1)
        f_t = torch.sigmoid(f_pre)
        i_t = torch.sigmoid(i_pre)
        g_t = torch.tanh(g_pre)
        o_t = torch.sigmoid(o_pre)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return (h_t, c_t)


class RecurrentEncoder(nn.Module):
    def __init__(self, cell_cls, input_size, hidden_size, num_layers=1, bidirectional=False, is_lstm=False):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.is_lstm = is_lstm
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.cells = nn.ModuleList()
        for l in range(self.num_layers):
            if l == 0:
                in_dim = input_size
            else:
                in_dim = self.hidden_size * self.num_directions  # this ensures that the weight setup is different for different layers
            for _dir in range(self.num_directions):
                temp = cell_cls(in_dim, hidden_size)  # cell_cls is the class of the cell that we want to instantiate
                self.cells.append(temp)

    def _run_direction(self, cell, x, reverse):
        batch_size, seq_len, _ = x.shape
        device = x.device
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        c = None
        if self.is_lstm:
            c = torch.zeros(batch_size, self.hidden_size).to(device)
        if reverse:
            steps = reversed(range(seq_len))
        else:
            steps = range(seq_len)
        outputs = [None] * seq_len
        for t in steps:
            x_t = x[:, t, :]  # gives all the batches at timestep t
            if self.is_lstm == True:
                h, c = cell(x_t, (h, c))
            else:
                h = cell(x_t, h)
            outputs[t] = h
        stacked_outputs = torch.stack(outputs, dim=1)
        if self.is_lstm:
            final_state = (h, c)
        else:
            final_state = h
        return (stacked_outputs, final_state)

    def forward(self, x):
        layer_input = x
        cell_idx = 0
        final_h = []
        for l in range(self.num_layers):
            dir_outputs, dir_finals = [], []
            for d in range(self.num_directions):
                cell = self.cells[cell_idx]
                cell_idx = cell_idx + 1
                is_reverse = self.bidirectional and d == 1
                out, final = self._run_direction(cell, layer_input, is_reverse)
                dir_outputs.append(out)
                if self.is_lstm:
                    dir_finals.append(final[0])
                else:
                    dir_finals.append(final)
            layer_input = torch.cat(dir_outputs, dim=-1)
            final_h = torch.cat(dir_finals, dim=-1)
        return (layer_input, final_h)


class CRF(nn.Module):
    # score of labels y is sum of E[i, y_i] + sum of T[y_(i-1), y_i] + start[y_1] + end[y_n]
    def __init__(self, n):
        super().__init__()
        self.T = nn.Parameter(torch.zeros(n, n))  # T[a, b] = score for label b coming right after label a
        self.start = nn.Parameter(torch.zeros(n))
        self.end = nn.Parameter(torch.zeros(n))

    def score(self, E, y):
        s = self.start[y[0]] + self.end[y[-1]]
        for i in range(len(y)):
            s = s + E[i, y[i]]
        for i in range(1, len(y)):
            s = s + self.T[y[i - 1], y[i]]
        return s

    def log_Z(self, E):
        # forward algorithm, alpha[b] = log of summed exp scores of all paths ending in label b
        alpha = self.start + E[0]
        for i in range(1, len(E)):
            alpha = torch.logsumexp(alpha.unsqueeze(1) + self.T, dim=0) + E[i]
        return torch.logsumexp(alpha + self.end, dim=0)

    def loss(self, E, y):
        # -log p(gold sequence)
        return self.log_Z(E) - self.score(E, y)

    def viterbi(self, E):
        # same as forward algorithm but max instead of sum, then trace back
        best = self.start + E[0]
        back = []
        for i in range(1, len(E)):
            s = best.unsqueeze(1) + self.T
            back.append(s.argmax(dim=0))
            best = s.max(dim=0).values + E[i]
        path = [int((best + self.end).argmax())]
        for b in reversed(back):
            path.append(int(b[path[-1]]))
        return path[::-1]

    def marginals(self, E):
        # p(label of sentence i = l) using forward and backward passes
        alphas = [self.start + E[0]]
        for i in range(1, len(E)):
            alphas.append(torch.logsumexp(alphas[-1].unsqueeze(1) + self.T, dim=0) + E[i])
        betas = [self.end]
        for i in range(len(E) - 1, 0, -1):
            betas.append(torch.logsumexp(self.T + (E[i] + betas[-1]).unsqueeze(0), dim=1))
        betas = betas[::-1]
        return (torch.stack(alphas) + torch.stack(betas) - self.log_Z(E)).exp()


class RRModel(nn.Module):
    def __init__(self, backbone, hidden=HIDDEN, context="centered"):
        super().__init__()
        self.backbone = backbone
        d = backbone.d_model
        self.window = WINDOW[backbone.arch] - len(backbone.prefix)
        self.set_context(context)
        self.role_lstm = RecurrentEncoder(LSTMCell, d + 4, hidden, num_layers=1, bidirectional=True, is_lstm=True)
        self.shift_lstm = RecurrentEncoder(LSTMCell, 2 * d, hidden, num_layers=1, bidirectional=True, is_lstm=True)
        self.shift_out = nn.Linear(2 * hidden, 1)
        self.emit = nn.Linear(4 * hidden, len(LABELS))
        self.crf = CRF(len(LABELS))
        self.drop = nn.Dropout(DROPOUT)
        # arch1 did not learn in fp16, so it runs in fp32
        self.fp16 = backbone.arch != "arch1"

    def set_context(self, context):
        # none = no overlap, centered = half the window is context around the core, left = context before it
        self.context = context
        if context == "none":
            self.core = self.window
        else:
            self.core = self.window // 2

    def encode(self, toks, a=0, b=None):
        # vector of sentence s = mean of its token vectors
        device = self.emit.weight.device
        if b is None:
            b = len(toks)
        vecs = [None] * (b - a)
        if self.context == "none":
            # nothing from outside a..b
            windows = []
            for lo, i, j, hi in make_windows(toks[a:b], self.core, self.window):
                windows.append((lo + a, i + a, j + a, hi + a))
        else:
            windows = make_windows(toks, self.core, self.window, a, b, left=(self.context == "left"))
        use_fp16 = device.type == "cuda" and self.fp16
        for lo, i, j, hi in windows:
            ids = list(self.backbone.prefix)
            for s in range(lo, hi):
                ids += toks[s]
            x = torch.tensor([ids]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                h = self.backbone(x)[0]
            h = h.float()
            pos = len(self.backbone.prefix)
            for s in range(lo, hi):
                if i <= s < j:
                    vecs[s - a] = h[pos:pos + len(toks[s])].mean(dim=0)
                pos += len(toks[s])
        return torch.stack(vecs)

    def forward(self, doc, a, b, vecs=None):
        if vecs is None:
            vecs = self.encode(doc["toks"], a, b)
        v = self.drop(vecs)
        # next sentence, zeros for the last one
        nxt = torch.cat([v[1:], torch.zeros_like(v[:1])], dim=0)
        pairs = torch.cat([v, nxt], dim=1)
        n_doc = len(doc["sents"])
        pos = torch.arange(a, b).float().to(v.device) / max(n_doc - 1, 1)
        if doc["domain"] == "CL":
            cl = 1.0
        else:
            cl = 0.0
        extra = torch.stack([pos, pos ** 2, torch.full_like(pos, cl), torch.full_like(pos, 1 - cl)], dim=1)
        role, _ = self.role_lstm(torch.cat([v, extra], dim=1).unsqueeze(0))
        shift, _ = self.shift_lstm(pairs.unsqueeze(0))
        role = role[0]
        shift = shift[0]
        features = torch.cat([role, shift], dim=1)
        emissions = self.emit(self.drop(features))
        shift_scores = self.shift_out(shift).squeeze(1)
        return emissions, shift_scores


def build_model(arch, model_dir, method, bidirectional, context="centered"):
    backbone = Backbone(arch, model_dir, bidirectional=bidirectional, dropout=BACKBONE_DROPOUT)
    backbone = add_peft(backbone, method)
    return RRModel(backbone, context=context)
