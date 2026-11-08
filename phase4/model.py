# chunk classifier (A4 q2 skeleton, completed) and the bilstm over chunk vectors (my A2 lstm)
import torch
import torch.nn as nn

from backbone import Backbone


# vector at the last position whose mask is 1
def last_nonpad(hidden_states, attention_mask):
    T = attention_mask.shape[1]
    pos = T - 1 - attention_mask.long().flip(dims=[1]).argmax(dim=1)
    rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[rows, pos]


def mean_nonpad(hidden_states, attention_mask):
    m = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    return (hidden_states * m).sum(1) / m.sum(1).clamp(min=1)


class SentimentClassifier(nn.Module):
    def __init__(self, backbone, hidden_size, num_classes=2, pool="last"):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, num_classes)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        self.pool = pool

    def features(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        h = self.backbone(input_ids).float()
        if self.pool == "last":
            return last_nonpad(h, attention_mask)
        return mean_nonpad(h, attention_mask)

    def forward(self, input_ids=None, attention_mask=None):
        return self.classifier(self.features(input_ids, attention_mask))


def make_classifier(arch, model_dir, bidirectional=False):
    bb = Backbone(arch, model_dir, bidirectional)
    if bb.bidirectional:
        pool = "mean"
    else:
        pool = "last"
    return SentimentClassifier(bb, bb.d_model, pool=pool)


# bilstm from my A2
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2g = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.h2g = nn.Linear(hidden_size, 4 * hidden_size, bias=True)

    def forward(self, x_t, state):
        h_prev, c_prev = state[0], state[1]
        gates = self.x2g(x_t) + self.h2g(h_prev)
        f_pre, i_pre, g_pre, o_pre = gates.chunk(4, dim=-1)
        f_t = torch.sigmoid(f_pre)
        i_t = torch.sigmoid(i_pre)
        g_t = torch.tanh(g_pre)
        o_t = torch.sigmoid(o_pre)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return (h_t, c_t)


class RecurrentEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.cells = nn.ModuleList()
        for l in range(self.num_layers):
            if l == 0:
                in_dim = input_size
            else:
                in_dim = self.hidden_size * self.num_directions
            for _dir in range(self.num_directions):
                self.cells.append(LSTMCell(in_dim, hidden_size))

    def _run_direction(self, cell, x, reverse):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)
        if reverse:
            steps = reversed(range(seq_len))
        else:
            steps = range(seq_len)
        outputs = [None] * seq_len
        for t in steps:
            x_t = x[:, t, :]
            h, c = cell(x_t, (h, c))
            outputs[t] = h
        return torch.stack(outputs, dim=1), h

    def forward(self, x):
        layer_input = x
        cell_idx = 0
        final_h = []
        for l in range(self.num_layers):
            dir_outputs, dir_finals = [], []
            for d in range(self.num_directions):
                cell = self.cells[cell_idx]
                cell_idx += 1
                out, final = self._run_direction(cell, layer_input, self.bidirectional and d == 1)
                dir_outputs.append(out)
                dir_finals.append(final)
            layer_input = torch.cat(dir_outputs, dim=-1)
            final_h = torch.cat(dir_finals, dim=-1)
        return layer_input, final_h


class Aggregator(nn.Module):
    def __init__(self, d, hidden=128, dropout=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.lstm = RecurrentEncoder(d, hidden, num_layers=1, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 2)

    def forward(self, x):
        _, h = self.lstm(self.norm(x))
        return self.out(self.drop(h))
