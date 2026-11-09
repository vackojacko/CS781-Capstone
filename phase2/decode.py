# constrained BIOES viterbi, spans, within-document propagation, boundary repair, strict macro-F1
import re
from collections import Counter, defaultdict

import numpy as np

from preprocess import LABELS, TAGS, N_TAGS, tags_to_spans


def make_transitions():
    allowed = np.full((N_TAGS, N_TAGS), -np.inf, dtype=np.float32)
    for a in range(N_TAGS):
        ta = TAGS[a]
        for b in range(N_TAGS):
            tb = TAGS[b]
            if ta[:2] in ("B-", "I-"):
                ok = tb in ("I-" + ta[2:], "E-" + ta[2:])
            else:
                ok = tb == "O" or tb[:2] in ("B-", "S-")
            if ok:
                allowed[a, b] = 0.0
    start = np.zeros(N_TAGS, dtype=np.float32)
    end = np.zeros(N_TAGS, dtype=np.float32)
    for i in range(N_TAGS):
        t = TAGS[i]
        if not (t == "O" or t[:2] in ("B-", "S-")):
            start[i] = -np.inf
        if not (t == "O" or t[:2] in ("E-", "S-")):
            end[i] = -np.inf
    return allowed, start, end


TRANS, START, END = make_transitions()


def viterbi(logp):
    T = len(logp)
    score = START + logp[0]
    back = np.zeros((T, N_TAGS), dtype=np.int32)
    for t in range(1, T):
        cand = score[:, None] + TRANS
        back[t] = cand.argmax(0)
        score = cand.max(0) + logp[t]
    score = score + END
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(score.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode_doc(doc, logp):
    if len(logp) == 0:
        return []
    path = viterbi(logp)
    probs = np.exp(logp[np.arange(len(path)), path])
    spans = tags_to_spans(path, doc["starts"], doc["ends"], probs)
    text = doc["text"]
    clean = []
    for s, e, lab, c in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            clean.append((s, e, lab, c))
    return clean


def propagate(text, spans, min_len=4, min_conf=0.0, relabel=True):
    # one label per string in a document, and every free word-bounded occurrence gets tagged
    votes = defaultdict(Counter)
    count = Counter()
    for s, e, lab, c in spans:
        votes[text[s:e]][lab] += c
        count[text[s:e]] += 1
    label_of = {}
    for t, v in votes.items():
        label_of[t] = v.most_common(1)[0][0]
    out = []
    for s, e, lab, c in spans:
        if relabel:
            lab = label_of[text[s:e]]
        out.append((s, e, lab, c))
    taken = np.zeros(len(text) + 1, dtype=bool)
    for sp in out:
        taken[sp[0]:sp[1]] = True
    for t in sorted(label_of, key=len, reverse=True):
        if len(t) < min_len or sum(votes[t].values()) / count[t] < min_conf:
            continue
        for m in re.finditer(r"(?<!\w)" + re.escape(t) + r"(?!\w)", text):
            if not taken[m.start():m.end()].any():
                out.append((m.start(), m.end(), label_of[t], 0.0))
                taken[m.start():m.end()] = True
    return sorted(out)


ABBREV_END = re.compile(r"(?:\b(?:Ors|Anr|Ltd|Co|Inc|Pvt|Bros|Corpn|Assn|Sons|Jr|Sr|No|Nos|etc|Govt|Dept|M/s)|\b[A-Z])\.$")


def repair_spans(text, spans):
    # boundary repair from the training-set error analysis
    out = []
    for s, e, lab, c in spans:
        if lab != "PREC":
            while s > 0 and text[s - 1].isalnum() and text[s].isalnum():
                s -= 1
            while e < len(text) and text[e - 1].isalnum() and text[e].isalnum():
                e += 1
        while e > s + 1 and text[e - 1] in "[({,;:-":
            e -= 1
        while e > s + 1 and text[e - 1].isdigit() and text[e - 2] in "[(":
            e -= 2
        while e > s + 1 and text[e - 1] in "[({,;: ":
            e -= 1
        if lab == "PREC" and e > s + 1 and text[e - 1] == "." and not ABBREV_END.search(text[s:e]):
            e -= 1
        if e > s:
            out.append((s, e, lab, c))
    return out


def strict_scores(gold, pred):
    tp = Counter()
    fp = Counter()
    fn = Counter()
    for d in gold:
        g = set()
        for sp in gold[d]:
            g.add((sp[0], sp[1], sp[2]))
        p = set()
        for sp in pred.get(d, []):
            p.add((sp[0], sp[1], sp[2]))
        for x in g & p:
            tp[x[2]] += 1
        for x in p - g:
            fp[x[2]] += 1
        for x in g - p:
            fn[x[2]] += 1
    per = {}
    for l in LABELS:
        P = 0.0
        if tp[l] + fp[l]:
            P = tp[l] / (tp[l] + fp[l])
        R = 0.0
        if tp[l] + fn[l]:
            R = tp[l] / (tp[l] + fn[l])
        F1 = 0.0
        if P + R:
            F1 = 2 * P * R / (P + R)
        per[l] = {"P": P, "R": R, "F1": F1}
    macro = {}
    for k in ("P", "R", "F1"):
        macro[k] = float(np.mean([per[l][k] for l in LABELS]))
    return macro, per
