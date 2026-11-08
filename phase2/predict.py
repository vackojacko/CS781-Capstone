# final submission (best, 0.6872): ensemble of saved test log-probabilities, propagation, boundary repair,
# then document rules
import json
import os
import zipfile

import numpy as np

import rules as RL
from decode import decode_doc, propagate, repair_spans
from preprocess import DATA_DIR, LABELS, encode_docs, load_split, load_tokenizer

RUNS_DIR = "runs"
OUT_PATH = "submission_best.zip"

# causal ensemble: arch 4 (2 seeds), arch 2 header 256, arch 4 after continued pretraining
ENSEMBLE = [(2.0, "arch4_full_h128"), (1.0, "arch4_full_h128_s1"), (1.0, "arch2_full_h256_s1"), (1.0, "arch4_tapt_full_h128")]
# bidirectional arch 4 + LoRA, seed 0
BIDIR = [(1.0, "arch4_mntp_lora_h128_bidir")]
W_ENSEMBLE = 5.0
W_BIDIR = 3.0
PROP_MIN_LEN = 6
PROP_MIN_CONF = 0.5


def load_logprobs(spec):
    # log of the weighted mean of the runs' probabilities
    out = {}
    tot = 0.0
    for w, run in spec:
        tot += w
        z = np.load(os.path.join(RUNS_DIR, run, "test_logprobs.npz"))
        for k in z.files:
            out[k] = out.get(k, 0) + w * np.exp(z[k].astype(np.float32))
    for k in out:
        out[k] = np.log(np.clip(out[k] / tot, 1e-9, 1))
    return out


def mix_runs():
    ens = load_logprobs(ENSEMBLE)
    bid = load_logprobs(BIDIR)
    mixed = {}
    for d in ens:
        p = (W_ENSEMBLE * np.exp(ens[d]) + W_BIDIR * np.exp(bid[d])) / (W_ENSEMBLE + W_BIDIR)
        mixed[d] = np.log(np.clip(p, 1e-9, 1))
    return mixed


def decode_all(docs, logp):
    pred = {}
    for d, doc in docs.items():
        spans = []
        if len(logp[d]):
            spans = decode_doc(doc, logp[d])
        spans = propagate(doc["text"], spans, PROP_MIN_LEN, PROP_MIN_CONF, True, ("APP",))
        pred[d] = repair_spans(doc["text"], spans)
    return pred


def apply_rules(text, spans):
    spans = repair_spans(text, RL.resolve(text, spans))
    spans = RL.prec_repair(text, spans)
    spans = RL.prune_roles(text, spans)
    spans = RL.label_repair(text, spans)
    spans = RL.counsel_side(text, spans)
    spans = RL.link_header_parties(text, spans)
    return RL.relabel_to_parties(text, spans)


def predict_all(docs, texts, logp):
    base = decode_all(docs, logp)
    pred = {}
    for d in docs:
        pred[d] = apply_rules(texts[d], base[d])
    return pred


def write_submission(pred, texts, path):
    sub = {}
    for d, text in texts.items():
        rows = []
        for sp in pred.get(d, []):
            s, e, lab = sp[0], sp[1], sp[2]
            assert lab in LABELS and 0 <= s < e <= len(text), (d, s, e, lab)
            rows.append([int(s), int(e), lab])
        sub[d] = sorted(rows)
    assert set(sub) == set(texts)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.json", json.dumps(sub, ensure_ascii=False))
    n = 0
    for v in sub.values():
        n += len(v)
    return n


if __name__ == "__main__":
    pt = load_tokenizer()
    texts, _ = load_split(DATA_DIR, ["fold3"])
    docs = encode_docs(pt, texts)
    pred = predict_all(docs, texts, mix_runs())
    print("wrote", write_submission(pred, texts, OUT_PATH), "spans to", OUT_PATH)
