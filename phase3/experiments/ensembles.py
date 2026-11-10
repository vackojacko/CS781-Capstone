# the other ensemble and decoding submissions, built from saved sentence probabilities
# python -m experiments.ensembles            (all of them)
# python -m experiments.ensembles 07_kdov6_rr1_2_x2_2 bias_none_up
import os
import sys

import numpy as np

from data import DATA_DIR, LABELS, load_docs, transition_matrix
from decode import macro_f1, transition_viterbi
from predict import ensemble, load_probs, write_submission
from rules import dissent_rule

RUNS_DIR = "runs"
OUT_DIR = RUNS_DIR + "/experiments"

DISSENT = LABELS.index("Dissent")
OVERRULED = LABELS.index("PrecedentOverruled")
# the first submissions never predicted Dissent and PrecedentOverruled, later only PrecedentOverruled
NEVER_BOTH = [DISSENT, OVERRULED]
NEVER_OVERRULED = [OVERRULED]
NEVER_NONE = []

NEW6 = ["arch2_kd_s1", "arch2_kd_s2", "arch2_kd_s3", "arch4_lora_s1", "arch4_lora_s2", "student"]

MEMBER_FILES = {
    "rr1": RUNS_DIR + "/rr1_{}_probs.npz",
    "x2": RUNS_DIR + "/x2_{}_probs.npz",
    "kd": RUNS_DIR + "/arch2_kd_{}_probs.npz",
    "kdov": RUNS_DIR + "/arch2_kd_{}_probs_centered.npz",
    "lorma": RUNS_DIR + "/rr2b_{}_probs.npz",
    "lormaov": RUNS_DIR + "/rr2b_{}_probs_centered.npz",
    "rerun": RUNS_DIR + "/rr1_frozen_{}_probs.npz",
    "b3": RUNS_DIR + "/b3_{}_probs_centered.npz",
    "b4": RUNS_DIR + "/b4_{}_probs_centered.npz",
    "fw": RUNS_DIR + "/fixed_weights_{}_probs_centered.npz",
    "r1c": RUNS_DIR + "/refit_{}_probs_centered.npz",
    "r1l": RUNS_DIR + "/refit_{}_probs_left.npz",
}
for name in NEW6:
    MEMBER_FILES[name] = RUNS_DIR + "/big_" + name + "_{}_probs_centered.npz"
SHIFT_FILE = RUNS_DIR + "/arch2_kd_{}_shift.npz"

EM_ITERS = 10
EM_MIX = 0.5

# grid for the CL blend of the per-domain submission (weights of kdov, rr1 + x2, the six big-run models)
GRID_KD = [0, 0.15, 0.3]
GRID_RR = [0, 0.2, 0.3]
GRID_NEW = [0, 0.2, 0.35, 0.5]
GRID_BETA = [0.2, 0.3, 0.4]


def make(name, weights, decode="argmax", beta=0.3, never=NEVER_BOTH, trans="train", dissent=False, combine="mean",
         prior=False, bias=None, smooth=None, blocks=None, weights_it=None, beta_it=None):
    if weights_it is None:
        weights_it = weights
    if beta_it is None:
        beta_it = beta
    return {"name": name, "weights": weights, "weights_it": weights_it, "decode": decode, "beta": beta, "beta_it": beta_it,
            "never": never, "trans": trans, "dissent": dissent, "combine": combine, "prior": prior, "bias": bias,
            "smooth": smooth, "blocks": blocks}


def equal(names):
    w = {}
    for n in names:
        w[n] = 1.0 / len(names)
    return w


def add_new6(weights, total):
    w = dict(weights)
    for n in NEW6:
        w[n] = total / len(NEW6)
    return w


def normalise(weights):
    total = sum(weights.values())
    w = {}
    for k in weights:
        w[k] = weights[k] / total
    return w


TRIO = {"kdov": 0.6, "rr1": 0.2, "x2": 0.2}

SUBMISSIONS = [
    # first ensembles, argmax of the averaged probabilities
    make("submission_ens_rr1_x2", {"rr1": 0.5, "x2": 0.5}, combine="raw"),
    make("submission_ens_rr1_x2_rerun", equal(["rr1", "x2", "rerun"]), combine="raw"),
    make("submission_ens_rr1_kd", {"rr1": 0.5, "kd": 0.5}),
    make("submission_ens_rr1_x2_kd", equal(["rr1", "x2", "kd"])),
    # one label shifted up or down in log probability
    make("bias_argpet_up", {"rr1": 0.5, "x2": 0.5}, bias={"ArgumentPetitioner": 0.3}, combine="raw"),
    make("bias_lowercourt_up", {"rr1": 0.5, "x2": 0.5}, bias={"RulingByLowerCourt": 0.5}, combine="raw"),
    make("bias_none_up", {"rr1": 0.5, "x2": 0.5}, bias={"None": 1.5}, combine="raw"),
    make("bias_ratio_down", {"rr1": 0.5, "x2": 0.5}, bias={"RatioOfTheDecision": -0.5}, combine="raw"),
    # distilled arch2 with overlapping windows and its ensembles
    make("01_kdov_argmax", {"kdov": 1.0}),
    make("02_kdov7_rr1_3", {"kdov": 0.7, "rr1": 0.3}, combine="raw"),
    make("07_kdov6_rr1_2_x2_2", TRIO),
    make("13_geo_kdov7_rr1_3", {"kdov": 0.7, "rr1": 0.3}, combine="geo"),
    make("16_kdov6_rr1_2_x2_2_transvit03", TRIO, decode="viterbi", beta=0.3),
    make("18_kdov_em_transvit05", {"kdov": 1.0}, decode="viterbi", beta=0.5, prior=True),
    make("19_kdov5_rr1_5_em", {"kdov": 0.5, "rr1": 0.5}, prior=True),
    # weights and transition strength
    make("1_kdov6_rr1_x2_lorma_transvit04", {"kdov": 0.6, "rr1": 0.133, "x2": 0.133, "lorma": 0.134}, decode="viterbi", beta=0.4),
    make("2_kdov7_rr1_15_x2_15_transvit05", {"kdov": 0.7, "rr1": 0.15, "x2": 0.15}, decode="viterbi", beta=0.5),
    make("3_kdov5_rr1_x2_lorma_transvit02", {"kdov": 0.5, "rr1": 0.167, "x2": 0.167, "lorma": 0.166}, decode="viterbi", beta=0.2),
    make("4_kdov6_rr1_2_x2_2_transvit04", TRIO, decode="viterbi", beta=0.4),
    make("5_kdov6_rr1_2_x2_2_transvit02", TRIO, decode="viterbi", beta=0.2),
    make("6_kdov5_rr1_25_x2_25_transvit03", {"kdov": 0.5, "rr1": 0.25, "x2": 0.25}, decode="viterbi", beta=0.3),
    # b4 (contrastive, trained on train + dev), transitions counted on train + dev
    make("1_b4_transvit03", {"b4": 1.0}, decode="viterbi", trans="train_dev"),
    make("3_b4_6_rr1_2_x2_2_transvit03", {"b4": 0.6, "rr1": 0.2, "x2": 0.2}, decode="viterbi", trans="train_dev"),
    make("4_b4_3_kdov_3_rr1_2_x2_2_transvit03", {"b4": 0.3, "kdov": 0.3, "rr1": 0.2, "x2": 0.2}, decode="viterbi",
         trans="train_dev", combine="raw"),
    make("5_b4_5_kdov_5_transvit03", {"b4": 0.5, "kdov": 0.5}, decode="viterbi", trans="train_dev"),
    # smoothing inside the blocks predicted by the shift head
    make("1_ens_shiftsmooth09", TRIO, smooth=0.9),
    make("2_ens_shiftsmooth07", TRIO, smooth=0.7),
    make("3_ens_shiftblocks02", TRIO, blocks=0.2),
    # b3 (contrastive + prototypes) in the ensemble
    make("1_b340_kdov30_rr115_x215_transvit03", {"b3": 0.4, "kdov": 0.3, "rr1": 0.15, "x2": 0.15}, decode="viterbi"),
    make("2_b330_kdov30_rr120_x220_transvit03", {"b3": 0.3, "kdov": 0.3, "rr1": 0.2, "x2": 0.2}, decode="viterbi",
         combine="raw"),
    # dissent allowed, no rule yet
    make("best_with_dissent_allowed", TRIO, decode="viterbi", never=NEVER_OVERRULED),
    # bigger ensembles with the dissent rule
    make("1_half_oldbest_half_new6_transvit_dissent", add_new6({"kdov": 0.3, "rr1": 0.1, "x2": 0.1}, 0.5), decode="viterbi",
         never=NEVER_OVERRULED, dissent=True),
    make("2_all11_equal_transvit_dissent", equal(NEW6 + ["kdov", "rr1", "x2", "b3", "lormaov"]), decode="viterbi",
         never=NEVER_OVERRULED, dissent=True),
    make("3_new6_kdov_rr1_x2_transvit_dissent", equal(NEW6 + ["kdov", "rr1", "x2"]), decode="viterbi", never=NEVER_OVERRULED,
         dissent=True),
    # separate blends for CL and IT, the CL blend is picked on the CL dev documents
    make("5_perdomain_blend", "cl_grid", decode="viterbi", never=NEVER_NONE, dissent=True,
         weights_it=add_new6({"kdov": 0.3}, 0.7), beta_it=0.3),
    # refit on train + dev, both context settings
    make("4_r1_alone", {"r1c": 0.5, "r1l": 0.5}, decode="viterbi", never=NEVER_NONE, trans="domain", dissent=True),
    make("old_trio_perdomain_cl03", TRIO, decode="viterbi", never=NEVER_NONE, trans="domain", dissent=True),
    # fixed class weighting
    make("1_trio_plus_fixedweights", normalise({"kdov": 0.45, "fw": 0.25, "rr1": 0.15, "x2": 0.15}), decode="viterbi",
         never=NEVER_NONE, dissent=True),
    make("2_fixedweights_replaces_kd", {"fw": 0.6, "rr1": 0.2, "x2": 0.2}, decode="viterbi", never=NEVER_NONE, dissent=True),
    make("3_kd_fixedweights_rr1_x2_equal_arch2", normalise({"kdov": 0.3, "fw": 0.3, "rr1": 0.2, "x2": 0.2}), decode="viterbi",
         never=NEVER_NONE, dissent=True),
    make("4_trio_fixedweights_new6", normalise(add_new6({"kdov": 0.25, "fw": 0.2, "rr1": 0.1, "x2": 0.1}, 0.35)),
         decode="viterbi", never=NEVER_NONE, dissent=True),
]


def load_member(name, split, docs, cache):
    key = name + "_" + split
    if key not in cache:
        p = load_probs(MEMBER_FILES[name].format(split))
        cache[key] = []
        for d in docs:
            cache[key].append(np.asarray(p[d["id"]], dtype=np.float64))
    return cache[key]


def load_shift(split, docs):
    s = load_probs(SHIFT_FILE.format(split))
    out = []
    for d in docs:
        out.append(np.asarray(s[d["id"]], dtype=np.float64))
    return out


def geometric(P, weights, i):
    x = 0
    for k in weights:
        x = x + weights[k] * np.log(np.clip(P[k][i], 1e-9, 1.0))
    x = np.exp(x - x.max(axis=1, keepdims=True))
    return x / x.sum(axis=1, keepdims=True)


def raw_average(P, weights, i):
    # the first ensembles averaged the stored probabilities without normalising each member first
    x = 0
    for k in weights:
        x = x + weights[k] * P[k][i]
    return x


def label_prior(train_docs):
    counts = np.ones(len(LABELS))
    for d in train_docs:
        for y in d["labels"]:
            counts[y] += 1
    return counts / counts.sum()


def em_prior(p, prior):
    # saerens et al 2002: re-estimate the label prior of the document, half way back to the training prior
    q = prior.copy()
    x = p
    for it in range(EM_ITERS):
        x = p * (q / prior)
        x = x / x.sum(axis=1, keepdims=True)
        q = x.mean(axis=0) * EM_MIX + prior * (1 - EM_MIX)
    return x


def soft_blocks(p, s, alpha):
    # mix each sentence with its neighbours unless the shift head predicts a label change between them
    f = p.copy()
    b = p.copy()
    for i in range(1, len(p)):
        f[i] = (1 - alpha) * p[i] + alpha * (1 - s[i - 1]) * f[i - 1] + alpha * s[i - 1] * p[i]
    for i in range(len(p) - 2, -1, -1):
        b[i] = (1 - alpha) * p[i] + alpha * (1 - s[i]) * b[i + 1] + alpha * s[i] * p[i]
    x = (f + b) / 2
    return x / x.sum(axis=1, keepdims=True)


def hard_blocks(p, s, threshold):
    out = p.copy()
    start = 0
    for i in range(len(p)):
        if i == len(p) - 1 or s[i] > threshold:
            out[start:i + 1] = p[start:i + 1].mean(axis=0)
            start = i + 1
    return out


def argmax_decode(p, never, bias):
    s = np.log(np.clip(p, 1e-9, 1.0))
    if bias is not None:
        for label in bias:
            s[:, LABELS.index(label)] += bias[label]
    s[:, never] = -1e9
    return s.argmax(axis=1).tolist()


def viterbi_decode(p, T, beta, never):
    p = p.copy()
    p[:, never] = 0
    return transition_viterbi(p, T, beta)


def best_cl_blend(train_docs, dev_docs, cache):
    T = transition_matrix(train_docs)
    idx = []
    for i in range(len(dev_docs)):
        if dev_docs[i]["domain"] == "CL":
            idx.append(i)
    gold = []
    for i in idx:
        gold.append(dev_docs[i]["labels"])
    P = {}
    for name in ["kdov", "rr1", "x2"] + NEW6:
        P[name] = load_member(name, "dev", dev_docs, cache)
    best = (-1, None, None)
    for kd in GRID_KD:
        for rr in GRID_RR:
            for new in GRID_NEW:
                total = kd + rr + new
                if total == 0:
                    continue
                w = {}
                if kd > 0:
                    w["kdov"] = kd / total
                if rr > 0:
                    w["rr1"] = rr / 2 / total
                    w["x2"] = rr / 2 / total
                if new > 0:
                    for n in NEW6:
                        w[n] = new / len(NEW6) / total
                for beta in GRID_BETA:
                    preds = []
                    for i in idx:
                        preds.append(transition_viterbi(ensemble(P, w, i), T, beta))
                    f1 = macro_f1(gold, preds)
                    if f1 > best[0]:
                        best = (f1, w, beta)
    print("cl blend on dev:", round(best[0], 4), "beta", best[2], best[1])
    return best[1], best[2]


def run(sub, docs, split, matrices, cache, prior):
    names = set(sub["weights"]) | set(sub["weights_it"])
    P = {}
    for name in names:
        P[name] = load_member(name, split, docs, cache)
    shift = None
    if sub["smooth"] is not None or sub["blocks"] is not None:
        shift = load_shift(split, docs)
    preds = []
    for i in range(len(docs)):
        d = docs[i]
        if d["domain"] == "CL":
            w = sub["weights"]
            beta = sub["beta"]
        else:
            w = sub["weights_it"]
            beta = sub["beta_it"]
        if sub["combine"] == "geo":
            x = geometric(P, w, i)
        elif sub["combine"] == "raw":
            x = raw_average(P, w, i)
        else:
            x = ensemble(P, w, i)
        if sub["prior"]:
            x = em_prior(x, prior)
        if sub["smooth"] is not None:
            x = soft_blocks(x, shift[i], sub["smooth"])
        if sub["blocks"] is not None:
            x = hard_blocks(x, shift[i], sub["blocks"])
        if sub["decode"] == "viterbi":
            if sub["trans"] == "domain":
                T = matrices[d["domain"]]
            else:
                T = matrices[sub["trans"]]
            y = viterbi_decode(x, T, beta, sub["never"])
        else:
            y = argmax_decode(x, sub["never"], sub["bias"])
        if sub["dissent"]:
            y = dissent_rule(d, y)
        preds.append(y)
    return preds


def main(names):
    train_docs = load_docs(DATA_DIR, "train")
    dev_docs = load_docs(DATA_DIR, "dev")
    test_docs = load_docs(DATA_DIR, "test")
    matrices = {"train": transition_matrix(train_docs), "train_dev": transition_matrix(train_docs + dev_docs)}
    for domain in ["CL", "IT"]:
        docs = []
        for d in train_docs:
            if d["domain"] == domain:
                docs.append(d)
        matrices[domain] = transition_matrix(docs)
    prior = label_prior(train_docs)
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = {}
    for sub in SUBMISSIONS:
        if len(names) > 0 and sub["name"] not in names:
            continue
        if sub["weights"] == "cl_grid":
            sub = dict(sub)
            sub["weights"], sub["beta"] = best_cl_blend(train_docs, dev_docs, cache)
        dev_f1 = macro_f1([d["labels"] for d in dev_docs], run(sub, dev_docs, "dev", matrices, cache, prior))
        preds = run(sub, test_docs, "test", matrices, cache, prior)
        path = OUT_DIR + "/" + sub["name"] + ".zip"
        write_submission(test_docs, preds, path)
        print(f"{sub['name']} | dev macro F1 {dev_f1:.4f} | wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
