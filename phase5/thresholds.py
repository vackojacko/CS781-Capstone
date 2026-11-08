import numpy as np

GRID = np.arange(0.02, 0.9, 0.01)


def label_f1(y, pred):
    tp = (y & pred).sum(0)
    fp = (~y & pred).sum(0)
    fn = (y & ~pred).sum(0)
    return np.where(tp > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)


def macro_f1(y, pred):
    # mean over all 100 labels, a label never predicted counts 0
    return float(label_f1(y.astype(bool), pred.astype(bool)).mean())


def at_least_one(p, pred, thr):
    pred = pred.copy()
    empty = ~pred.any(1)
    pred[empty, np.argmax(p[empty] / thr, 1)] = True
    return pred


def apply(p, thr):
    thr = thr * np.ones(p.shape[1:])
    return at_least_one(p, p > thr, np.maximum(thr, 1e-6))


def best_global(p, y):
    y = y.astype(bool)
    f = []
    for t in GRID:
        f.append(macro_f1(y, p > t))
    i = int(np.argmax(f))
    return f[i], float(GRID[i])


def per_label(p, y, shrink=50):
    # best threshold per label, pulled toward the global one for labels with few positives
    y = y.astype(bool)
    _, g = best_global(p, y)
    f = []
    for t in GRID:
        f.append(label_f1(y, p > t))
    f = np.stack(f)
    t = GRID[f.argmax(0)]
    n = y.sum(0)
    return (n * t + shrink * g) / (n + shrink)


def crossfit(p, y, seed=0, shrink=50):
    # thresholds tuned on one half of dev, scored on the other half, and swapped
    idx = np.random.RandomState(seed).permutation(len(p))
    a = idx[: len(p) // 2]
    b = idx[len(p) // 2:]
    f_global = []
    f_label = []
    for tr, te in [(a, b), (b, a)]:
        t = best_global(p[tr], y[tr])[1]
        f_global.append(macro_f1(y[te], apply(p[te], t)))
        t = per_label(p[tr], y[tr], shrink)
        f_label.append(macro_f1(y[te], apply(p[te], t)))
    return {"global": float(np.mean(f_global)), "per_label": float(np.mean(f_label))}


def report(p, y):
    y = y.astype(bool)
    r = {"f1_0.5": macro_f1(y, apply(p, 0.5))}
    r["f1_global"], r["thr_global"] = best_global(p, y)
    r["f1_per_label_dev"] = macro_f1(y, apply(p, per_label(p, y)))
    cf = crossfit(p, y)
    r["crossfit_global"] = cf["global"]
    r["crossfit_per_label"] = cf["per_label"]
    pred = apply(p, per_label(p, y))
    r["micro_f1"] = float(2 * (y & pred).sum() / max(y.sum() + pred.sum(), 1))
    r["subset_acc"] = float((y == pred).all(1).mean())
    return r
