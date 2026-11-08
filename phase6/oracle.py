# greedy rouge oracle (nallapati et al 2017): sentence labels from the reference summary
from collections import Counter

REF_WORDS = 420


def ngrams(words, n):
    grams = Counter()
    for i in range(len(words) - n + 1):
        grams[tuple(words[i:i + n])] += 1
    return grams


def overlap_f1(cand, ref):
    # rouge-n f1 from two n-gram counters
    hit = sum((cand & ref).values())
    if hit == 0:
        return 0.0
    p = hit / sum(cand.values())
    r = hit / sum(ref.values())
    return 2 * p * r / (p + r)


def oracle_labels(sents, summary, max_words=500):
    ref = summary.lower().split()[:REF_WORDS]
    ref1 = ngrams(ref, 1)
    ref2 = ngrams(ref, 2)
    one, two, size = [], [], []
    for s in sents:
        w = s.lower().split()
        one.append(ngrams(w, 1))
        two.append(ngrams(w, 2))
        size.append(len(w))
    sel1, sel2 = Counter(), Counter()
    chosen = []
    best = 0.0
    used = 0
    while used < max_words:
        pick, pick_score = -1, best
        for i in range(len(sents)):
            if i in chosen:
                continue
            score = overlap_f1(sel1 + one[i], ref1) + overlap_f1(sel2 + two[i], ref2)
            if score > pick_score:
                pick, pick_score = i, score
        if pick < 0:
            break
        chosen.append(pick)
        sel1 = sel1 + one[pick]
        sel2 = sel2 + two[pick]
        best = pick_score
        used += size[pick]
    labels = [0] * len(sents)
    for i in chosen:
        labels[i] = 1
    return labels
