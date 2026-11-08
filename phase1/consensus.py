# final submission (Codabench 0.8688): for every prompt, pick the continuation that agrees most with the
# other model outputs (mean of ROUGE-1 and ROUGE-2 F1), the same rule as MBR but across finished runs
import json
import zipfile
from collections import Counter

OUT_PATH = "submission_consensus.zip"

# retrieval prompts: candidates and extra voters
RAG_CONTINUED = "runs/rag_continued_mbr16.zip"  # retrieval fine-tune of continued Arch 2, MBR x16
RAG_ORIGINAL = "runs/rag_original_mbr16.zip"  # retrieval fine-tune of original Arch 2, MBR x16 (0.8681)
RAG_SEED82 = "runs/rag_selected_anchors_seed82.zip"  # same model, other seed and anchor choice
RAG_CONTINUED_GREEDY = "runs/rag_continued_greedy.zip"
RAG_ORIGINAL_GREEDY = "runs/rag_original_greedy.zip"

# other prompts: Arch 2 MBR runs with different pool sizes and seeds, plus the MBR x16 output (0.8402)
POOL_FILES = ["runs/pool/mbr40a.zip", "runs/pool/mbr32a.zip", "runs/pool/mbr16b.zip", "runs/pool/mbr24b.zip",
              "runs/pool/mbr20a.zip", "runs/pool/mbr24a.zip", "runs/pool/mbr32b.zip", "runs/pool/mbr40b.zip",
              "runs/pool/mbr20b.zip", "runs/pool/mbr48.zip", "runs/pool/mbr16c.zip", "runs/pool/mbr24c.zip",
              "runs/pool/mbr12c.zip", "runs/pool/mbr32c.zip", "runs/pool/mbr12b.zip", "runs/pool/mbr16a.zip",
              "runs/pool/mbr12a.zip", "runs/pool/mbr12d.zip"]
MBR16 = "runs/mbr16_t35.zip"
ARCH4 = "runs/arch4.zip"  # extra voter
ARCH2_CONTINUED = "runs/arch2_continued.zip"  # extra voter


def load(path):
    z = zipfile.ZipFile(path)
    return json.loads(z.read("prediction.json"))


def ngrams(words, n):
    grams = []
    for i in range(len(words) - n + 1):
        grams.append(tuple(words[i:i + n]))
    return Counter(grams)


def overlap(a, b):
    # mean of ROUGE-1 and ROUGE-2 F1 between two continuations
    total = 0
    for n in [1, 2]:
        x = ngrams(a.split(), n)
        y = ngrams(b.split(), n)
        common = sum((x & y).values())
        if common > 0:
            total += 2 * common / (sum(x.values()) + sum(y.values()))
    return total / 2


def unique(texts):
    out = []
    for t in texts:
        if t not in out:
            out.append(t)
    return out


def pick(candidates, voters, prefer):
    candidates = unique(candidates)
    everyone = unique(candidates + voters)
    scores = []
    for c in candidates:
        s = 0
        for v in everyone:
            if v != c:
                s += overlap(c, v)
        scores.append(s)
    best = max(scores)
    # keep the preferred output when it ties for the best score
    if prefer in candidates and scores[candidates.index(prefer)] >= best - 1e-12:
        return prefer
    return candidates[scores.index(best)]


def main():
    cont = load(RAG_CONTINUED)
    orig = load(RAG_ORIGINAL)
    s82 = load(RAG_SEED82)
    cont_g = load(RAG_CONTINUED_GREEDY)
    orig_g = load(RAG_ORIGINAL_GREEDY)
    pool = []
    for path in POOL_FILES:
        pool.append(load(path))
    mbr16 = load(MBR16)
    arch4 = load(ARCH4)
    arch2c = load(ARCH2_CONTINUED)
    mbr40 = pool[0]

    out = {}
    for t in mbr16:
        if cont[t]["text"] != mbr16[t]["text"]:
            # prompt with a retrieved passage
            cands = [cont[t]["text"], orig[t]["text"]]
            if s82[t]["text"] != mbr40[t]["text"]:
                cands.append(s82[t]["text"])
            text = pick(cands, [cont_g[t]["text"], orig_g[t]["text"]], cont[t]["text"])
        else:
            cands = []
            for p in pool:
                cands.append(p[t]["text"])
            cands.append(mbr16[t]["text"])
            text = pick(cands, [arch4[t]["text"], arch2c[t]["text"]], mbr40[t]["text"])
        assert len(text.split()) == 128
        out[t] = {"text": text}

    with zipfile.ZipFile(OUT_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("prediction.json", json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
