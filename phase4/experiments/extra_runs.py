# the other phase 4 runs: python -m experiments.extra_runs <run>
# e7 is arch 2 trained on the last 3 chunks, e13 and e14 are arch 4 seeds trained on the last 3 chunks,
# e15 to e20 are arch 4 seeds trained on the last 2 chunks
import sys

from train import E1, E2, run

E7 = dict(E2, tag="e7_arch2_unfreeze_tail3_s1", last_k=3, seed=1)
E13 = dict(E1, tag="e13_arch4_lorma_tail3_s4", last_k=3, seed=4, teacher_n=None)
E14 = dict(E1, tag="e14_arch4_lorma_tail3_s5", last_k=3, seed=5, teacher_n=None)
SEEDS = [6, 7, 8, 9, 10, 11]

CONFIGS = {}
for c in [E7, E13, E14]:
    CONFIGS[c["tag"]] = c
for seed in SEEDS:
    tag = "e" + str(seed + 9) + "_arch4_lorma_s" + str(seed)
    CONFIGS[tag] = dict(E1, tag=tag, seed=seed, teacher_n=None)


if __name__ == "__main__":
    run(CONFIGS[sys.argv[1]])
