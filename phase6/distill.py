# distilled students: each student learns from the gold labels and a teacher's sentence probabilities
import os
import sys

import numpy as np

RUNS_DIR = "work/runs"

BASE = dict(epochs=2, lr_body=3e-4, max_minutes=330)
A3_PFEIFFER = dict(BASE, arch="arch3", method="pfeiffer", lr_body=1e-3)
A1_HOULSBY = dict(BASE, arch="arch1", method="houlsby", lr_body=1e-3)
A2_LORMA = dict(BASE, arch="arch2", method="lorma")
A4_LORA = dict(BASE, arch="arch4", bidirectional=True, method="lora")

# student tag: (config, teacher run)
STUDENTS = {
    "k1_arch3_pfeiffer_kd": (A3_PFEIFFER, "s1_arch2_lorma"),
    "k2_arch1_houlsby_kd": (A1_HOULSBY, "s1_arch2_lorma"),
    "k3_arch2_lorma_kd_a4": (dict(A2_LORMA, seed=2), "s2_arch4_lora"),
    "k4_arch4_lora_kd_a2": (dict(A4_LORA, seed=2), "s1_arch2_lorma"),
    "k5_arch3_pfeiffer_kd_a4": (dict(A3_PFEIFFER, seed=1), "s2_arch4_lora"),
    "k6_arch1_houlsby_kd_a4": (dict(A1_HOULSBY, seed=1), "s2_arch4_lora"),
}


def teacher_path(teacher_tag):
    return os.path.join(RUNS_DIR, teacher_tag, "scores.npz")


def load_teacher(path, train_docs):
    # teacher probabilities for every training judgment, in the order of train_docs
    t = np.load(path, allow_pickle=True)
    by_id = {}
    for i, p in zip(t["ids"], t["train"]):
        by_id[str(i)] = p
    teacher = []
    for d in train_docs:
        teacher.append(by_id[str(d["id"])])
    return teacher


if __name__ == "__main__":
    import train
    tag = sys.argv[1]
    cfg, teacher_tag = STUDENTS[tag]
    train.run(tag, cfg, teacher_path(teacher_tag))
