# knowledge distillation into the Arch 2 student (62M)
# python distill.py teacher | student
import os
import sys

import numpy as np
import torch

import backbone as B
from model import build_model
from predict import RUN_FILES, load_probs, save_probs
from train import OUT_DIR, load_splits, save_run, train

TEACHER_PATH = OUT_DIR + "/teacher_rr1_x2_train_probs.npz"
ARCH2_DIR = os.path.join(B.KIT_DIR, "arch2")
STUDENT_EPOCHS = 6
LR_BODY = 5e-4


def make_teacher(train_docs):
    # teacher is the average of the two best arch4 runs (rr1 and x2) on the training docs
    a = load_probs(RUN_FILES["rr1"].format("train"))
    b = load_probs(RUN_FILES["x2"].format("train"))
    probs = []
    for d in train_docs:
        p = (a[d["id"]] + b[d["id"]]) / 2
        # the teacher file was first stored in float16, then normalised again
        p = p.astype(np.float16).astype(np.float64)
        probs.append(p / p.sum(axis=1, keepdims=True))
    save_probs(TEACHER_PATH, train_docs, probs)


def run_student(device):
    # arch2_kd: arch2 + lora distilled from the rr1 + x2 teacher, Codabench 0.5354
    train_docs, dev_docs, test_docs = load_splits(B.load_tokenizer())
    soft = load_probs(TEACHER_PATH)
    torch.manual_seed(0)
    model = build_model("arch2", ARCH2_DIR, "lora", False, context="none").to(device)
    train(model, train_docs, dev_docs, device, epochs=STUDENT_EPOCHS, lr_body=LR_BODY, soft=soft, seed=0)
    save_run(model, "arch2_kd", [("train", train_docs), ("dev", dev_docs), ("test", test_docs)])
    torch.save(model.state_dict(), OUT_DIR + "/arch2_kd_model.pt")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    step = sys.argv[1]
    if step == "teacher":
        make_teacher(load_splits(B.load_tokenizer())[0])
    elif step == "student":
        run_student(device)
