# A4 q6 skeleton (knowledge distillation), completed. teacher logits are cached once before the student trains
import numpy as np
import torch
import torch.nn.functional as F

# written by train.py for e1, e6 and e2 (teacher_n = 16000), in this order
TEACHER_FILES = ["runs/e1_arch4_lorma/teacher_train.npz", "runs/e6_arch4_lorma_tail3_s1/teacher_train.npz",
                 "runs/e2_arch2_unfreeze/teacher_train.npz"]


def distillation_loss(student_logits, teacher_logits, labels, tau=2.0, kd_weight=0.5):
    s = student_logits.float()
    t = teacher_logits.detach().float()
    ce = F.cross_entropy(s, labels)
    kd = F.kl_div(F.log_softmax(s / tau, -1), F.log_softmax(t / tau, -1), log_target=True, reduction="batchmean")
    kd = kd * tau ** 2
    total = (1 - kd_weight) * ce + kd_weight * kd
    return total, {"ce": ce.item(), "kd": kd.item()}


def check_distillation_loss():
    torch.manual_seed(0)
    s = torch.randn(8, 2, requires_grad=True)
    t = torch.randn(8, 2, requires_grad=True)
    y = torch.randint(0, 2, (8,))
    assert torch.allclose(distillation_loss(s, t, y, kd_weight=0.0)[0], F.cross_entropy(s, y))
    assert distillation_loss(s, s.detach(), y, kd_weight=1.0)[0].abs() < 1e-6
    big = distillation_loss(s * 1e4, t * 1e4, y)[0]
    assert torch.isfinite(big)
    distillation_loss(s, t, y)[0].backward()
    assert s.grad is not None and t.grad is None
    print("distillation checks passed")


def save_teacher_cache(path, case_ids, logits):
    # logits of the last chunk and the one before it (the last chunk again for one-chunk cases)
    rows = []
    for l in logits:
        if len(l) > 1:
            rows.append(np.stack([l[-1], l[-2]]))
        else:
            rows.append(np.stack([l[-1], l[-1]]))
    np.savez(path, case_ids=np.array(case_ids), logits=np.stack(rows).astype(np.float32))


def load_teacher_cache(paths, case_ids):
    # average of the teachers' chunk logits, every training case present exactly once
    total = None
    for p in paths:
        z = np.load(p)
        ids = [str(i) for i in z["case_ids"]]
        assert len(ids) == len(set(ids)) and ids == list(case_ids)
        if total is None:
            total = z["logits"].astype(np.float64)
        else:
            total = total + z["logits"]
    return total / len(paths)
