import numpy as np
import torch
import torch.nn.functional as F


# (1 - kd_weight) CE + kd_weight tau^2 KL, the teacher is detached
def distillation_loss(student_logits, teacher_logits, labels, tau=2.0, kd_weight=0.5):
    s = student_logits.float()
    t = teacher_logits.detach().float()
    ce = F.cross_entropy(s, labels)
    kd = F.kl_div(F.log_softmax(s / tau, -1), F.log_softmax(t / tau, -1), log_target=True, reduction="batchmean")
    kd = kd * tau ** 2
    total = (1 - kd_weight) * ce + kd_weight * kd
    return total, {"ce": ce.item(), "kd": kd.item()}


def to_two_class(z):
    # each label becomes a 2-class problem with logits [0, z], so softmax is the sigmoid
    return torch.stack([torch.zeros_like(z), z], -1).reshape(-1, 2)


def kd_case_loss(z, t, y, tau, kd_weight):
    return distillation_loss(to_two_class(z), to_two_class(t), y.reshape(-1).long(), tau, kd_weight)


def check_distillation_loss():
    torch.manual_seed(0)
    s = torch.randn(8, 2, requires_grad=True)
    t = torch.randn(8, 2, requires_grad=True)
    y = torch.randint(0, 2, (8,))
    assert torch.allclose(distillation_loss(s, t, y, kd_weight=0.0)[0], F.cross_entropy(s, y))
    assert distillation_loss(s, s.detach(), y, kd_weight=1.0)[0].abs() < 1e-6
    assert torch.isfinite(distillation_loss(s * 1e4, t * 1e4, y)[0])
    distillation_loss(s, t, y)[0].backward()
    assert s.grad is not None and t.grad is None
    print("distillation checks passed")


def load_teacher_cache(paths, case_ids):
    # mean of the teachers' logits, in the order of the training cases
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
