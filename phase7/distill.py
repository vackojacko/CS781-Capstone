import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, labels, tau=2.0, kd_weight=0.5):
    # a4 q6 loss
    s = student_logits.float()
    t = teacher_logits.detach().float()
    ce = F.cross_entropy(s, labels)
    kd = F.kl_div(F.log_softmax(s / tau, -1), F.log_softmax(t / tau, -1), log_target=True, reduction="batchmean") * tau ** 2
    return (1 - kd_weight) * ce + kd_weight * kd
