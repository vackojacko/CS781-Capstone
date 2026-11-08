# knowledge distillation: arch 2 student (61.6M) from the fine-tuned arch 4 teacher (139M)
# loss = 0.5 * T^2 KL(teacher || student) at T = 2 + 0.5 * cross-entropy on gold tags
import os

import model as M
from train import DEVICE, OUT_DIR, main, run_config

TEACHER_RUN = "arch4_full_h128"
TEACHER_ARCH = "arch4"
KD_ALPHA = 0.5
KD_TEMP = 2.0

# no cross-validation: the teacher was trained on both folds
STUDENT = run_config("arch2", "full", 4, 1e-4, 8, run_cv=False)


if __name__ == "__main__":
    M.HEADER = STUDENT["header"]
    teacher = M.load_classifier(TEACHER_ARCH, M.MODEL_DIRS[TEACHER_ARCH],
                                os.path.join(OUT_DIR, TEACHER_RUN, "model_final.pt"))
    teacher = teacher.to(DEVICE).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    main("arch2_full_h128_kd", STUDENT, teacher=teacher, kd_alpha=KD_ALPHA, kd_temp=KD_TEMP)
