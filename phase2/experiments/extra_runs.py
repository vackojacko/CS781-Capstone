# settings of the other single model phase 2 submissions, trained with train.py
# python -m experiments.extra_runs <run name>
import sys

import train

EXTRA_RUNS = {
    "arch2_full_h128_bidir": train.run_config("arch2", "full", 8, 1e-4, 8, bidir=True, repair=False),
    "arch4_lora_h128": train.run_config("arch4", "lora", 10, 5e-4, 4),
}


if __name__ == "__main__":
    name = sys.argv[1]
    train.main(name, EXTRA_RUNS[name])
