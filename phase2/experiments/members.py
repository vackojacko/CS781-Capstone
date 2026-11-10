# settings of the runs that are only used inside the ensembles, trained with train.py
# python -m experiments.members <run name>
import sys

import train

MEMBER_RUNS = {
    "arch4_full_h128_s1": train.run_config("arch4", "full", 8, 1e-4, 4, seed=1, repair=False),
    "arch2_full_h256_s1": train.run_config("arch2", "full", 8, 1e-4, 8, header=256, seed=1, repair=False),
}


if __name__ == "__main__":
    name = sys.argv[1]
    train.main(name, MEMBER_RUNS[name])
