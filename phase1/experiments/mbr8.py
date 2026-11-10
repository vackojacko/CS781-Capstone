# Arch 2 with MBR over 8 T35 samples per prompt (my first MBR submission)
import os
import json
from generate_mbr import load_arch2, generate_many, pick_mbr, write_submission, T35, ARCH2_PATH, DATA_DIR, TEST_FILE

OUTPUT_DIR = os.path.join("outputs", "mbr8")
N_SAMPLES = 8
SEED_BASE = 2000000
SEED_STEP = 7919

if __name__ == "__main__":
    model = load_arch2(ARCH2_PATH)
    with open(os.path.join(DATA_DIR, TEST_FILE), encoding="utf-8") as f:
        test_data = json.load(f)
    test_ids = [str(k) for k in test_data.keys()]

    jobs = []
    for i, tid in enumerate(test_ids):
        seeds = [SEED_BASE + i + SEED_STEP * k for k in range(N_SAMPLES)]
        jobs.append((test_data[tid]["text"], seeds))
    outs = generate_many(model, jobs, T35)

    preds = {}
    for tid, o in zip(test_ids, outs):
        preds[tid] = {"text": pick_mbr(o)}
    write_submission(preds, OUTPUT_DIR)
    print("Wrote", len(preds), "MBR x8 predictions to", OUTPUT_DIR)
