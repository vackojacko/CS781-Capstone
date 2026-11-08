# gathers runs/<tag>/result.json into results.json and results_table.md
import os
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "runs")

configs = json.load(open(os.path.join(HERE, "configs.json")))
results = {}
for c in configs:
    p = os.path.join(OUTPUT_DIR, c["tag"], "result.json")
    if os.path.exists(p):
        results[c["tag"]] = json.load(open(p))

json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1)

lines = []
for group in ["A", "B"]:
    # baselines are the runs whose tag starts with A0 or B0
    base_by_seed = {}
    for c in configs:
        if c["group"] == group and c["tag"].startswith(group + "0") and c["tag"] in results:
            base_by_seed[c.get("seed", 42)] = results[c["tag"]]["final_val_loss"]
    if not base_by_seed:
        continue
    vals = list(base_by_seed.values())
    mean = sum(vals) / len(vals)
    std = 0.0
    if len(vals) > 1:
        sq = 0.0
        for v in vals:
            sq += (v - mean) ** 2
        std = math.sqrt(sq / (len(vals) - 1))
    n_seeds = len(vals)
    spread = max(vals) - min(vals)
    lines.append("")
    lines.append(f"Group {group}: baseline over {n_seeds} seeds, mean val loss {mean:.4f}, seed std {std:.4f}, "
                 f"range {spread:.4f}")
    lines.append("")
    lines.append("| Run | Seed | Change | Params | Val loss | Val PPL | Delta vs same-seed baseline | tok/s | Train min |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for c in configs:
        if c["group"] != group or c["tag"] not in results:
            continue
        r = results[c["tag"]]
        seed = c.get("seed", 42)
        d = "-"
        if seed in base_by_seed and not c["tag"].startswith(group + "0"):
            delta = r["final_val_loss"] - base_by_seed[seed]
            d = f"{delta:+.4f}"
        flag = ""
        if r["stopped_early"]:
            flag = " (time cap)"
        tag = c["tag"]
        change = c["change"]
        params = r["params"] / 1e6
        loss = r["final_val_loss"]
        ppl = r["final_val_ppl"]
        tok_s = r["tokens_per_sec"]
        minutes = r["train_minutes"]
        lines.append(f"| {tag} | {seed} | {change} | {params:.1f}M | {loss:.4f} | {ppl:.3f} | {d} | {tok_s:.0f} | "
                     f"{minutes:.1f}{flag} |")

lines.append("")
lines.append("Validation loss at fixed checkpoints (tokens seen in millions)")
for c in configs:
    if c["tag"] not in results:
        continue
    parts = []
    for h in results[c["tag"]]["history"]:
        tokens_m = h["tokens"] / 1e6
        loss = h["val_loss"]
        parts.append(f"{tokens_m:.1f}M:{loss:.3f}")
    lines.append("- " + c["tag"] + ": " + " ".join(parts))

text = "\n".join(lines)
open(os.path.join(HERE, "results_table.md"), "w").write(text + "\n")
print(text)
