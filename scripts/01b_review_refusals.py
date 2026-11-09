"""Section 1.2 - dump a random sample of outputs with the detector's label to a CSV,
so you can label them by hand and refine src/refusal.py.

  python scripts/01b_review_refusals.py --split val --kind harmful --n 20
"""
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_paths, load_config
from src.io_utils import read_jsonl
from src.refusal import detect_refusal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--kind", default="harmful")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    cfg = load_config()
    paths = get_paths()
    rows = read_jsonl(paths["outputs"] / "baseline" / f"{args.split}_{args.kind}.jsonl")
    if not rows:
        sys.exit("No baseline outputs found. Run scripts/01_baseline.py first.")

    random.Random(cfg["seed"]).shuffle(rows)
    window = cfg["refusal"]["window_chars"]
    out = paths["results"] / f"baseline_review_{args.split}_{args.kind}.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "prompt", "output_start", "detector_label", "matched_phrase", "manual_label"])
        for r in rows[: args.n]:
            flag, phrase = detect_refusal(r["output"], window)
            w.writerow([r["id"], r["prompt"], r["output"][:300].replace("\n", " "),
                        "refusal" if flag else "no_refusal", phrase or "", ""])
    print(f"Wrote {out}. Fill 'manual_label' (refusal / no_refusal) and compare with 'detector_label'.")


if __name__ == "__main__":
    main()
