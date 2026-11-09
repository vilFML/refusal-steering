"""Section 0.4 - build disjoint train/val/test splits and save them to data/splits.json."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_paths, load_config, set_seed
from src.data import build_splits, load_prompts, save_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing splits.json")
    args = ap.parse_args()

    cfg = load_config()
    paths = get_paths()
    set_seed(cfg["seed"])
    out = paths["data"] / "splits.json"
    if out.exists() and not args.force:
        print(f"{out} already exists; use --force to rebuild. Splits must stay fixed across runs.")
        return

    d = cfg["data"]
    harmful = load_prompts(d["harmful_dataset"])
    harmless = load_prompts(d["harmless_dataset"])
    print(f"Loaded {len(harmful)} harmful and {len(harmless)} harmless prompts")

    splits = build_splits(harmful, harmless, d["sizes"], cfg["seed"])
    save_splits(splits, out, meta={"seed": cfg["seed"], "sizes": d["sizes"],
                                   "harmful_dataset": d["harmful_dataset"],
                                   "harmless_dataset": d["harmless_dataset"]})
    for s, kinds in splits.items():
        print(f"  {s}: " + ", ".join(f"{k}={len(v)}" for k, v in kinds.items()))
    print(f"Saved {out} (disjointness verified)")


if __name__ == "__main__":
    main()
