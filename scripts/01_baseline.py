"""Section 1 - run the unmodified model and compute its refusal rate.

Examples
  python scripts/01_baseline.py --smoke                      # 3 prompts, checks fp16 output is sane
  python scripts/01_baseline.py --split val --kind harmful   # Section 1.3 (gate before Section 2)
  python scripts/01_baseline.py --split test --kind harmful --allow-test   # only in Section 4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_paths, load_config, set_seed
from src.data import load_split
from src.env import check_environment
from src.generation import make_generate_fn, run_generation
from src.io_utils import write_json
from src.model import load_model_and_tokenizer
from src.refusal import detect_refusal, summarize

CONDITION = "baseline"


def smoke_test(cfg, paths, generate_fn):
    """Catch fp16 overflow / broken template early: outputs must be non-empty, non-degenerate text."""
    prompts = [r["text"] for r in load_split(paths, "val", "harmful")[:2]] \
            + [r["text"] for r in load_split(paths, "val", "harmless")[:1]]
    for p, res in zip(prompts, generate_fn(prompts)):
        out = res["output"]
        degenerate = len(out.strip()) == 0 or len(set(out)) < 5
        print(f"\nPROMPT: {p[:80]}\nOUTPUT: {out[:200]}\n=> {'SUSPICIOUS' if degenerate else 'ok'}")
        if degenerate:
            print("   Possible fp16 overflow. Try dtype: bfloat16 in config.yaml (slow on T4) or check the template.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--kind", choices=["harmful", "harmless"], default="harmful")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N prompts")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--allow-test", action="store_true", help="required to touch the test split")
    args = ap.parse_args()

    if args.split == "test" and not args.allow_test and not args.smoke:
        sys.exit("Test split is locked until Section 4. Pass --allow-test if this is intentional.")

    cfg = load_config()
    paths = get_paths()
    set_seed(cfg["seed"])
    check_environment(save_to=paths["results"] / "env.json")

    model, tok = load_model_and_tokenizer(cfg)
    generate_fn = make_generate_fn(model, tok, cfg)

    if args.smoke:
        smoke_test(cfg, paths, generate_fn)
        return

    records = load_split(paths, args.split, args.kind)
    if args.limit:
        records = records[: args.limit]

    out_path = paths["outputs"] / CONDITION / f"{args.split}_{args.kind}.jsonl"
    meta = {"condition": CONDITION, "split": args.split, "kind": args.kind}
    rows = run_generation(records, out_path, generate_fn, cfg["generation"]["batch_size"], meta)

    # Refusal is computed after generation, so refining the detector never needs a re-run.
    window = cfg["refusal"]["window_chars"]
    stats = summarize(rows, window)
    stats.update(meta)
    stats["model"] = cfg["model"]["name"]
    stats["max_new_tokens"] = cfg["generation"]["max_new_tokens"]
    stats["window_chars"] = window
    result_path = paths["results"] / f"{CONDITION}_{args.split}_{args.kind}.json"
    write_json(result_path, stats)

    print(f"\nRefusal rate ({CONDITION}, {args.split}, {args.kind}): "
          f"{stats['n_refusals']}/{stats['n']} = {stats['refusal_rate']:.1%}")
    print(f"empty={stats['n_empty']}  truncated={stats['n_truncated']}  "
          f"mean_new_tokens={stats['mean_new_tokens']:.0f}")
    print(f"Saved {result_path}")

    if args.kind == "harmful" and stats["refusal_rate"] < cfg["refusal"]["sanity_min_rate"]:
        print("\nGATE FAILED: the baseline should refuse most harmful prompts "
              f"(>= {cfg['refusal']['sanity_min_rate']:.0%}). Check, in this order: "
              "(1) enable_thinking=False in config, (2) chat template applied, "
              "(3) refusal phrase list (run scripts/01b_review_refusals.py), (4) fp16 outputs (--smoke).")


if __name__ == "__main__":
    main()
