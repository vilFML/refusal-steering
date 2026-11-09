# Refusal steering: Prefill vs. Decoding (Qwen3-4B, Colab T4)

Compares the refusal rate of **Baseline**, **A (Prefill steering)**, **B (Decoding steering)** and **Both**
on the same held-out harmful prompts, plus a small benign check for obvious degradation.
This repo currently contains **Sections 0 and 1** (setup, splits, baseline, refusal detector).

## Folder structure

```
refusal-steering/
├── README.md
├── config.yaml              # model, generation, splits, refusal settings (single source of truth)
├── requirements.txt         # pinned deps (torch comes from Colab)
├── .gitignore               # raw outputs are NOT committed (may contain harmful text)
├── src/
│   ├── config.py            # load config, artifact paths, seeding
│   ├── env.py               # GPU / version check and logging
│   ├── io_utils.py          # JSON / JSONL helpers
│   ├── data.py              # dataset loading, disjoint splits
│   ├── model.py             # load Qwen3-4B (fp16, sdpa), chat template
│   ├── generation.py        # batched greedy generation, checkpointing, OOM backoff
│   └── refusal.py           # keyword refusal detector + summary stats
├── scripts/
│   ├── 00_make_splits.py    # Section 0.4
│   ├── 01_baseline.py       # Section 1.1 / 1.3 (+ --smoke)
│   └── 01b_review_refusals.py   # Section 1.2 manual review sheet
├── notebooks/
│   ├── 00_setup_and_splits.ipynb
│   └── 01_baseline.ipynb
├── tests/                   # offline tests (no GPU needed)
├── data/                    # splits.json            (Drive when using Colab)
├── vectors/                 # refusal direction      (Section 2)
├── outputs/                 # raw generations, one folder per condition
└── results/                 # metrics JSON, review CSVs, env.json
```

Code lives in Git; heavy/private artifacts (`data/ vectors/ outputs/ results/`) live on Drive in Colab
via the `STEERING_DATA_ROOT` environment variable (set by the notebooks). Without it, they go inside the repo.

## Quick start (Colab T4)

1. Push this repo to GitHub and put your URL in `REPO_URL` (first cell of each notebook).
2. Open `notebooks/00_setup_and_splits.ipynb` in Colab, select **T4 GPU**, run all cells.
3. Open `notebooks/01_baseline.ipynb` and run in order: smoke test, baseline on val harmful, detector review.

Local (CPU only, for the tests): `pip install pyyaml pytest && python -m pytest -q`

## Design decisions

- **Fixed generation:** Qwen3 chat template with `enable_thinking=False`, greedy decoding, `max_new_tokens=200`.
- **fp16 + sdpa:** the T4 has no native bf16 or FlashAttention-2. `--smoke` detects fp16 overflow early.
- **Resumable:** each batch is appended to `outputs/<condition>/<split>_<kind>.jsonl`; re-running skips finished ids.
- **Refusal is computed after generation**, so improving the detector never requires regenerating.
- **Test split is locked:** `01_baseline.py --split test` refuses to run without `--allow-test` (Section 4 only).
- **Splits are fixed by seed and saved**; `00_make_splits.py` will not overwrite them without `--force`.

## Gates

| After | Check |
|---|---|
| Section 0 | Splits created, sizes as in config, disjointness verified |
| Section 1 | Smoke test sane, val-harmful refusal rate >= 70%, detector matches manual labels |

## Ethics

Raw generations can contain harmful content: keep `outputs/` private (it is git-ignored) and only report
aggregated numbers or redacted examples.
