"""Dataset loading and disjoint train / val / test splits (Section 0.4)."""
import hashlib
import random

from src.io_utils import read_json, write_json

TEXT_COLUMNS = ("text", "prompt", "instruction")
KINDS = ("harmful", "harmless")


def make_id(kind, text):
    """Stable id derived from the text, independent of dataset order."""
    return f"{kind}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def load_prompts(dataset_name):
    """Download a HF dataset and return all prompt texts from every available split."""
    from datasets import load_dataset

    ds = load_dataset(dataset_name)
    texts = []
    for split in ds.values():
        col = next((c for c in TEXT_COLUMNS if c in split.column_names), None)
        if col is None:
            raise ValueError(f"{dataset_name}: no text column among {TEXT_COLUMNS}, "
                             f"found {split.column_names}")
        texts.extend(str(t).strip() for t in split[col])
    return texts


def _dedupe(texts):
    seen, out = set(), []
    for t in texts:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def check_disjoint(splits):
    """No prompt may appear twice anywhere (across splits or across harmful/harmless)."""
    texts = [r["text"] for s in splits.values() for k in KINDS for r in s[k]]
    ids = [r["id"] for s in splits.values() for k in KINDS for r in s[k]]
    assert len(texts) == len(set(texts)), "Duplicate prompt text across splits"
    assert len(ids) == len(set(ids)), "Duplicate prompt id across splits"


def build_splits(harmful, harmless, sizes, seed):
    """Shuffle each pool with `seed` and slice consecutive, non-overlapping chunks."""
    rng = random.Random(seed)
    pools = {}
    for kind, texts in zip(KINDS, (harmful, harmless)):
        pool = _dedupe(texts)
        needed = sum(sizes[s][kind] for s in sizes)
        if len(pool) < needed:
            raise ValueError(f"{kind}: need {needed} prompts, only {len(pool)} available")
        rng.shuffle(pool)
        pools[kind] = pool

    offsets = {k: 0 for k in KINDS}
    splits = {}
    for split_name, split_sizes in sizes.items():
        splits[split_name] = {}
        for kind in KINDS:
            n = split_sizes[kind]
            chunk = pools[kind][offsets[kind]: offsets[kind] + n]
            offsets[kind] += n
            splits[split_name][kind] = [{"id": make_id(kind, t), "text": t} for t in chunk]
    check_disjoint(splits)
    return splits


def save_splits(splits, path, meta=None):
    write_json(path, {"meta": meta or {}, "splits": splits})


def load_splits(path):
    return read_json(path)["splits"]


def load_split(paths, split, kind):
    """Convenience: records [{id, text}, ...] for one split and kind."""
    return load_splits(paths["data"] / "splits.json")[split][kind]
