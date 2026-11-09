import pytest

from src.data import build_splits, check_disjoint

SIZES = {
    "train": {"harmful": 5, "harmless": 5},
    "val": {"harmful": 2, "harmless": 2},
    "test": {"harmful": 3, "harmless": 1},
}


def pools(n=20):
    return [f"harmful prompt {i}" for i in range(n)], [f"harmless prompt {i}" for i in range(n)]


def test_sizes_and_disjointness():
    h, b = pools()
    splits = build_splits(h, b, SIZES, seed=0)
    for s, sz in SIZES.items():
        assert len(splits[s]["harmful"]) == sz["harmful"]
        assert len(splits[s]["harmless"]) == sz["harmless"]
    check_disjoint(splits)


def test_deterministic_with_seed_and_different_across_seeds():
    h, b = pools()
    assert build_splits(h, b, SIZES, 1) == build_splits(h, b, SIZES, 1)
    assert build_splits(h, b, SIZES, 1) != build_splits(h, b, SIZES, 2)


def test_duplicates_are_removed():
    h, b = pools()
    splits = build_splits(h + h, b, SIZES, 0)
    check_disjoint(splits)


def test_not_enough_prompts_raises():
    h, b = pools(n=5)
    with pytest.raises(ValueError):
        build_splits(h, b, SIZES, 0)
