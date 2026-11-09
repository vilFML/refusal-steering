from src.generation import run_generation
from src.io_utils import read_jsonl

RECORDS = [{"id": f"id{i}", "text": "x" * (i + 1)} for i in range(7)]


def test_resume_skips_finished_and_keeps_order(tmp_path):
    out = tmp_path / "out.jsonl"
    calls = []

    def fake_fn(prompts):
        calls.append(len(prompts))
        return [{"output": p, "n_new_tokens": 1, "truncated": False} for p in prompts]

    # First run: stop after 2 batches by raising on the 3rd (simulates a Colab disconnect)
    state = {"n": 0}

    def flaky(prompts):
        state["n"] += 1
        if state["n"] == 3:
            raise RuntimeError("disconnect")
        return fake_fn(prompts)

    try:
        run_generation(RECORDS, out, flaky, batch_size=2)
    except RuntimeError:
        pass
    assert len(read_jsonl(out)) == 4  # two finished batches were saved

    calls.clear()
    rows = run_generation(RECORDS, out, fake_fn, batch_size=2)
    assert sum(calls) == 3                          # only the remaining prompts were generated
    assert [r["id"] for r in rows] == [r["id"] for r in RECORDS]  # original order preserved
    assert len(read_jsonl(out)) == 7                # no duplicates written
