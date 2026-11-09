"""Batched greedy generation with per-batch checkpointing (Section 1.1)."""
from pathlib import Path

from src.io_utils import append_jsonl, read_jsonl
from src.model import build_prompt


def make_generate_fn(model, tok, cfg):
    """Return generate_fn(list[str]) -> list[{output, n_new_tokens, truncated}].
    Splits a batch in half automatically if the T4 runs out of memory."""
    import torch

    gen = cfg["generation"]
    thinking = cfg["model"]["enable_thinking"]
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])

    def _generate(prompts):
        texts = [build_prompt(tok, p, thinking) for p in prompts]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                do_sample=gen["do_sample"],
                max_new_tokens=gen["max_new_tokens"],
                temperature=None, top_p=None, top_k=None,  # silence sampling defaults under greedy
                pad_token_id=tok.pad_token_id,
            )
        new_tokens = out[:, enc["input_ids"].shape[1]:].tolist()
        results = []
        for ids in new_tokens:
            stop = next((i for i, t in enumerate(ids) if t in eos_ids), None)
            n_new = len(ids) if stop is None else stop + 1
            results.append({
                "output": tok.decode(ids[:n_new], skip_special_tokens=True).strip(),
                "n_new_tokens": n_new,
                "truncated": stop is None,  # hit max_new_tokens without finishing
            })
        return results

    def generate_fn(prompts):
        try:
            return _generate(prompts)
        except torch.cuda.OutOfMemoryError:
            if len(prompts) == 1:
                raise
            torch.cuda.empty_cache()
            mid = len(prompts) // 2
            print(f"OOM with batch of {len(prompts)}; retrying as {mid} + {len(prompts) - mid}")
            return generate_fn(prompts[:mid]) + generate_fn(prompts[mid:])

    return generate_fn


def run_generation(records, out_path, generate_fn, batch_size, meta=None):
    """Generate for `records` ([{id, text}]) and append each batch to `out_path` (JSONL).
    Already-finished ids are skipped, so a Colab disconnect only loses the current batch.
    Returns rows in the same order as `records`."""
    out_path = Path(out_path)
    done = {r["id"] for r in read_jsonl(out_path)}
    todo = [r for r in records if r["id"] not in done]
    todo.sort(key=lambda r: len(r["text"]), reverse=True)  # similar lengths per batch, less padding
    print(f"{len(records)} prompts: {len(done & {r['id'] for r in records})} already done, {len(todo)} to run")

    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        results = generate_fn([r["text"] for r in batch])
        rows = [{"id": r["id"], "prompt": r["text"], **res, **(meta or {})}
                for r, res in zip(batch, results)]
        append_jsonl(out_path, rows)
        print(f"  {min(i + batch_size, len(todo))}/{len(todo)}")

    by_id = {r["id"]: r for r in read_jsonl(out_path)}
    return [by_id[r["id"]] for r in records]
