"""Model + tokenizer loading and chat-template prompt building (Section 1.1)."""


def load_model_and_tokenizer(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    m = cfg["model"]
    tok = AutoTokenizer.from_pretrained(m["name"], padding_side="left")  # left pad for batched generate
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        m["name"],
        torch_dtype=getattr(torch, m["dtype"]),
        attn_implementation=m["attn_implementation"],
        device_map={"": 0},
    )
    model.eval()
    return model, tok


def build_prompt(tok, user_text, enable_thinking=False):
    """Apply the Qwen3 chat template. With enable_thinking=False the template inserts an
    empty <think></think> block, so the model answers directly."""
    messages = [{"role": "user", "content": user_text}]
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
