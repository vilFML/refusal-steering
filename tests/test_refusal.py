from src.refusal import detect_refusal, is_refusal, summarize


def test_typical_refusals():
    assert is_refusal("I'm sorry, but I can't assist with that request.")
    assert is_refusal("I\u2019m sorry, but I can\u2019t help with that.")  # curly apostrophes
    assert is_refusal("I cannot provide instructions for that.")
    assert is_refusal("  I CAN'T do that.")


def test_compliance_not_flagged():
    assert not is_refusal("Sure! Here is a step-by-step guide to baking bread.")
    assert not is_refusal("")
    assert not is_refusal(None)


def test_phrase_outside_window_is_ignored():
    text = "Here is a long and detailed answer. " * 5 + "I'm sorry, this ends here."
    assert not is_refusal(text, window_chars=100)
    assert is_refusal(text, window_chars=1000)


def test_returns_matched_phrase():
    assert detect_refusal("I won't do that.") == (True, "i won't")


def test_summarize():
    rows = [
        {"output": "I'm sorry, no.", "n_new_tokens": 5, "truncated": False},
        {"output": "Sure, here you go.", "n_new_tokens": 100, "truncated": True},
        {"output": "", "n_new_tokens": 1, "truncated": False},
    ]
    s = summarize(rows)
    assert s["n"] == 3 and s["n_refusals"] == 1 and s["n_empty"] == 1 and s["n_truncated"] == 1
    assert abs(s["refusal_rate"] - 1 / 3) < 1e-9
