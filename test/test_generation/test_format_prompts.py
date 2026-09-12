from gen.generate import format_prompts


def test_format_prompts() -> None:
    messages = [[
        {"role": "system", "content": "ignore this"},
        {"role": "user", "content": "hello"},
    ]]

    assert format_prompts(messages, tokenizer=None) == ["hello"]


if __name__ == "__main__":
    test_format_prompts()
    print("PASS")
