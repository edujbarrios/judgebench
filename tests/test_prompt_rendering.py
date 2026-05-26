from judgebench.prompts.renderer import render_pairwise_prompt, render_standard_prompt


def test_render_standard_prompt_includes_inputs() -> None:
    prompt = render_standard_prompt("ref text", "gen text", strictness="balanced")
    assert "ref text" in prompt.user
    assert "gen text" in prompt.user
    assert "STRICTNESS" in prompt.user
    assert prompt.system


def test_render_pairwise_prompt_includes_inputs() -> None:
    prompt = render_pairwise_prompt("ref", "a", "b", strictness="conservative")
    assert "GENERATED_A" in prompt.user
    assert "GENERATED_B" in prompt.user
    assert "conservative" in prompt.user

