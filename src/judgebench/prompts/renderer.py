from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Literal

from jinja2 import Environment, StrictUndefined

Strictness = Literal["conservative", "balanced", "aggressive"]


@dataclass(frozen=True)
class RenderedPrompt:
    system: str
    user: str


def _load_template(name: str) -> str:
    return (resources.files("judgebench.prompts.templates") / name).read_text(encoding="utf-8")


def render_standard_prompt(
    reference: str,
    generated: str,
    *,
    strictness: Strictness = "balanced",
    system_prompt: str | None = None,
) -> RenderedPrompt:
    env = Environment(undefined=StrictUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(_load_template("semantic_eval.j2"))
    user = template.render(reference=reference, generated=generated, strictness=strictness)
    system = system_prompt or _load_template("judge_description.j2").strip()
    return RenderedPrompt(system=system, user=user)


def render_pairwise_prompt(
    reference: str,
    generated_a: str,
    generated_b: str,
    *,
    strictness: Strictness = "balanced",
    system_prompt: str | None = None,
) -> RenderedPrompt:
    env = Environment(undefined=StrictUndefined, autoescape=False, trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(_load_template("pairwise_comparison.j2"))
    user = template.render(
        reference=reference,
        generated_a=generated_a,
        generated_b=generated_b,
        strictness=strictness,
    )
    system = system_prompt or _load_template("judge_description.j2").strip()
    return RenderedPrompt(system=system, user=user)

