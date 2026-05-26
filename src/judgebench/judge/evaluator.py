from __future__ import annotations

from typing import Any

from judgebench.config.settings import JudgeBenchSettings
from judgebench.judge.client import ChatRequest, OpenAICompatibleClient
from judgebench.metrics.aggregation import aggregate_final_score
from judgebench.metrics.normalization import clamp_score
from judgebench.prompts.renderer import Strictness, render_pairwise_prompt, render_standard_prompt
from judgebench.schemas.evaluation import (
    EvaluationResult,
    PairwiseJudgeOutput,
    PairwiseResult,
    StandardJudgeOutput,
)


def _client_from_settings(settings: JudgeBenchSettings) -> OpenAICompatibleClient:
    if not settings.api_key:
        raise ValueError(
            "Missing API key. Set JUDGEBENCH_API_KEY or pass --api-key in the CLI."
        )
    return OpenAICompatibleClient(
        api_key=settings.api_key,
        base_url=settings.api_base_url,
        model=settings.model,
    )


def evaluate_pair(
    *,
    id: str = "0",
    reference: str,
    generated: str,
    settings: JudgeBenchSettings | None = None,
    strictness: Strictness = "balanced",
    temperature: float = 0.2,
    max_tokens: int = 800,
    system_prompt: str | None = None,
    client: OpenAICompatibleClient | None = None,
) -> EvaluationResult:
    settings = settings or JudgeBenchSettings()
    client = client or _client_from_settings(settings)

    prompt = render_standard_prompt(
        reference,
        generated,
        strictness=strictness,
        system_prompt=system_prompt,
    )
    raw: dict[str, Any] = client.chat_json(
        ChatRequest(system=prompt.system, user=prompt.user, temperature=temperature, max_tokens=max_tokens)
    )
    parsed = StandardJudgeOutput.model_validate(raw)

    semantic_similarity = clamp_score(parsed.semantic_similarity)
    correctness = clamp_score(parsed.correctness)
    completeness = clamp_score(parsed.completeness)
    hallucination_risk = clamp_score(parsed.hallucination_risk)
    clarity = clamp_score(parsed.clarity)
    final_score = aggregate_final_score(
        semantic_similarity=semantic_similarity,
        correctness=correctness,
        completeness=completeness,
        hallucination_risk=hallucination_risk,
        clarity=clarity,
    )
    return EvaluationResult(
        id=id,
        reference=reference,
        generated=generated,
        semantic_similarity=semantic_similarity,
        correctness=correctness,
        completeness=completeness,
        hallucination_risk=hallucination_risk,
        clarity=clarity,
        final_score=final_score,
        reasoning=parsed.reasoning.strip(),
        raw_judge_output=raw,
    )


def evaluate_pairwise(
    *,
    id: str = "0",
    reference: str,
    generated_a: str,
    generated_b: str,
    settings: JudgeBenchSettings | None = None,
    strictness: Strictness = "balanced",
    temperature: float = 0.2,
    max_tokens: int = 800,
    system_prompt: str | None = None,
    client: OpenAICompatibleClient | None = None,
) -> PairwiseResult:
    settings = settings or JudgeBenchSettings()
    client = client or _client_from_settings(settings)

    prompt = render_pairwise_prompt(
        reference,
        generated_a,
        generated_b,
        strictness=strictness,
        system_prompt=system_prompt,
    )
    raw: dict[str, Any] = client.chat_json(
        ChatRequest(system=prompt.system, user=prompt.user, temperature=temperature, max_tokens=max_tokens)
    )
    parsed = PairwiseJudgeOutput.model_validate(raw)
    return PairwiseResult(
        id=id,
        reference=reference,
        generated_a=generated_a,
        generated_b=generated_b,
        winner=parsed.winner,
        confidence=clamp_score(parsed.confidence),
        reasoning=parsed.reasoning.strip(),
        raw_judge_output=raw,
    )

