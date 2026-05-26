from judgebench.judge.evaluator import evaluate_pair, evaluate_pairwise


class _FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)

    def chat_json(self, request, *, retries=3):  # noqa: ARG002
        return self._responses.pop(0)


def test_evaluate_pair_computes_final_score() -> None:
    client = _FakeClient(
        [
            {
                "semantic_similarity": 90,
                "correctness": 90,
                "completeness": 80,
                "hallucination_risk": 10,
                "clarity": 85,
                "reasoning": "ok",
                "final_score": 1,
            }
        ]
    )
    result = evaluate_pair(id="1", reference="r", generated="g", client=client)
    assert result.final_score != 1
    assert result.id == "1"
    assert result.raw_judge_output


def test_evaluate_pairwise_parses_winner() -> None:
    client = _FakeClient([{"winner": "A", "confidence": 77, "reasoning": "A is better"}])
    result = evaluate_pairwise(
        id="1",
        reference="r",
        generated_a="a",
        generated_b="b",
        client=client,
    )
    assert result.winner == "A"
    assert result.confidence == 77

