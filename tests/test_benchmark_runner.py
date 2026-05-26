from pathlib import Path

from judgebench.benchmark.runner import run_dataset


class _FakeClient:
    def __init__(self):
        self.calls = 0

    def chat_json(self, request, *, retries=3):  # noqa: ARG002
        self.calls += 1
        return {
            "semantic_similarity": 90,
            "correctness": 90,
            "completeness": 80,
            "hallucination_risk": 10,
            "clarity": 85,
            "reasoning": "ok",
            "final_score": 88,
        }


def test_run_dataset_writes_csv(tmp_path: Path) -> None:
    dataset = tmp_path / "data.csv"
    dataset.write_text(
        "id,reference,generated\n1,ref one,gen one\n2,ref two,gen two\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.csv"
    client = _FakeClient()
    results = run_dataset(dataset, output_csv=out, client=client)
    assert len(results) == 2
    assert client.calls == 2
    assert out.exists()

