# judgebench

## What This Is

`judgebench` is a **local-first** benchmarking and evaluation framework that investigates whether **LLM-as-a-judge** methods can better approximate **semantic equivalence** and **human-aligned evaluation signals** for free-text generation than traditional deterministic lexical metrics.

“This project investigates whether LLM-as-a-judge methods can better approximate semantic and perceptual equivalence in generative AI outputs compared to traditional lexical metrics.”

It compares:
- **reference descriptions**
- **generated descriptions**

and produces normalized (0–100) scores for:
- semantic similarity
- factual alignment (correctness)
- completeness
- hallucination risk
- clarity
- reasoning (concise justification)
- final aggregated score

This project is intentionally research-oriented: it explores methodology and tooling, and avoids claiming perfect evaluation, objective truth, or universal metrics.

## Motivation

Traditional evaluation metrics (e.g. BLEU/ROUGE/exact match) often fail for modern generative systems because:
- multiple outputs can be semantically correct
- correct outputs can use different wording
- free-text outputs are non-deterministic
- lexical overlap is not semantic equivalence

Example:
- Reference: “Small opacity in the left lung base.”
- Generated: “Subtle opacity in the lower left lung.”

Lexical overlap can be low despite near-equivalent meaning. An LLM judge can provide a more human-like assessment by focusing on meaning and factual alignment.

## Architecture (High Level)

Pipeline:

`reference + generated` → prompt rendering → OpenAI-compatible LLM → structured JSON → validated parsing → normalization + aggregation → benchmark CSV → reports (Markdown/HTML)

Key modules:
- `src/judgebench/prompts/`: Jinja2 prompt templates + renderer
- `src/judgebench/judge/`: OpenAI-compatible client, JSON parsing, evaluators
- `src/judgebench/metrics/`: score normalization and aggregation
- `src/judgebench/datasets/`: CSV loading
- `src/judgebench/benchmark/`: dataset runners + summary statistics
- `src/judgebench/reports/`: CSV/Markdown/HTML report writers
- `src/judgebench/cli/`: Typer CLI

## Installation

```bash
git clone https://github.com/edujbarrios/judgebench.git
cd judgebench
python -m pip install -e ".[dev]"
```

## Configuration (Provider-Agnostic)

This project supports **any OpenAI-compatible API endpoint**. Configure via environment variables:

- `JUDGEBENCH_API_BASE_URL`
- `JUDGEBENCH_API_KEY`
- `JUDGEBENCH_MODEL`

Example `.env` (see `.env.example`):

```bash
JUDGEBENCH_API_BASE_URL=https://api.llm7.io/v1
JUDGEBENCH_API_KEY=replace-me
JUDGEBENCH_MODEL=replace-me
```

The CLI loads `.env` automatically if present.

## CLI Usage

Run evaluation:

```bash
judgebench run examples/sample_dataset.csv --output results.csv
```

Generate Markdown report:

```bash
judgebench report results.csv --output report.md
```

Generate HTML report:

```bash
judgebench report results.csv --output report.html --format html
```

Pairwise comparison:

```bash
judgebench pairwise examples/pairwise_dataset.csv --output pairwise_results.csv
```

Common options:
- `--api-base-url`, `--api-key`, `--model`
- `--temperature`, `--max-tokens`
- `--system-prompt`
- `--strictness conservative|balanced|aggressive`
- `--mode standard|pairwise` (for `judgebench run`)

## Python API

```python
from judgebench import evaluate_pair, run_dataset

result = evaluate_pair(
    reference="Small opacity in the left lung base.",
    generated="Subtle opacity in the lower left lung.",
)

results = run_dataset("examples/sample_dataset.csv")
```

## Dataset Formats

Standard dataset:

```csv
id,reference,generated
1,"Small opacity in the left lung base.","Subtle opacity in the lower left lung."
```

Pairwise dataset:

```csv
id,reference,generated_a,generated_b
1,"Small opacity in the left lung base.","A...","B..."
```

## Judge Methodology (LLM-as-a-Judge)

The judge is prompted to:
- compare semantic meaning (not lexical overlap)
- avoid rewarding verbosity or formatting similarity
- penalize hallucinations and unsupported claims
- return a single structured JSON object
- provide a short justification without chain-of-thought

The framework validates the JSON output and computes a consistent aggregated score from the dimension scores.

## Reports

Reports include:
- mean scores and simple distributions
- top/bottom examples with rationales
- HTML table view for quick inspection

See `results/baseline_report.md` for an example.

## Limitations

LLM judges are **probabilistic** and their outputs are not objective truth:
- results can vary across models and runs
- prompts can meaningfully change scoring behavior
- judges can exhibit bias and inconsistency
- hallucination detection is imperfect
- scores are approximate semantic estimates, not guaranteed correctness

## Roadmap

Planned extensions:
- embedding-based semantic similarity
- judge agreement + consistency experiments
- confidence calibration
- multimodal / image-aware evaluation (optional image path input)
- local LLM support
- judge ensembles
- experiment tracking and dashboards

## Contributing

Issues and PRs are welcome. Please:
- keep changes focused and well-tested
- avoid provider lock-in (OpenAI-compatible endpoints only)
- never commit real API keys

## Acknowledgements

For internal testing and development, this project used LLM7.io as an example OpenAI-compatible provider. All credit for that work goes to @chigwell. Any OpenAI-compatible API can be used by simply replacing the API endpoint, API key, and model name.

## License

MIT — see `LICENSE`.
