from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from judgebench.benchmark.statistics import describe_distribution, summarize_scores


@dataclass(frozen=True)
class MarkdownReport:
    markdown: str


def generate_markdown_report(df: pd.DataFrame, *, title: str = "JudgeBench Report") -> MarkdownReport:
    summary = summarize_scores(df)
    distro = describe_distribution(df, "final_score")

    best = df.sort_values("final_score", ascending=False).head(5)
    worst = df.sort_values("final_score", ascending=True).head(5)

    md = []
    md.append(f"# {title}")
    md.append("")
    md.append("## Score Summary")
    md.append("")
    md.append(
        f"- Mean final score: **{summary.mean_final_score:.2f}**\n"
        f"- Mean semantic similarity: **{summary.mean_semantic_similarity:.2f}**\n"
        f"- Mean correctness: **{summary.mean_correctness:.2f}**\n"
        f"- Mean completeness: **{summary.mean_completeness:.2f}**\n"
        f"- Mean hallucination risk (higher=worse): **{summary.mean_hallucination_risk:.2f}**\n"
        f"- Mean clarity: **{summary.mean_clarity:.2f}**"
    )
    md.append("")
    md.append("## Final Score Distribution")
    md.append("")
    md.append(
        f"- min={distro['min']:.1f}, p25={distro['p25']:.1f}, median={distro['median']:.1f}, "
        f"p75={distro['p75']:.1f}, max={distro['max']:.1f}, mean={distro['mean']:.1f}"
    )
    md.append("")
    md.append("## Top Examples")
    md.append("")
    md.append(best[["id", "final_score", "reasoning"]].to_markdown(index=False))
    md.append("")
    md.append("## Lowest Examples")
    md.append("")
    md.append(worst[["id", "final_score", "reasoning"]].to_markdown(index=False))
    md.append("")
    return MarkdownReport(markdown="\n".join(md))


def write_markdown_report(df: pd.DataFrame, path: str | Path, *, title: str = "JudgeBench Report") -> None:
    report = generate_markdown_report(df, title=title)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.markdown, encoding="utf-8")

