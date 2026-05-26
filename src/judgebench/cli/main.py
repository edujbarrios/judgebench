from __future__ import annotations

# ruff: noqa: B008
from pathlib import Path
from typing import Literal

import pandas as pd
import typer
from dotenv import load_dotenv

from judgebench.benchmark.runner import run_dataset, run_pairwise_dataset
from judgebench.config.settings import JudgeBenchSettings
from judgebench.reports.html_report import write_html_report
from judgebench.reports.markdown_report import write_markdown_report

app = typer.Typer(
    name="judgebench",
    help="Local-first benchmarking for LLM-as-a-judge evaluation workflows.",
    no_args_is_help=True,
)


@app.callback()
def _env() -> None:
    load_dotenv(override=False)


@app.command()
def version() -> None:
    """Print package version."""
    from importlib.metadata import version as _version

    typer.echo(_version("judgebench"))


def _settings_with_overrides(
    *,
    api_base_url: str | None,
    api_key: str | None,
    model: str | None,
) -> JudgeBenchSettings:
    settings = JudgeBenchSettings()
    if api_base_url is not None:
        settings.api_base_url = api_base_url
    if api_key is not None:
        settings.api_key = api_key
    if model is not None:
        settings.model = model
    return settings


Strictness = Literal["conservative", "balanced", "aggressive"]
RunMode = Literal["standard", "pairwise"]
ReportFormat = Literal["markdown", "html"]


@app.command()
def run(
    dataset_csv: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path = typer.Option(..., "--output", "-o", dir_okay=False, help="Output CSV path."),
    mode: RunMode = typer.Option("standard", "--mode", help="Evaluation mode."),
    strictness: Strictness = typer.Option("balanced", "--strictness", help="Scoring strictness."),
    api_base_url: str | None = typer.Option(None, "--api-base-url"),
    api_key: str | None = typer.Option(None, "--api-key"),
    model: str | None = typer.Option(None, "--model"),
    temperature: float = typer.Option(0.2, "--temperature"),
    max_tokens: int = typer.Option(800, "--max-tokens"),
    system_prompt: str | None = typer.Option(None, "--system-prompt"),
) -> None:
    """Run evaluation on a CSV dataset and write a results CSV."""
    settings = _settings_with_overrides(api_base_url=api_base_url, api_key=api_key, model=model)
    if mode == "pairwise":
        run_pairwise_dataset(
            dataset_csv,
            output_csv=output,
            settings=settings,
            strictness=strictness,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        )
        typer.echo(f"Wrote pairwise results to {output}")
        raise typer.Exit(0)

    results = run_dataset(
        dataset_csv,
        output_csv=output,
        settings=settings,
        strictness=strictness,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    typer.echo(f"Wrote {len(results)} rows to {output}")


@app.command()
def pairwise(
    dataset_csv: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path = typer.Option(..., "--output", "-o", dir_okay=False, help="Output CSV path."),
    strictness: Strictness = typer.Option("balanced", "--strictness", help="Scoring strictness."),
    api_base_url: str | None = typer.Option(None, "--api-base-url"),
    api_key: str | None = typer.Option(None, "--api-key"),
    model: str | None = typer.Option(None, "--model"),
    temperature: float = typer.Option(0.2, "--temperature"),
    max_tokens: int = typer.Option(800, "--max-tokens"),
    system_prompt: str | None = typer.Option(None, "--system-prompt"),
) -> None:
    """Run pairwise comparison on a CSV dataset."""
    settings = _settings_with_overrides(api_base_url=api_base_url, api_key=api_key, model=model)
    results = run_pairwise_dataset(
        dataset_csv,
        output_csv=output,
        settings=settings,
        strictness=strictness,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    typer.echo(f"Wrote {len(results)} rows to {output}")


@app.command()
def report(
    results_csv: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    output: Path = typer.Option(..., "--output", "-o", dir_okay=False, help="Output report path."),
    format: ReportFormat = typer.Option("markdown", "--format", help="Report format."),
    title: str = typer.Option("JudgeBench Report", "--title"),
) -> None:
    """Generate a Markdown or HTML report from a results CSV."""
    df = pd.read_csv(results_csv)
    if format == "html":
        write_html_report(df, output, title=title)
        typer.echo(f"Wrote HTML report to {output}")
        return

    write_markdown_report(df, output, title=title)
    typer.echo(f"Wrote Markdown report to {output}")
