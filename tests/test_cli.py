from pathlib import Path

from typer.testing import CliRunner

from judgebench.cli.main import app


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip()


def test_cli_report_markdown(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "report.md"
    result = runner.invoke(app, ["report", "examples/sample_results.csv", "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("# JudgeBench Report")


def test_cli_run_requires_api_key(tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "out.csv"
    result = runner.invoke(app, ["run", "examples/sample_dataset.csv", "--output", str(out)])
    assert result.exit_code == 1
    assert "Missing API key" in (result.stderr or result.stdout)

