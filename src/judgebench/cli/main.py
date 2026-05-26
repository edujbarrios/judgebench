import typer

app = typer.Typer(
    name="judgebench",
    help="Local-first benchmarking for LLM-as-a-judge evaluation workflows.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print package version."""
    from importlib.metadata import version as _version

    typer.echo(_version("judgebench"))

