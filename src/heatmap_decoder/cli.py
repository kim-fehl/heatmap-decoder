"""Typer adapter: configuration and filesystem handling stay outside the core."""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from .io import extract_file
from .models import CellFlag, ExtractionConfig
from .version import __version__

app = typer.Typer(
    no_args_is_help=True, help="Approximate matrix reconstruction from a raster heatmap."
)


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[
        bool, typer.Option("--version", help="Show the software version and exit.", is_eager=True)
    ] = False,
) -> None:
    if version:
        typer.echo(f"heatmap-decoder {__version__}")
        raise typer.Exit()


@app.command("extract")
def extract_command(
    figure: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON extraction configuration.",
        ),
    ],
    out: Annotated[Path, typer.Option("--out", file_okay=False, help="Output directory.")] = Path(
        "results"
    ),
    mask: Annotated[
        Path | None,
        typer.Option(
            "--mask",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Optional boolean H × W .npy mask; True means missing.",
        ),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace existing output files.")
    ] = False,
    fail_on_qc: Annotated[
        bool,
        typer.Option("--fail-on-qc", help="Write outputs, then exit 3 if any cells are flagged."),
    ] = False,
) -> None:
    """Write matrix.csv, quality.csv, provenance.json and diagnostic.png."""
    try:
        settings = ExtractionConfig.model_validate_json(config.read_text(encoding="utf-8"))
        outputs = [
            out / name
            for name in ("matrix.csv", "quality.csv", "provenance.json", "diagnostic.png")
        ]
        inputs = [figure, config] + ([mask] if mask is not None else [])
        if any(output.resolve() == source.resolve() for output in outputs for source in inputs):
            raise ValueError("output paths would overwrite an input figure, configuration or mask")
        existing = [output.name for output in outputs if output.exists()]
        if existing and not overwrite:
            raise ValueError(f"outputs already exist ({', '.join(existing)}); use --overwrite")
        pixel_mask = np.load(mask, allow_pickle=False) if mask is not None else None
        if pixel_mask is not None and not isinstance(pixel_mask, np.ndarray):
            pixel_mask.close()
            raise ValueError("mask must be a boolean .npy array, not an archive")
        result = extract_file(figure, settings, out, mask=pixel_mask)
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from None
    total = result.flags.size
    flagged = int(np.count_nonzero(result.flags))
    typer.echo(f"Wrote {settings.rows} × {settings.columns} matrix and QC outputs to {out}")
    typer.echo(f"Flagged cells: {flagged}/{total} ({flagged / total:.1%})")
    if flagged:
        counts = ", ".join(
            f"{flag.name}={np.count_nonzero(result.flags & flag)}" for flag in CellFlag
        )
        typer.echo(
            f"Warning: {counts}. Missing/ambiguous values are NaN; poor matches retain estimates. "
            "Inspect quality.csv and diagnostic.png.",
            err=True,
        )
        if fail_on_qc:
            raise typer.Exit(code=3)


if __name__ == "__main__":
    app()
