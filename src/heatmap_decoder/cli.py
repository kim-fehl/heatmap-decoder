"""Typer adapter: configuration and filesystem handling stay outside the core."""

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from .extract import UnresolvedLabelsError
from .io import LABEL_OUTPUT_NAMES, extract_file, extract_labels_file
from .models import CellFlag, ExtractionConfig, GridConfig, LabelExtractionResult
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
    labels_review: Annotated[
        Path | None,
        typer.Option(
            "--labels-review",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Complete UTF-8 review TSV: axis, index, text.",
        ),
    ] = None,
) -> None:
    """Write matrix.csv, quality.csv, provenance.json and diagnostic.png."""
    try:
        settings = ExtractionConfig.model_validate_json(config.read_text(encoding="utf-8"))
        outputs = [
            out / name
            for name in ("matrix.csv", "quality.csv", "provenance.json", "diagnostic.png")
        ]
        if settings.labels is not None and (
            settings.labels.rows is not None or settings.labels.columns is not None
        ):
            outputs += [out / name for name in LABEL_OUTPUT_NAMES if name != "provenance.json"]
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
        result = extract_file(
            figure,
            settings,
            out,
            mask=pixel_mask,
            base_directory=config.parent,
            config_path=config,
            overwrite=overwrite,
            labels_review=labels_review,
        )
    except UnresolvedLabelsError as error:
        typer.echo(
            f"Wrote label review artifacts to {out}. {error}. No new matrix/quality outputs.",
            err=True,
        )
        raise typer.Exit(code=4) from None
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from None
    total = result.flags.size
    flagged = int(np.count_nonzero(result.flags))
    typer.echo(f"Wrote {settings.rows} × {settings.columns} matrix and QC outputs to {out}")
    typer.echo(f"Flagged cells: {flagged}/{total} ({flagged / total:.1%})")
    if result.labels is not None:
        label_warnings(result.labels)
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


@app.command("labels")
def labels_command(
    figure: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="JSON grid/label configuration; colourbar optional.",
        ),
    ],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("review"),
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Replace existing review files.")
    ] = False,
) -> None:
    """Write label TSVs, a diagnostic overlay and provenance; exit 4 for unresolved slots."""
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
        model = (
            ExtractionConfig
            if isinstance(payload, dict) and "colourbar_bbox" in payload
            else GridConfig
        )
        settings = model.model_validate(payload)
        result = extract_labels_file(
            figure,
            settings,
            out,
            base_directory=config.parent,
            config_path=config,
            overwrite=overwrite,
        )
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(f"Wrote label review artifacts to {out}")
    typer.echo(f"Unresolved requested slots: {result.unresolved_count}")
    label_warnings(result)
    if result.unresolved_count:
        raise typer.Exit(code=4)


def label_warnings(result: LabelExtractionResult) -> None:
    unmatched = sum(len(axis.unmatched_candidate_indices) for axis in result.axes)
    if unmatched:
        typer.echo(
            f"Warning: {unmatched} unmatched OCR candidates; inspect provenance and overlay.",
            err=True,
        )
    if any(slot.duplicate for axis in result.axes for slot in axis.slots):
        typer.echo(
            "Warning: duplicate label text is preserved; "
            "some CSV readers rename duplicate headers.",
            err=True,
        )
    if result.ocr_runs:
        typer.echo("Verify all OCR text against the figure, including accepted labels.", err=True)


if __name__ == "__main__":
    app()
