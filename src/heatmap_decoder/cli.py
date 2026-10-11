"""Typer adapter: configuration and filesystem handling stay outside the core."""

import hashlib
import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from .extract import UnresolvedLabelsError
from .io import (
    LABEL_OUTPUT_NAMES,
    detect_file,
    extract_file,
    extract_labels_file,
    load_raster,
    review_detection,
)
from .models import (
    CellFlag,
    DetectionHints,
    ExtractionConfig,
    GridConfig,
    LabelExtractionResult,
    ROIReview,
)
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
    roi_review: Annotated[
        Path | None,
        typer.Option(
            "--roi-review",
            exists=True,
            dir_okay=False,
            help="Saved roi_candidates.json to validate and preserve.",
        ),
    ] = None,
    roi_selections: Annotated[
        Path | None,
        typer.Option(
            "--roi-selections",
            exists=True,
            dir_okay=False,
            help="Reviewed JSON region/count candidate ID selections.",
        ),
    ] = None,
    heatmap_candidate: Annotated[
        str | None, typer.Option("--heatmap-candidate", help="Explicit reviewed candidate ID.")
    ] = None,
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
        payload = json.loads(config.read_text(encoding="utf-8"))
        if (heatmap_candidate is not None or roi_selections is not None) and roi_review is None:
            raise ValueError("--heatmap-candidate requires --roi-review")
        if roi_review is not None:
            settings, _ = review_detection(
                roi_review.read_bytes(),
                load_raster(figure),
                hashlib.sha256(figure.read_bytes()).hexdigest(),
                payload,
                heatmap_candidate,
                ROIReview.model_validate_json(roi_selections.read_bytes())
                if roi_selections is not None
                else None,
            )
        else:
            settings = ExtractionConfig.model_validate(payload)
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
            roi_review=roi_review,
            heatmap_candidate=heatmap_candidate,
            roi_selections=roi_selections,
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


@app.command("detect")
def detect_command(
    figure: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            help="Detection hints JSON; supply counts or explicitly request inference.",
        ),
    ],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("roi_review"),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Write review-only ROI/count proposals; exit 5 until explicitly reviewed.

    v0.3 automatic ROI/count acceptance is disabled: M9 gates are not established.
    """
    try:
        hints = DetectionHints.model_validate_json(config.read_text(encoding="utf-8"))
        saved = detect_file(figure, hints, out, config_path=config, overwrite=overwrite)
    except (ValueError, OSError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(f"Wrote ROI review artifacts to {out}; {saved.result.status}")
    typer.echo(f"ROI candidates: {len(saved.result.candidates)}; scores are not probabilities.")
    typer.echo(f"Count proposals: {len(saved.result.count_proposals)}; review required.")
    for warning in saved.result.warnings:
        typer.echo(f"Warning: {warning}", err=True)
    if not saved.proposed_config_complete:
        typer.echo(
            "Proposed extraction config is incomplete: "
            + ", ".join(saved.missing_extraction_fields),
            err=True,
        )
    if saved.result.unresolved_requirements:
        for requirement in saved.result.unresolved_requirements:
            typer.echo(f"Unresolved: {requirement}", err=True)
        typer.echo(
            "Review the overlay and select a candidate or edit the proposed config.", err=True
        )
        raise typer.Exit(code=5)


if __name__ == "__main__":
    app()
