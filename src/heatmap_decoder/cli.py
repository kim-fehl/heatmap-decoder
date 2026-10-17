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
    LEGEND_OUTPUT_NAMES,
    UnresolvedLegendError,
    detect_file,
    extract_file,
    extract_labels_file,
    inspect_legend_file,
    load_raster,
    review_detection,
)
from .models import (
    CellFlag,
    DetectionHints,
    ExtractionConfig,
    GridConfig,
    LabelExtractionResult,
    LegendConfig,
    LegendOptions,
    ROIReview,
)
from .version import __version__

app = typer.Typer(
    no_args_is_help=True, help="Approximate matrix reconstruction from a raster heatmap."
)


@app.command("extract-ranges")
def extract_ranges_command(
    figure: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    request: Annotated[Path, typer.Option("--request", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("results/ranges"),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Recover reviewed expression ranges, including open-ended ranges."""
    import csv

    from .io import check_output_paths, load_file_labels
    from .models import FileLabelSource
    from .range_scale import RangeExtractionRequest, extract_ranges

    try:
        settings = RangeExtractionRequest.model_validate_json(request.read_bytes())
        protected = [figure, request]
        config = settings.position_config
        labels, hashes = load_file_labels(config, request.parent)
        if config.labels is not None:
            for axis in ("rows", "columns"):
                source = getattr(config.labels, axis)
                if isinstance(source, FileLabelSource):
                    protected.append(request.parent / source.path)
        check_output_paths([out / "ranges.json", out / "ranges.tsv"], tuple(protected), overwrite)
        result = extract_ranges(load_raster(figure), settings, file_labels=labels)
        result.update(
            input_sha256=hashlib.sha256(figure.read_bytes()).hexdigest(),
            file_label_sha256=hashes,
            request=settings.model_dump(mode="json"),
        )
        out.mkdir(parents=True, exist_ok=True)
        (out / "ranges.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        with (out / "ranges.tsv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
                    "row",
                    "column",
                    "status",
                    "range",
                    "lower",
                    "upper",
                    "lower_inclusive",
                    "upper_inclusive",
                    "colour_distance",
                ]
            )
            for record in result["records"]:
                band = record["range"] or {}
                writer.writerow(
                    [
                        record["row"],
                        record["column"],
                        record["status"],
                        band.get("label"),
                        band.get("lower"),
                        band.get("upper"),
                        band.get("lower_inclusive"),
                        band.get("upper_inclusive"),
                        record["colour_distance"],
                    ]
                )
    except (ValueError, OSError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(2) from error
    typer.echo(f"Wrote expression ranges to {out}")
    if not result["complete"]:
        raise typer.Exit(4)


@app.command("group-labels")
def group_labels_command(
    figure: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    request: Annotated[Path, typer.Option("--request", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("results/groups"),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Read group tracks independently of optional individual sample names."""
    import csv

    from .group_labels import GroupLabelsRequest, infer_group_labels
    from .io import check_output_paths, load_file_labels
    from .models import FileLabelSource

    try:
        settings = GroupLabelsRequest.model_validate_json(request.read_bytes())
        names = ("groups.json", "groups.tsv", f"{settings.axis}.txt")
        protected = [figure, request]
        file_labels, hashes = (
            ({}, {}) if settings.grid is None else load_file_labels(settings.grid, request.parent)
        )
        if settings.grid is not None and settings.grid.labels is not None:
            for axis in ("rows", "columns"):
                source = getattr(settings.grid.labels, axis)
                if isinstance(source, FileLabelSource):
                    protected.append(request.parent / source.path)
        check_output_paths([out / name for name in names], tuple(protected), overwrite)
        result = infer_group_labels(load_raster(figure), settings, file_labels=file_labels)
        result["file_label_sha256"] = hashes
        result["input_sha256"] = hashlib.sha256(figure.read_bytes()).hexdigest()
        result["request"] = settings.model_dump(mode="json")
        out.mkdir(parents=True, exist_ok=True)
        (out / "groups.json").write_text(json.dumps(result, indent=2) + "\n")
        with (out / "groups.tsv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(
                [
                    "index",
                    "display_name",
                    "generated_name",
                    "sample_name",
                    "sample_status",
                    "entity_kind",
                    "entity_name",
                    "entity_status",
                    *result["track_names"],
                ]
            )
            for record in result["records"]:
                writer.writerow(
                    [
                        record["index"],
                        record["display_name"],
                        record["generated_name"],
                        record["sample_name"],
                        record["sample_status"],
                        result["entity_kind"],
                        record["entity_name"],
                        record["entity_status"],
                        *(t["category"] for t in record["tracks"]),
                    ]
                )
        label_path = out / f"{settings.axis}.txt"
        if result["complete"]:
            label_path.write_text(
                "\n".join(r["display_name"] for r in result["records"]) + "\n",
                encoding="utf-8",
            )
        elif label_path.exists():
            label_path.unlink()  # Remove a stale complete file on an explicit overwrite.
    except (ValueError, OSError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(2) from error
    if not result["complete"]:
        typer.echo(f"Unresolved categorical assignments; review {out / 'groups.json'}")
        raise typer.Exit(4)
    typer.echo(f"Wrote {settings.count} group/position names to {label_path}")


@app.command("automate")
def automate_command(
    figure: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    request: Annotated[Path, typer.Option("--request", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("results/automation"),
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    experimental_export: Annotated[
        bool,
        typer.Option(
            "--experimental-export",
            help="Export qualified experimental numeric results; identifiers remain separate.",
        ),
    ] = False,
) -> None:
    """Compose ROI/count/scale decisions; emit diagnostics when release gates fail."""
    from PIL import Image, ImageDraw

    from .automation import AutomationRequest, automate
    from .extract import extract
    from .io import (
        NUMERIC_OUTPUT_NAMES,
        _decode_raster,
        check_output_paths,
        write_legend_outputs,
        write_outputs,
        write_positional_outputs,
    )

    names = (
        "automation.json",
        "roi_diagnostic.png",
        "label_evidence.json",
        *LEGEND_OUTPUT_NAMES,
        *(f"legend_candidates/{i}/{name}" for i in range(1, 4) for name in LEGEND_OUTPUT_NAMES),
    )
    try:
        settings = AutomationRequest.model_validate_json(request.read_bytes())
        protected = {figure.resolve(), request.resolve()}
        for name in names:
            target = out / name
            if target.resolve() in protected:
                raise ValueError("automation outputs cannot overwrite an input")
            if target.exists() and not overwrite:
                raise ValueError(f"output exists: {target}; use --overwrite")
        data = figure.read_bytes()
        image = _decode_raster(data)
        result = automate(image, settings, experimental_export=experimental_export)
        positional = not settings.numeric_only or result.numeric_state != "complete"
        numeric_out = out / "numeric_positional" if positional else out
        if result.numeric_retention_allowed:
            numeric_names = (
                tuple(n for n in NUMERIC_OUTPUT_NAMES if n != "matrix.csv")
                + ("numeric_positional.csv", "numeric_partial.csv")
                if positional
                else NUMERIC_OUTPUT_NAMES
            )
            check_output_paths(
                [numeric_out / name for name in numeric_names], (figure, request), overwrite
            )
        payload = result.model_dump(mode="json")
        payload["input"] = {
            "filename": figure.name,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        payload["request"] = settings.model_dump(mode="json")
        out.mkdir(parents=True, exist_ok=True)
        (out / "automation.json").write_text(json.dumps(payload, indent=2) + "\n")
        (out / "label_evidence.json").write_text(
            json.dumps(
                {
                    "identifiers": result.identifier_state,
                    "labelled_request_fulfilled": False if not settings.numeric_only else None,
                    "positional_ids_are_names": False,
                    "evidence": list(result.label_evidence),
                    "geometry_decision": result.label_geometry_complete.model_dump(mode="json"),
                },
                indent=2,
            )
            + "\n"
        )
        overlay = Image.fromarray(image[:, :, :3]).convert("RGB")
        painter = ImageDraw.Draw(overlay)
        for candidate in result.detection.candidates:
            x, y, r, s = candidate.bbox
            painter.rectangle((x, y, r - 1, s - 1), outline="orange", width=2)
            painter.text((x, max(0, y - 14)), f"{candidate.id} {candidate.kind}", fill="black")
        overlay.save(out / "roi_diagnostic.png")
        if result.legend is not None:
            write_legend_outputs(image, result.legend, out, payload["input"]["sha256"], figure.name)
        elif result.legend_diagnostics:
            for index, diagnostic in enumerate(result.legend_diagnostics, 1):
                write_legend_outputs(
                    image,
                    diagnostic,
                    out / "legend_candidates" / str(index),
                    payload["input"]["sha256"],
                    figure.name,
                )
        if result.numeric_retention_allowed:
            assert result.diagnostic_config is not None
            numeric = extract(
                image, result.diagnostic_config, ocr_anchors=result.ocr_anchors or None
            )
            writer = write_positional_outputs if positional else write_outputs
            writer(
                numeric,
                numeric_out,
                source_image=image,
                input_filename=figure.name,
                input_sha256=payload["input"]["sha256"],
                automation_metadata={
                    "maturity": "experimental",
                    "numeric_state": result.numeric_state,
                    "identifiers": "positional",
                    "ocr_anchors": [a.model_dump(mode="json") for a in result.ocr_anchors],
                    "calibration": result.numeric_complete.evidence,
                    "stable_release": False,
                    "policy_version": result.policy_version,
                    "runtime_control": "--experimental-export",
                    "request": settings.model_dump(mode="json"),
                    "decisions": [d.model_dump(mode="json") for d in result.decisions],
                    "diagnostic_report_sha256": hashlib.sha256(
                        (out / "automation.json").read_bytes()
                    ).hexdigest(),
                    "resolved_config": result.diagnostic_config.model_dump(mode="json"),
                },
            )
    except (ValueError, OSError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(2) from error
    typer.echo(f"experimental complete decision: {result.complete.status}; diagnostics: {out}")
    if result.automatic_export_allowed:
        typer.echo(
            f"Wrote experimental numeric matrix and QC to {out}; stable release remains off."
        )
        return
    if result.numeric_retention_allowed:
        remaining = (
            "Identifier review remains unfinished."
            if not settings.numeric_only
            else "Numeric output remains partial; inspect quality.csv."
        )
        typer.echo(
            f"Retained {result.numeric_state} positional results in {numeric_out}. {remaining}"
        )
    if not experimental_export and result.complete.status == "accepted":
        typer.echo("Diagnostic mode; use --experimental-export for qualified numeric output.")
        raise typer.Exit(5)
    typer.echo("Review required; see automation.json for each capability and export scope.")
    raise typer.Exit(5)


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
    legend_review: Annotated[
        Path | None,
        typer.Option(
            "--legend-review",
            exists=True,
            dir_okay=False,
            help="Complete image-bound legend review TSV.",
        ),
    ] = None,
    auto_roi: Annotated[
        bool,
        typer.Option("--auto-roi", help="Evaluate M10 ROI decisions; M12 export gate is closed."),
    ] = False,
    numeric_only: Annotated[
        bool, typer.Option("--numeric-only", help="Explicitly request generated positional IDs.")
    ] = False,
) -> None:
    """Write matrix.csv, quality.csv, provenance.json and diagnostic.png."""
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
        if numeric_only and not auto_roi:
            raise ValueError(
                "--numeric-only requires --auto-roi; manual configs already set labels"
            )
        if auto_roi:
            if any(
                value is not None
                for value in (
                    roi_review,
                    roi_selections,
                    heatmap_candidate,
                    labels_review,
                    legend_review,
                    mask,
                )
            ):
                raise ValueError("--auto-roi cannot be combined with saved reviews or a mask")
            hints = DetectionHints.model_validate(payload)
            requested_identifiers = any(k.startswith("labels_") for k in hints.requested) or (
                hints.labels is not None
                and (hints.labels.rows is not None or hints.labels.columns is not None)
            )
            if numeric_only and requested_identifiers:
                raise ValueError("--numeric-only conflicts with requested gene/sample identities")
            if not numeric_only and not requested_identifiers:
                raise ValueError(
                    "choose --numeric-only explicitly or request gene/sample identities"
                )
            saved = detect_file(
                figure, hints, out / "roi_review", config_path=config, overwrite=overwrite
            )
            assert saved.result.acceptance is not None
            typer.echo(f"M10 development decision: {saved.result.acceptance.overall.status}")
            typer.echo(
                f"Wrote diagnostics to {out / 'roi_review'}; M12 release gate is closed. "
                "No new matrix/quality outputs.",
                err=True,
            )
            raise typer.Exit(code=5)
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
        if settings.scale is not None and settings.scale.source == "ocr_ticks":
            outputs += [out / name for name in LEGEND_OUTPUT_NAMES]
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
            legend_review=legend_review,
        )
    except UnresolvedLegendError as error:
        typer.echo(
            f"Wrote legend review artifacts to {out}. {error}. No new matrix/quality outputs.",
            err=True,
        )
        raise typer.Exit(code=6) from None
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

    M10 decisions are experimental. Released automatic export awaits the M12 gate.
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
    if saved.result.acceptance is not None:
        typer.echo(
            f"M10 development decision: {saved.result.acceptance.overall.status}; "
            "automatic export disabled until M12 validation."
        )
        for decision in saved.result.acceptance.decisions:
            typer.echo(
                f"  {decision.requirement}: {decision.status} ({', '.join(decision.reasons)})"
            )
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


@app.command("legend")
def legend_command(
    figure: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    config: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)],
    out: Annotated[Path, typer.Option("--out", file_okay=False)] = Path("legend_review"),
    review: Annotated[Path | None, typer.Option("--review", exists=True, dir_okay=False)] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Recognise upright numeric ticks without requiring a heatmap; exit 6 until reviewed."""
    try:
        payload = json.loads(config.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "heatmap_bbox" in payload:
            extraction = ExtractionConfig.model_validate(payload)
            settings = LegendConfig(
                colourbar_bbox=extraction.colourbar_bbox,
                colourbar_orientation=extraction.colourbar_orientation,
                legend=extraction.scale.legend if extraction.scale else LegendOptions(),
                scale_model=extraction.scale.model if extraction.scale else "linear",
            )
        else:
            settings = LegendConfig.model_validate(payload)
        result = inspect_legend_file(
            figure, settings, out, config_path=config, review_path=review, overwrite=overwrite
        )
    except (OSError, ValueError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(f"Wrote {len(result.ticks)} tick observations to {out}; {result.status}")
    if result.fit is not None:
        typer.echo(
            f"Linear fit: value = {result.fit.slope:.6g} × position "
            f"+ {result.fit.intercept:.6g}; span={result.fit.recognised_span}"
        )
    for reason in result.reasons:
        typer.echo(f"Scale: {reason}")
    if result.status != "reviewed":
        raise typer.Exit(code=6)


if __name__ == "__main__":
    app()
