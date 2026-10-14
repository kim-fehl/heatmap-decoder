"""Rebuild source tables with `uv run --with openpyxl python .../build_tables.py`."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import openpyxl
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent


def write_tsv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / "sources/elife-78496/elife-78496-fig5-data1-v2.xlsx"
    sheet = openpyxl.load_workbook(source, data_only=True)["Figure 5E"]
    out = ROOT / "elife-78496-fig5e"
    genes = (out / "rows.txt").read_text().splitlines()
    # Empty spacer columns G, L and P do not represent samples.
    source_columns = [3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 17, 18, 19]
    groups = [
        ("BMAd-Pnpla2+/+", "Ad libitum", 4),
        ("BMAd-Pnpla2+/+", "CR", 4),
        ("BMAd-Pnpla2-/-", "Ad libitum", 3),
        ("BMAd-Pnpla2-/-", "CR", 3),
    ]
    columns, mappings = [], []
    i = 0
    for genotype, diet, count in groups:
        for ordinal in range(1, count + 1):
            label = f"{genotype}__{diet}__{ordinal:02d}"
            columns.append(label)
            mappings.append(
                [
                    i,
                    label,
                    genotype,
                    diet,
                    ordinal,
                    "display_position_within_group",
                    sheet.title,
                    get_column_letter(source_columns[i]),
                ]
            )
            i += 1
    source_rows = {}
    for row in range(3, sheet.max_row + 1):
        gene = sheet.cell(row, 2).value
        if gene in source_rows:
            raise ValueError(f"Duplicate source gene: {gene}")
        source_rows[gene] = row
    assert len(set(genes)) == len(genes) == 49
    assert set(genes) == set(source_rows)
    truth = np.asarray(
        [[sheet.cell(source_rows[gene], col).value for col in source_columns] for gene in genes],
        dtype=float,
    )
    assert truth.shape == (49, 14) and np.isfinite(truth).all()
    # The workbook supplies plotted z-scores directly. Do not rescale or reorder samples.
    (out / "columns.txt").write_text("\n".join(columns) + "\n")
    write_tsv(
        out / "truth.tsv",
        ["gene", *columns],
        [[gene, *values] for gene, values in zip(genes, truth, strict=True)],
    )
    write_tsv(
        out / "row_mapping.tsv",
        ["display_index", "display_label", "source_id", "sheet", "source_row"],
        [[i, gene, gene, sheet.title, source_rows[gene]] for i, gene in enumerate(genes)],
    )
    write_tsv(
        out / "column_mapping.tsv",
        [
            "display_index",
            "display_label",
            "genotype",
            "diet",
            "group_ordinal",
            "ordinal_kind",
            "sheet",
            "source_column",
        ],
        mappings,
    )
    manifest = {
        "doi": "10.7554/eLife.78496",
        "panel": "Figure 5E",
        "attribution": "Li et al., eLife 11:e78496 (2022)",
        "license": "CC BY",
        "source_url": "https://cdn.elifesciences.org/articles/78496/elife-78496-fig5-data1-v2.xlsx",
        "image_origin": "User-supplied crop named elife-78496-fig5-v2.jpg",
        "source_workbook_sha256": digest(source),
        "image_sha256": digest(out / "figure.jpg"),
        "truth_sha256": digest(out / "truth.tsv"),
        "shape": list(truth.shape),
        "value_type": "author_supplied_plotted_z_score",
        "additional_transformation": "none",
        "row_order": "visually transcribed top to bottom; exact symbol join to Figure 5E sheet",
        "column_order": "left to right, C:F H:K M:O Q:S; 4/4/3/3 samples",
        "sample_identity": "original sample IDs absent; names use group plus display ordinal",
        "row_order_verified": True,
        "column_group_order_verified": True,
        "column_within_group_order_check": (
            "color agreement to supplied table; no sample IDs visible"
        ),
        "truth_range": [float(truth.min()), float(truth.max())],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    source = ROOT / "sources/elife-17290/elife-17290-fig2-data2-v1.xlsx"
    sheet = openpyxl.load_workbook(source, data_only=True).active
    out = ROOT / "elife-17290-fig2"
    records = []
    for row in range(2, sheet.max_row + 1):
        values = [sheet.cell(row, col).value for col in range(1, 6)]
        if all(value is None for value in values):
            continue
        assert isinstance(values[0], str) and isinstance(values[1], str)
        assert np.isfinite(np.asarray(values[2:], dtype=float)).all()
        records.append([sheet.title, row, *values])
    assert len(records) == 598
    write_tsv(
        out / "source_table.tsv",
        [
            "sheet",
            "source_row",
            "sgRNA_sequence",
            "gene",
            "genome_wide_GFP_SQSTM1_high_vs_low_log2FC",
            "mini_pool_GFP_SQSTM1_high_vs_low_log2FC",
            "mini_pool_SQSTM1_high_vs_low_log2FC",
        ],
        records,
    )
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "doi": "10.7554/eLife.17290",
                "attribution": "DeJesus et al., eLife 5:e17290 (2016)",
                "license": "CC BY",
                "source_workbook_sha256": digest(source),
                "source_url": "https://cdn.elifesciences.org/articles/17290/elife-17290-fig2-data2-v1.xlsx",
                "source_table_sha256": digest(out / "source_table.tsv"),
                "records": len(records),
                "status": "source_table_only_no_corresponding_image_attached",
                "source_direction": "high versus low",
                "row_order_verified": False,
                "note": (
                    "Retains sgRNA and gene keys, source order and signs; no guide aggregation."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print("Rebuilt 49 x 14 plotted z-scores and 598 CRISPR source records.")


if __name__ == "__main__":
    main()
