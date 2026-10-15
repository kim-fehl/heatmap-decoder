"""Verify published source values, ordering, image correspondence and group names."""

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest

from heatmap_decoder.extract import extract
from heatmap_decoder.group_labels import GroupLabelsRequest, infer_group_labels
from heatmap_decoder.io import load_file_labels, load_raster
from heatmap_decoder.models import ExtractionConfig

ROOT = Path(__file__).resolve().parents[1] / "examples/real_world"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_tsv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.reader(stream, delimiter="\t"))


def test_truth_matches_xlsx_cells_and_display_mapping():
    folder = ROOT / "elife-78496-fig5e"
    table = read_tsv(folder / "truth.tsv")
    rows = list(csv.DictReader((folder / "row_mapping.tsv").open(), delimiter="\t"))
    columns = list(csv.DictReader((folder / "column_mapping.tsv").open(), delimiter="\t"))
    source = ROOT / "sources/elife-78496/elife-78496-fig5-data1-v2.xlsx"
    # Independent XML reader: validate the exported values against the untouched
    # workbook rather than calling the table builder again.
    with zipfile.ZipFile(source) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets = workbook.findall("s:sheets/s:sheet", NS)
        assert sheets[2].attrib["name"] == "Figure 5E"
        xml = ET.fromstring(archive.read("xl/worksheets/sheet3.xml"))
        cells = {c.attrib["r"]: c for c in xml.findall("s:sheetData/s:row/s:c", NS)}
        strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in s.findall(".//s:t", NS)) for s in strings]
    assert len(rows) == 49 and len(columns) == 14
    assert [r[0] for r in table[1:]] == (folder / "rows.txt").read_text().splitlines()
    assert table[0][1:] == (folder / "columns.txt").read_text().splitlines()
    assert [c["source_column"] for c in columns] == list("CDEFHIJKMNOQRS")
    for index, (mapping, row) in enumerate(zip(rows, table[1:], strict=True)):
        assert int(mapping["display_index"]) == index
        assert row[0] == mapping["display_label"] == mapping["source_id"]
        source_row = mapping["source_row"]
        gene_cell = cells["B" + source_row]
        assert shared[int(gene_cell.find("s:v", NS).text)] == row[0]
        expected = [
            float(cells[c["source_column"] + source_row].find("s:v", NS).text) for c in columns
        ]
        np.testing.assert_allclose(np.array(row[1:], dtype=float), expected, rtol=0, atol=1e-12)
    manifest = json.loads((folder / "manifest.json").read_text())
    for path, key in [
        (source, "source_workbook_sha256"),
        (folder / "figure.jpg", "image_sha256"),
        (folder / "truth.tsv", "truth_sha256"),
    ]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[key]


def test_published_truth_agrees_with_independent_color_extraction():
    folder = ROOT / "elife-78496-fig5e"
    truth = np.array([row[1:] for row in read_tsv(folder / "truth.tsv")[1:]], dtype=float)
    config = ExtractionConfig.model_validate_json((folder / "config.json").read_bytes())
    labels, _ = load_file_labels(config, folder)
    result = extract(load_raster(folder / "figure.jpg"), config, file_labels=labels)
    valid = np.isfinite(result.values)
    assert truth.shape == result.values.shape == (49, 14)
    assert valid.mean() > 0.85
    # Wide enough for raster/library differences; reordered rows/columns produce
    # much larger residuals. Uncalibrated and ambiguous cells remain excluded.
    assert np.sqrt(np.mean((truth[valid] - result.values[valid]) ** 2)) < 0.1
    assert np.corrcoef(truth[valid], result.values[valid])[0, 1] > 0.99


def test_attached_color_legends_name_exact_groups():
    for folder, expected in [
        ("elife-97830-fig4e", ["High_SE"] * 13 + ["Low_SE"] * 13),
        (
            "elife-78496-fig5-supp3a",
            ["AL", "AL", "AL", "AL"] + ["CR"] * 4 + ["AL"] * 3 + ["CR"] * 3,
        ),
    ]:
        path = ROOT / folder
        request = GroupLabelsRequest.model_validate_json(
            (path / "groups-request.json").read_bytes()
        )
        result = infer_group_labels(load_raster(path / "figure.jpg"), request)
        assert result["complete"]
        assert [r["tracks"][0]["category"] for r in result["records"]] == expected
        assert [r["generated_name"] for r in result["records"]] == (
            path / "columns.txt"
        ).read_text().splitlines()
        assert len(set(r["generated_name"] for r in result["records"])) == len(expected)


def test_unmatched_images_never_claim_numeric_ground_truth():
    for folder in ["elife-78496-fig5-supp3a", "elife-97830-fig4e"]:
        assert not (ROOT / folder / "truth.tsv").exists()
    table = read_tsv(ROOT / "elife-17290-fig2/source_table.tsv")
    assert len(table) == 599  # header + 598 untouched source records
    assert table[1][2:4] == ["TCTGGAAACGGAGGATGCGG", "ATG9A"]
    assert table[-1][3] == "MED19"
    assert float(table[-1][4]) == -3.0082  # Preserve the published high-vs-low sign.
    source = ROOT / "sources/elife-17290/elife-17290-fig2-data2-v1.xlsx"
    with zipfile.ZipFile(source) as archive:
        xml = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        cells = {c.attrib["r"]: c for c in xml.findall("s:sheetData/s:row/s:c", NS)}
        strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in s.findall(".//s:t", NS)) for s in strings]
    for index, record in enumerate(table[1:], 2):
        assert record[0] == "high versus low" and int(record[1]) == index
        for column, value in zip("AB", record[2:4], strict=True):
            assert shared[int(cells[column + str(index)].find("s:v", NS).text)] == value
        expected = [float(cells[column + str(index)].find("s:v", NS).text) for column in "CDE"]
        np.testing.assert_allclose(np.array(record[4:], dtype=float), expected, rtol=0, atol=1e-12)
    manifest = json.loads((ROOT / "elife-17290-fig2/manifest.json").read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_workbook_sha256"]
    exported = ROOT / "elife-17290-fig2/source_table.tsv"
    assert hashlib.sha256(exported.read_bytes()).hexdigest() == manifest["source_table_sha256"]


def xlsx_sheet_cells(source, sheet_number):
    """Read source cells independently of openpyxl and the fixture builder."""
    with zipfile.ZipFile(source) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            shared = ["".join(s.itertext()) for s in strings]
        xml = ET.fromstring(archive.read(f"xl/worksheets/sheet{sheet_number}.xml"))
        result = {}
        for c in xml.findall("s:sheetData/s:row/s:c", NS):
            value = c.find("s:v", NS)
            if c.get("t") == "s":
                result[c.attrib["r"]] = shared[int(value.text)]
            elif c.get("t") == "inlineStr":
                result[c.attrib["r"]] = "".join(c.find("s:is", NS).itertext())
            elif value is not None:
                result[c.attrib["r"]] = float(value.text)
    return result


def test_panel_g_truth_preserves_displayed_guide_records():
    folder = ROOT / "elife-17290-fig2"
    source = ROOT / "sources/elife-17290/elife-17290-fig2-data2-v1.xlsx"
    cells = xlsx_sheet_cells(source, 1)
    table = read_tsv(folder / "truth.tsv")
    mapping = list(csv.DictReader((folder / "row_mapping.tsv").open(), delimiter="\t"))
    assert len(mapping) == len(table) - 1 == 20
    assert table[0][1:] == (folder / "columns.txt").read_text().splitlines()
    assert [r[0] for r in table[1:]] == (folder / "rows.txt").read_text().splitlines()
    for i, (record, row) in enumerate(zip(mapping, table[1:], strict=True)):
        n = str(i + 2)
        assert int(record["display_index"]) == i
        assert record["source_row"] == n
        assert record["sgRNA_sequence"] == cells["A" + n]
        assert record["gene"] == row[0] == cells["B" + n]
        np.testing.assert_allclose(
            np.array(row[1:], float), [cells[col + n] for col in "CDE"], rtol=0, atol=1e-12
        )
    # The publication lists one guide twice with distinct gene names. Do not deduplicate it.
    assert mapping[6]["sgRNA_sequence"] == mapping[7]["sgRNA_sequence"]
    assert table[7][1:] == table[8][1:]
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, key in [("figure.jpg", "image_sha256"), ("truth.tsv", "truth_sha256")]:
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == manifest[key]


def test_panel_g_image_values_match_source():
    folder = ROOT / "elife-17290-fig2"
    truth = np.array([r[1:] for r in read_tsv(folder / "truth.tsv")[1:]], float)
    config = ExtractionConfig.model_validate_json((folder / "config.json").read_bytes())
    labels, _ = load_file_labels(config, folder)
    result = extract(load_raster(folder / "figure.jpg"), config, file_labels=labels)
    valid = np.isfinite(result.values)
    assert result.values.shape == (20, 3)
    assert valid.sum() >= 54
    assert np.sqrt(np.mean((truth[valid] - result.values[valid]) ** 2)) < 0.3
    assert np.corrcoef(truth[valid], result.values[valid])[0, 1] > 0.98


def test_blood_source_values_and_group_membership_without_inventing_order():
    folder = ROOT / "elife-97830-fig4e"
    source = ROOT / "elife-97830-fig4-data1-v1.xlsx"
    cells = xlsx_sheet_cells(source, 2)
    table = read_tsv(folder / "source_expression.tsv")
    mapping = list(csv.DictReader((folder / "source_row_mapping.tsv").open(), delimiter="\t"))
    samples = list(csv.DictReader((folder / "source_samples.tsv").open(), delimiter="\t"))
    assert len(mapping) == 25 and len(samples) == 39
    assert table[0][1:] == [s["sample"] for s in samples]
    group_cells = xlsx_sheet_cells(source, 6)
    expected_groups = {group_cells[f"B{i}"]: group_cells[f"A{i}"] for i in range(2, 41)}
    assert sum(s["group"] == "HIGH_SE" for s in samples) == 13
    assert sum(s["group"] == "LOW_SE" for s in samples) == 14
    for sample in samples:
        assert sample["group"] == expected_groups[sample["sample"]]
        assert sample["sample"] == cells[sample["source_column"] + "1"]
    for record, row in zip(mapping, table[1:], strict=True):
        n = record["source_row"]
        assert row[0] == record["gene"] == cells["B" + n]
        assert record["ENSID"] == cells["A" + n]
        np.testing.assert_allclose(
            np.array(row[1:], float),
            [cells[s["source_column"] + n] for s in samples],
            rtol=0,
            atol=1e-12,
        )
    manifest = json.loads((folder / "source_manifest.json").read_text())
    assert not manifest["column_order_verified"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_workbook_sha256"]
    assert (
        hashlib.sha256((folder / "source_expression.tsv").read_bytes()).hexdigest()
        == (manifest["source_expression_sha256"])
    )


# Keep real-engine OCR separate from deterministic numeric/source tests.
@pytest.mark.ocr_integration
def test_publication_ocr_against_reviewed_labels():
    import os

    from heatmap_decoder.extract import extract_labels

    if os.environ.get("HEATMAP_DECODER_OCR_TESTS") != "1":
        pytest.skip("opt in with HEATMAP_DECODER_OCR_TESTS=1 for real RapidOCR")
    for name, minimum_accepted, minimum_raw in [
        ("elife-17290-fig2", 9, 18),
        ("elife-78496-fig5e", 44, 45),
        ("elife-97830-fig4e", 25, 25),
    ]:
        folder = ROOT / name
        genes = (folder / "rows.txt").read_text().splitlines()
        config = ExtractionConfig.model_validate_json((folder / "labels-ocr.json").read_bytes())
        slots = extract_labels(load_raster(folder / "figure.jpg"), config).axes[0].slots
        assert len(slots) == len(genes)
        assert sum(g == s.final_text for g, s in zip(genes, slots, strict=True)) >= minimum_accepted
        assert sum(g == s.raw_text for g, s in zip(genes, slots, strict=True)) >= minimum_raw
        if name == "elife-17290-fig2":
            assert all(
                s.final_text is None or s.final_text == g for g, s in zip(genes, slots, strict=True)
            )


def test_all_uploaded_workbooks_have_verified_inventory():
    inventory = json.loads((ROOT / "workbooks.json").read_text())
    assert {r["path"] for r in inventory} == {p.name for p in ROOT.glob("*.xlsx")}
    for record in inventory:
        path = ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
        with zipfile.ZipFile(path) as archive:
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            sheets = workbook.findall("s:sheets/s:sheet", NS)
            assert [s.attrib["name"] for s in sheets] == [s["name"] for s in record["sheets"]]
    for article, filename in [
        ("elife-17290", "elife-17290-fig2-data2-v1"),
        ("elife-78496", "elife-78496-fig5-data1-v2"),
    ]:
        assert (ROOT / (filename + "(1).xlsx")).read_bytes() == (
            ROOT / "sources" / article / (filename + ".xlsx")
        ).read_bytes()
