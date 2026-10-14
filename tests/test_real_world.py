"""Verify published source values, ordering, image correspondence and group names."""

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

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
    for folder in ["elife-78496-fig5-supp3a", "elife-97830-fig4e", "elife-17290-fig2"]:
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
