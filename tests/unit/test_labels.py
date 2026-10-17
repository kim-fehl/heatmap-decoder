"""M4 geometry acceptance tests: fake evidence, no Tesseract/pytesseract."""

import unittest

from PIL import Image
from pydantic import ValidationError

from heatmap_decoder.geometry import axis_slots, cell_slices, validate_grid_geometry
from heatmap_decoder.labels import (
    CropTransform,
    assign_candidates,
    crop_label_region,
    generated_labels,
    group_tokens,
    labels_from_tokens,
    transform_tokens,
)
from heatmap_decoder.models import (
    ExtractionConfig,
    GridConfig,
    LabelCandidate,
    OCRConfig,
    OCRToken,
)


def grid(**changes):
    values = dict(heatmap_bbox=(100, 80, 220, 140), rows=3, columns=3)
    values.update(changes)
    return GridConfig(**values)


def candidate(text, box, confidence=95):
    return LabelCandidate(text, box, confidence, ())


class LabelGeometryTests(unittest.TestCase):
    def test_grid_matches_numeric_sampling(self):
        config = ExtractionConfig(
            **grid().model_dump(),
            colourbar_bbox=(240, 80, 250, 140),
            colourbar_orientation="vertical",
            colourbar_start_value=1,
            colourbar_end_value=-1,
        )
        self.assertEqual([s.centre for s in axis_slots(config, "rows")], [90, 110, 130])
        self.assertEqual([s.centre for s in axis_slots(config, "columns")], [120, 160, 200])
        self.assertEqual(cell_slices(config)[0], (slice(85, 95), slice(110, 130)))

    def test_fractional_slot_geometry(self):
        slots = axis_slots(grid(heatmap_bbox=(100, 80, 221, 141)), "columns")
        self.assertEqual(len(slots), 3)
        self.assertEqual(slots[0].start, 100)
        self.assertEqual(slots[-1].end, 221)
        self.assertEqual(slots[1].start, slots[0].end)

    def test_invalid_configuration(self):
        invalid = [
            {"rows": {"source": "ocr", "bbox": [0, 0, 0, 10]}},
            {"rows": {"source": "ocr", "bbox": [-1, 0, 2, 10]}},
            {"rows": {"source": "ocr", "bbox": [0, 0, 2, 10], "path": "x"}},
            {"rows": {"source": "file", "path": "x", "bbox": [0, 0, 2, 10]}},
            {"rows": {"source": "file", "path": ""}},
            {"rows": {"source": "ocr"}},
            {"rows": {"source": "auto", "bbox": [0, 0, 2, 10]}},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                grid(labels=payload)
        for payload in [
            {"upscale_factor": 0},
            {"upscale_factor": 1.5},
            {"upscale_factor": True},
            {"min_confidence": -1},
            {"min_confidence": 101},
            {"min_confidence": float("nan")},
            {"max_axis_offset_fraction": 0},
            {"max_axis_offset_fraction": 0.51},
            {"language": ""},
        ]:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                OCRConfig(**payload)
        self.assertEqual(OCRConfig(max_axis_offset_fraction=0.5).max_axis_offset_fraction, 0.5)

    def test_file_and_omitted_config(self):
        config = grid(labels={"rows": {"source": "file", "path": "genes.txt"}})
        self.assertEqual(config.labels.rows.path, "genes.txt")
        self.assertIsNone(config.labels.columns)
        self.assertEqual(GridConfig.model_validate_json(config.model_dump_json()), config)
        generated = generated_labels(config, "columns")
        self.assertEqual(
            [r.final_text for r in generated.slots], ["column_0", "column_1", "column_2"]
        )
        self.assertTrue(all(r.status == "generated" for r in generated.slots))

    def test_image_bounds_before_crop(self):
        for axis in ("rows", "columns"):
            config = grid(labels={axis: {"source": "ocr", "bbox": [0, 0, 301, 20]}})
            with self.subTest(axis=axis), self.assertRaisesRegex(ValueError, f"labels.{axis}"):
                crop_label_region(Image.new("RGB", (300, 200)), config, axis)
        with self.assertRaisesRegex(ValueError, "heatmap"):
            validate_grid_geometry(grid(), (100, 200))
        with self.assertRaisesRegex(ValueError, "one pixel"):
            validate_grid_geometry(grid(rows=61), (200, 300))

    def test_crop_transform_roundtrip(self):
        config = grid(
            labels={
                "rows": {"source": "ocr", "bbox": [220, 80, 300, 140]},
                "ocr": {"upscale_factor": 3},
            }
        )
        image = Image.new("RGB", (300, 200), "white")
        image.putpixel((220, 80), (0, 0, 0))
        crop, transform = crop_label_region(image, config, "rows")
        self.assertEqual(crop.size, (240, 180))
        box = (3.0, 18.0, 90.0, 42.0)
        original = transform.to_original(box)
        self.assertEqual(original, (221, 86, 250, 94))
        self.assertEqual(transform.to_crop(original), box)
        raw = (OCRToken(" TP53 ", box, 97, (1, 2, 3)), OCRToken("", (0, 0, 0, 0), -1))
        converted = transform_tokens(raw, transform)
        self.assertEqual(converted[0].bbox, original)
        self.assertEqual(converted[0].text, " TP53 ")
        self.assertEqual(converted[0].line_id, (1, 2, 3))
        self.assertEqual(raw[0].bbox, box)
        self.assertEqual(converted[1], raw[1])
        with self.assertRaisesRegex(ValueError, "resized crop"):
            transform.to_original((-1, 0, 20, 20))
        with self.assertRaises(ValueError):
            CropTransform((0, 0, 10, 10), 0)

    def test_left_and_right_rows(self):
        for x in (20, 230):
            config = grid(labels={"rows": {"source": "ocr", "bbox": [x, 80, x + 60, 140]}})
            tokens = [
                OCRToken(text, (x, y - 4, x + 30, y + 4), 96)
                for text, y in [("TP53", 90), ("HLA-DRA", 110), ("Ab_1", 130)]
            ]
            result = labels_from_tokens(tokens, config, "rows")
            self.assertEqual([r.final_text for r in result.slots], ["TP53", "HLA-DRA", "Ab_1"])
            self.assertEqual([r.index for r in result.slots], [0, 1, 2])

    def test_above_and_below_columns(self):
        for y in (40, 160):
            tokens = [
                OCRToken(text, (x - 12, y, x + 12, y + 8), 94)
                for text, x in [("a-1", 120), ("B_2", 160), ("s03", 200)]
            ]
            config = grid(labels={"columns": {"source": "ocr", "bbox": [100, y, 220, y + 10]}})
            result = labels_from_tokens(tokens, config, "columns")
            self.assertEqual([r.final_text for r in result.slots], ["a-1", "B_2", "s03"])

    def test_multitoken_spelling_confidence_and_membership(self):
        tokens = [
            OCRToken(" beta_2 ", (122, 40, 137, 48), 85, (1,)),
            OCRToken("Sample", (102, 41, 119, 49), 99, (7,)),
            OCRToken("", (0, 0, 0, 0), -1),
            OCRToken(" ignored ", (0, 0, 0, 0), -1),
        ]
        candidates = group_tokens(tokens, grid(), "columns")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].text, "Sample beta_2")
        self.assertEqual(candidates[0].confidence, 85)
        self.assertEqual(candidates[0].token_indices, (1, 0))
        internal = labels_from_tokens(
            [OCRToken(" a  b-0_I1 ", (105, 40, 135, 48), 99)], grid(), "columns"
        )
        self.assertEqual(internal.slots[0].final_text, "a  b-0_I1")

    def test_engine_line_id_does_not_merge_rows(self):
        tokens = [
            OCRToken("A", (20, 86, 30, 94), 99, (1,)),
            OCRToken("B", (20, 106, 30, 114), 99, (1,)),
        ]
        self.assertEqual(len(group_tokens(tokens, grid(), "rows")), 2)

    def test_missing_low_confidence_and_duplicate_order(self):
        candidates = [
            candidate("same", (20, 126, 30, 134)),
            candidate("same", (20, 86, 30, 94)),
            candidate("weak", (20, 106, 30, 114), 79),
        ]
        result = assign_candidates(candidates, grid(), "rows")
        self.assertEqual(
            [r.status for r in result.slots], ["accepted", "low_confidence", "accepted"]
        )
        self.assertEqual([r.final_text for r in result.slots], ["same", None, "same"])
        self.assertTrue(result.slots[0].duplicate)
        self.assertTrue(result.slots[2].duplicate)
        self.assertEqual(result.slots[1].raw_text, "weak")
        missing = assign_candidates(candidates[:1], grid(), "rows")
        self.assertEqual([r.status for r in missing.slots], ["missing", "missing", "accepted"])
        self.assertEqual(len(missing.slots), 3)

    def test_collision_never_chosen_by_confidence(self):
        result = assign_candidates(
            [
                candidate("A", (20, 86, 30, 94), 99),
                candidate("B", (60, 86, 70, 94), 40),
                candidate("C", (20, 126, 30, 134)),
            ],
            grid(),
            "rows",
        )
        self.assertEqual([r.status for r in result.slots], ["ambiguous", "missing", "accepted"])
        self.assertEqual(result.slots[0].candidate_indices, (0, 1))
        self.assertEqual(result.slots[0].raw_text, "A\nB")
        self.assertIsNone(result.slots[0].final_text)

    def test_spanning_and_outside_candidates_keep_evidence(self):
        candidates = [
            candidate("spans", (110, 40, 145, 48)),
            candidate("outside", (250, 40, 260, 48)),
        ]
        result = assign_candidates(candidates, grid(), "columns")
        self.assertEqual([r.status for r in result.slots], ["ambiguous", "ambiguous", "missing"])
        self.assertEqual(result.unmatched_candidate_indices, (0, 1))
        self.assertEqual(result.candidates, tuple(candidates))
        self.assertIn("spans_slots", result.slots[0].issues)

    def test_offset_extent_and_half_pitch_tie(self):
        result = assign_candidates([candidate("offset", (20, 80, 30, 82))], grid(), "rows")
        self.assertEqual(result.slots[0].status, "ambiguous")
        self.assertIn("axis_offset", result.slots[0].issues)
        config = grid(labels={"ocr": {"max_axis_offset_fraction": 0.5}})
        tie = assign_candidates([candidate("tie", (20, 96, 30, 104))], config, "rows")
        self.assertEqual([r.status for r in tie.slots], ["ambiguous", "ambiguous", "missing"])
        boundary = assign_candidates([candidate("band", (20, 80, 30, 100))], grid(), "rows")
        self.assertEqual(boundary.slots[0].status, "accepted")

    def test_neighboring_column_words_not_joined(self):
        tokens = [
            OCRToken("one", (124, 40, 138, 50), 99, (1,)),
            OCRToken("two", (142, 40, 156, 50), 99, (1,)),
        ]
        self.assertEqual(len(group_tokens(tokens, grid(), "columns")), 2)
        result = labels_from_tokens(tokens, grid(), "columns")
        self.assertEqual([r.final_text for r in result.slots], ["one", "two", None])


if __name__ == "__main__":
    unittest.main()
