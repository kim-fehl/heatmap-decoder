"""Shared accessible ROI/state and OCR/confidence drawing conventions."""

from collections.abc import Sequence

from PIL import Image, ImageDraw, ImageFont

ROI_STYLE = {
    "heatmap": ("Heatmap", "#2866c7"),
    "labels_rows": ("Gene labels", "#bd4189"),
    "labels_columns": ("Sample labels", "#9370bb"),
    "colourbar": ("Colourbar", "#6840a8"),
    "tick_search": ("Tick search", "#a05099"),
}


class DiagnosticCanvas:
    """Keep source at (0,0); place all titles in a collision-free white side ledger."""

    def __init__(self, source: Image.Image, lines: int = 12) -> None:
        self.width = source.width
        self.image = Image.new(
            "RGB", (source.width + 520, max(source.height, 100 + lines * 24)), "white"
        )
        self.image.paste(source.convert("RGB"), (0, 0))
        self.draw = ImageDraw.Draw(self.image)
        self.font: ImageFont.FreeTypeFont | ImageFont.ImageFont
        try:
            self.font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except OSError:
            self.font = ImageFont.load_default()
        self.y = 10
        self.note("OCR confidence: H teal / M yellow / L crimson / ? dashed")
        self.note("Confidence is distinct from status; verify text and signs.")

    def note(self, text: str, colour: str = "black") -> None:
        self.draw.rectangle((self.width + 4, self.y + 2, self.width + 9, self.y + 10), fill=colour)
        self.draw.text((self.width + 16, self.y), text, fill="black", font=self.font)
        self.y += 24

    def frame(self, box: Sequence[float], colour: str, dashed: bool = False) -> None:
        x, y, r, s = box
        r, s = max(x, r - 1), max(y, s - 1)
        if dashed:
            for start, end in [
                ((x, y), (r, y)),
                ((x, s), (r, s)),
                ((x, y), (x, s)),
                ((r, y), (r, s)),
            ]:
                length = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
                for offset in range(0, int(length), 10):
                    a, b = offset / max(1, length), min(offset + 5, length) / max(1, length)
                    self.draw.line(
                        (
                            start[0] + a * (end[0] - start[0]),
                            start[1] + a * (end[1] - start[1]),
                            start[0] + b * (end[0] - start[0]),
                            start[1] + b * (end[1] - start[1]),
                        ),
                        fill=colour,
                        width=2,
                    )
        else:
            self.draw.rectangle((x, y, r, s), outline=colour, width=2)

    def roi(self, box: Sequence[float], kind: str, state: str, identifier: str = "") -> None:
        title, colour = ROI_STYLE[kind]
        self.frame(box, colour, dashed=state == "proposed")
        self.note(f"{identifier} {title} [{state}] {tuple(box)}", colour)

    def ocr(
        self,
        box: Sequence[float] | None,
        confidence: float | None,
        identifier: str,
        text: str,
        status: str,
        minimum: float = 80,
        high: float = 90,
    ) -> None:
        if not 0 <= minimum <= high <= 100:
            raise ValueError("invalid confidence band ordering")
        unknown = confidence is None or confidence < 0
        score = confidence if confidence is not None else -1
        band, colour = (
            ("?", "#666666")
            if unknown
            else ("H", "#007e80")
            if score >= high
            else ("M", "#b58d00")
            if score >= minimum
            else ("L", "#c52647")
        )
        if box is not None:
            self.frame(box, colour, dashed=unknown)
        shown = text.replace("\n", " | ")[:56]
        self.note(f"{identifier} {band} {confidence} [{status}] {shown}", colour)
