"""Scale fixtures contracts and fixtures."""

from PIL import ImageFont

from tests.support import ROOT as REPO_ROOT

ROOT = REPO_ROOT


def case_for(index):
    return dict(
        index=index,
        seed=129000 + index,
        family=("vertical_flank", "horizontal_cap", "cross_axis")[index % 3],
        degradation="clean",
    )


try:
    ImageFont.truetype("DejaVuSans.ttf", 18)
    FONT_AVAILABLE = True
except OSError:
    FONT_AVAILABLE = False
