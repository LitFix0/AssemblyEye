import numpy as np
import pytest

from app.schemas.inspection import BBox
from app.schemas.spec import ColorModel
from app.services.verification.color import read_cable_color

MODEL = ColorModel(ranges={"yellow": [(20, 35)], "blue": [(100, 130)], "red": [(0, 8), (172, 179)]})
YELLOW, BLUE, RED, GREY = (0, 230, 240), (230, 40, 20), (20, 20, 225), (150, 150, 150)  # BGR


def canvas(w=400, h=200):
    return np.full((h, w, 3), 120, dtype=np.uint8)


def paint(img, x1, y1, x2, y2, bgr):
    img[y1:y2, x1:x2] = bgr
    return img


def box(x1, y1, x2, y2):
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2)


@pytest.mark.parametrize("bgr,name", [(YELLOW, "yellow"), (BLUE, "blue"), (RED, "red")])
def test_reads_each_cable_colour(bgr, name):
    img = paint(canvas(), 100, 80, 300, 120, bgr)
    r = read_cable_color(img, box(90, 70, 310, 130), [], MODEL)
    assert r.name == name and r.share > 0.95 and r.reason == ""


def test_connector_pixels_are_excluded():
    # a red connector overlaps the left end of a yellow cable's box; without exclusion red would dominate
    img = paint(canvas(), 150, 80, 300, 120, YELLOW)
    img = paint(img, 50, 60, 200, 140, RED)  # red covers 9000 px of the cable box, yellow keeps 4000
    cable = box(50, 70, 300, 130)
    assert read_cable_color(img, cable, [], MODEL).name == "red"
    r = read_cable_color(img, cable, [box(50, 60, 200, 140)], MODEL)
    assert r.name == "yellow" and r.n_saturated < r.n_pixels


def test_grey_cable_is_unknown_not_guessed():
    img = paint(canvas(), 100, 80, 300, 120, GREY)
    r = read_cable_color(img, box(90, 70, 310, 130), [], MODEL)
    assert r.name is None and "clearly coloured" in r.reason


def test_too_few_coloured_pixels_is_unknown():
    img = paint(canvas(), 100, 80, 103, 90, YELLOW)  # 30 coloured px, min_pixels is 40
    assert read_cable_color(img, box(90, 70, 310, 130), [], MODEL).name is None
    img = paint(canvas(), 100, 80, 104, 90, YELLOW)  # exactly 40: enough
    assert read_cable_color(img, box(90, 70, 310, 130), [], MODEL).name == "yellow"


def test_close_call_between_two_colours_is_unknown():
    # 55% yellow vs 45% blue: above min_share but inside the margin
    img = paint(canvas(), 100, 80, 210, 120, YELLOW)
    img = paint(img, 210, 80, 300, 120, BLUE)
    r = read_cable_color(img, box(90, 70, 310, 130), [], MODEL)
    assert r.name is None and 0.5 < r.share < 0.6


def test_two_colours_with_no_clear_winner_is_unknown():
    img = paint(canvas(), 100, 80, 200, 120, YELLOW)
    img = paint(img, 200, 80, 300, 120, BLUE)
    r = read_cable_color(img, box(90, 70, 310, 130), [], MODEL)
    assert r.name is None and "no colour wins" in r.reason
    assert abs(r.shares["yellow"] - 0.5) < 0.01 and abs(r.shares["blue"] - 0.5) < 0.01


def test_everything_excluded_and_outside_image():
    img = paint(canvas(), 100, 80, 300, 120, YELLOW)
    r = read_cable_color(img, box(100, 80, 300, 120), [box(0, 0, 400, 200)], MODEL)
    assert r.name is None and "no cable pixels" in r.reason
    assert read_cable_color(img, box(500, 500, 600, 600), [], MODEL).name is None


def test_rejects_bad_image_types():
    with pytest.raises(ValueError):
        read_cable_color(np.zeros((10, 10), dtype=np.uint8), box(0, 0, 5, 5), [], MODEL)
    with pytest.raises(ValueError):
        read_cable_color(np.zeros((10, 10, 3), dtype=np.float32), box(0, 0, 5, 5), [], MODEL)


def test_color_model_validation():
    with pytest.raises(ValueError):
        ColorModel(ranges={})
    with pytest.raises(ValueError):
        ColorModel(ranges={"x": [(30, 20)]})
    with pytest.raises(ValueError):
        ColorModel(ranges={"x": [(0, 200)]})
    with pytest.raises(ValueError):
        ColorModel(ranges={"x": [(0, 10)]}, min_share=0.8, confident_share=0.6)
    with pytest.raises(ValueError):
        ColorModel(ranges={"x": [(0, 10)]}, min_margin=1.5)
