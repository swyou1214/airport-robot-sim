"""Unit tests for nav_map's pure geometry and figure construction."""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nav_map as nm


SYNTH_INIT = {
    "roads": [[[0, 0], [28, 0]], [[14, 0], [14, 8]], [[0, 8], [28, 8]]],
    "route": [[0, 0], [14, 0], [14, 8]],
    "bay": [10.0, 20.0, 18.0, 26.0],
    "target": [14, 23],
    "car_colors": [[0.9, 0.2, 0.2], [0.2, 0.5, 0.9]],
}


def test_robot_marker_points_along_heading():
    # heading 0: apex directly ahead of the robot on +x
    pts = nm.robot_marker(5.0, 7.0, 0.0, size=0.85)
    apex = max(pts, key=lambda p: p[0])
    assert apex == pytest.approx((5.85, 7.0))

    # rotated 90 degrees: apex on +y
    pts = nm.robot_marker(5.0, 7.0, math.pi / 2, size=0.85)
    apex = max(pts, key=lambda p: p[1])
    assert apex[0] == pytest.approx(5.0)
    assert apex[1] == pytest.approx(7.85)


def test_road_polygon_width_and_orientation():
    poly = nm.road_polygon((0, 0), (10, 0), half_width=2.0)
    xs = sorted(p[0] for p in poly)
    ys = sorted(p[1] for p in poly)
    assert xs == pytest.approx([0, 0, 10, 10])
    assert ys == pytest.approx([-2, -2, 2, 2])


def test_road_polygon_zero_length_is_none():
    assert nm.road_polygon((3, 3), (3, 3)) is None


def test_apply_view_centres_on_robot():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    nm._view["span"] = 10.0
    nm.apply_view(ax, 5.0, 7.0)
    assert sum(ax.get_xlim()) / 2 == pytest.approx(5.0)
    assert sum(ax.get_ylim()) / 2 == pytest.approx(7.0)
    assert ax.get_xlim()[1] - ax.get_xlim()[0] == pytest.approx(20.0)
    plt.close(fig)


def test_build_figure_offscreen_and_max_span():
    fig, ax, art = nm.build_figure(SYNTH_INIT, controls=False)
    # art carries every dynamic element the render loop updates
    for key in ("trail", "cars", "robot", "banner", "readout", "world"):
        assert key in art
    assert len(art["cars"]) == len(SYNTH_INIT["car_colors"])
    # world extent: x 0..28, y 0..26, bay included, +/- road half width
    assert art["world"] == pytest.approx((-2, -2, 30, 28))
    # zoom-out limit derived from that extent
    assert nm._view["max_span"] == pytest.approx(32.0)
    import matplotlib.pyplot as plt
    plt.close(fig)
