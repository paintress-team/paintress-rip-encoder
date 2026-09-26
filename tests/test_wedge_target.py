# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the wedge calibration target generator (rip.py --target wedge).

Builds a wedge, halftones it with the production dither, and checks that the
printed coverage of each patch tracks its commanded coverage (monotonic,
near-identity for the raw target), the endpoints are blank/solid, and the
fiducials are solid K. Runs under pytest or as a plain script.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))

import rip  # noqa: E402


def _measure(dpi=90, steps=33):
    channels, layout = rip.build_wedge_channels(dpi=dpi, steps=steps, gap_mm=2.0,
                                                max_x_mm=200.0, max_y_mm=200.0)
    dither = rip.DitherConfig(method="floyd_steinberg")
    halftone = rip.convert_to_halftone_bitmaps(channels, dither)

    by_series = {}
    for p in layout["patches"]:
        ch = "C" if p["series"] == "NEUTRAL" else p["series"]  # neutral measured on C
        region = halftone[ch][p["y0"]:p["y1"], p["x0"]:p["x1"]]
        by_series.setdefault(p["series"], []).append((p["coverage"], float(region.mean())))
    for pts in by_series.values():
        pts.sort()
    return channels, layout, by_series


def test_wedge_patches_track_commanded_coverage():
    _, layout, by_series = _measure()
    assert set(by_series) == set(rip.WEDGE_SERIES)
    for series, pts in by_series.items():
        printed = [pr for _, pr in pts]
        assert printed[0] < 0.02, f"{series}: 0% patch not blank ({printed[0]:.3f})"
        assert printed[-1] > 0.98, f"{series}: 100% patch not solid ({printed[-1]:.3f})"
        assert all(printed[i + 1] >= printed[i] - 0.02 for i in range(len(printed) - 1)), \
            f"{series}: not monotonic"
        max_dev = max(abs(pr - c) for c, pr in pts)
        assert max_dev < 0.05, f"{series}: raw target deviates from identity ({max_dev:.3f})"


def test_wedge_layout_fits_box_and_is_roughly_square():
    dpi = 90
    _, layout, _ = _measure(dpi=dpi)
    width_mm = layout["image_width_px"] / dpi * 25.4
    height_mm = layout["image_height_px"] / dpi * 25.4
    assert width_mm <= 200.0 + 1e-6 and height_mm <= 200.0 + 1e-6
    # Roughly square: neither side more than ~2x the other.
    assert 0.5 <= width_mm / height_mm <= 2.0
    assert len(layout["patches"]) == len(rip.WEDGE_SERIES) * layout["steps"]
    assert layout["wrap"] * layout["rows_per_panel"] >= layout["steps"]


def test_wedge_fiducials_are_solid_k():
    channels, layout, _ = _measure()
    assert len(layout["fiducials"]) == 4
    for f in layout["fiducials"]:
        region = channels["K"][f["y0"]:f["y1"], f["x0"]:f["x1"]]
        assert region.min() == 255, "fiducial is not solid K"


if __name__ == "__main__":
    test_wedge_patches_track_commanded_coverage()
    print("PASS test_wedge_patches_track_commanded_coverage")
    test_wedge_layout_fits_box_and_is_roughly_square()
    print("PASS test_wedge_layout_fits_box_and_is_roughly_square")
    test_wedge_fiducials_are_solid_k()
    print("PASS test_wedge_fiducials_are_solid_k")
