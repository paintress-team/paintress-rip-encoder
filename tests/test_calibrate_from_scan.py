# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Offline test for tools/calibrate_from_scan.py using a synthetic scan.

We render a wedge target through a *known* non-linear per-channel device model
(so we know the ground-truth response), rotate + noise it like a real flatbed
scan, then check that calibrate_from_scan:

  * finds the fiducials and registers the target despite the rotation,
  * recovers a monotonic LUT spanning 0..255,
  * and, composed back with the device model, linearises perceptual tone
    (small RMSE from an ideal ramp).

Runs under pytest or as a plain script.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))
sys.path.insert(0, str(REPO / "tools"))

import rip  # noqa: E402
import calibrate_from_scan as cal  # noqa: E402

# Ground-truth device model per channel: reflectance R(c) = 1 - (1-Rmin)*c^gamma
# in the channel the ink absorbs. Deliberately non-linear and per-channel.
DEVICE = {
    "C": {"gamma": 0.75, "rmin": 0.10, "chan": 0},
    "M": {"gamma": 1.30, "rmin": 0.12, "chan": 1},
    "Y": {"gamma": 0.65, "rmin": 0.25, "chan": 2},
    "K": {"gamma": 1.50, "rmin": 0.05, "chan": None},  # absorbs all channels
}
PAPER = 245.0


def _reflectance(series: str, coverage: float) -> float:
    d = DEVICE["K"] if series == "NEUTRAL" else DEVICE[series]
    return 1.0 - (1.0 - d["rmin"]) * (coverage ** d["gamma"])


def _render_synthetic_scan(layout) -> Image.Image:
    """Paint each patch with its device-model reflectance, add fiducials."""
    W = layout["image_width_px"]
    H = layout["image_height_px"]
    img = np.full((H, W, 3), PAPER, dtype=np.float32)

    for p in layout["patches"]:
        series, cov = p["series"], p["coverage"]
        refl = _reflectance(series, cov)
        val = PAPER * refl
        reg = img[p["y0"]:p["y1"], p["x0"]:p["x1"]]
        if series == "NEUTRAL" or series == "K":
            reg[:] = val  # neutral/black reduce all channels
        else:
            reg[..., DEVICE[series]["chan"]] = val  # selective absorption

    for f in layout["fiducials"]:
        img[f["y0"]:f["y1"], f["x0"]:f["x1"]] = 8.0  # near-black fiducials

    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="RGB")


def _make_scan_and_payload(tmp: Path, rotate_deg: float = 1.5, streak_frac: float = 0.0):
    """Write a wedge payload and a rotated+noisy synthetic scan of it.

    The scan is rendered from the payload's ``metadata.target`` (which is in
    *printed* (vertically flipped) space), so it matches what a real scan of
    the sheet placed roughly upright looks like. ``rotate_deg`` is the scanner
    skew. ``streak_frac`` blanks that fraction of raster rows to paper white,
    simulating dead/clogged nozzles (bright streaks through every patch).
    """
    import json
    payload = tmp / "wedge.json"
    rip.process_target_for_printing(str(payload), dpi=90, max_x_mm=200, max_y_mm=200)
    layout = json.loads(payload.read_text())["metadata"]["target"]

    base = _render_synthetic_scan(layout)
    if streak_frac > 0:
        arr = np.asarray(base).copy()
        rng = np.random.default_rng(99)
        rows = rng.choice(arr.shape[0], int(streak_frac * arr.shape[0]), replace=False)
        arr[rows, :, :] = int(PAPER)  # dead-nozzle streaks
        base = Image.fromarray(arr, "RGB")
    # Emulate the scanner: pad, rotate a little (skew), add mild noise.
    canvas = Image.new("RGB", (base.width + 60, base.height + 60), (int(PAPER),) * 3)
    canvas.paste(base, (30, 30))
    rotated = canvas.rotate(rotate_deg, resample=Image.BILINEAR, expand=True,
                            fillcolor=(int(PAPER),) * 3)
    arr = np.asarray(rotated, dtype=np.float32)
    rng = np.random.default_rng(1234)
    arr = np.clip(arr + rng.normal(0, 2.0, arr.shape), 0, 255).astype(np.uint8)
    scan_path = tmp / f"scan_{int(rotate_deg)}.png"
    Image.fromarray(arr, mode="RGB").save(scan_path)
    return scan_path, payload


def _lstar_darkness(series: str, coverage: np.ndarray) -> np.ndarray:
    """Ground-truth perceptual darkness (0..1) the tool aims to linearise."""
    refl = np.array([_reflectance(series, float(c)) for c in coverage])
    lstar = cal.reflectance_to_lstar(refl)
    l_white = cal.reflectance_to_lstar(np.array([1.0]))[0]
    l_dark = cal.reflectance_to_lstar(np.array([_reflectance(series, 1.0)]))[0]
    return (l_white - lstar) / (l_white - l_dark)


def test_calibrate_recovers_linearising_lut():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        scan_path, payload = _make_scan_and_payload(tmp)
        dpi, luts, report = cal.calibrate(str(scan_path), str(payload), inset=0.3)

        assert dpi == 90
        assert set(luts) == set(cal.LUT_SERIES)

        for series in cal.LUT_SERIES:
            lut = np.array(luts[series])
            assert lut.shape == (256,)
            # Monotonic non-decreasing.
            assert np.all(np.diff(lut) >= -1), f"{series}: LUT not monotonic"
            # Spans the range: blank in, blank out; full in, full out.
            assert lut[0] <= 3, f"{series}: LUT[0]={lut[0]}"
            assert lut[255] >= 250, f"{series}: LUT[255]={lut[255]}"

            # Closed loop: apply LUT, print through the device, measure tone.
            t = np.linspace(0, 1, 33)
            commanded = lut[np.rint(t * 255).astype(int)] / 255.0
            achieved = _lstar_darkness(series, commanded)
            rmse = float(np.sqrt(np.mean((achieved - t) ** 2)))
            assert rmse < 0.05, f"{series}: linearisation RMSE {rmse:.3f} too high"


def test_calibrate_tolerates_scan_skew():
    # A noticeably skewed (but roughly upright) scan must still register and
    # calibrate: position-based fiducial matching tolerates real scanner skew.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        scan_path, payload = _make_scan_and_payload(tmp, rotate_deg=6.0)
        dpi, luts, report = cal.calibrate(str(scan_path), str(payload), inset=0.3)
        for series in cal.LUT_SERIES:
            lut = np.array(luts[series])
            assert lut[0] <= 3 and lut[255] >= 250, f"{series}: bad endpoints under skew"
            t = np.linspace(0, 1, 33)
            commanded = lut[np.rint(t * 255).astype(int)] / 255.0
            rmse = float(np.sqrt(np.mean((_lstar_darkness(series, commanded) - t) ** 2)))
            assert rmse < 0.05, f"{series}: skewed-scan RMSE {rmse:.3f} too high"


def test_calibrate_survives_nozzle_streaks():
    # ~8% of raster rows blanked (clogged nozzles). The robust per-row median
    # should ignore the bright streaks and still linearise.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        scan_path, payload = _make_scan_and_payload(tmp, streak_frac=0.08)
        dpi, luts, report = cal.calibrate(str(scan_path), str(payload), inset=0.3)
        assert not cal.degenerate_channels(luts), "streaks made the curve look degenerate"
        for series in cal.LUT_SERIES:
            lut = np.array(luts[series])
            t = np.linspace(0, 1, 33)
            commanded = lut[np.rint(t * 255).astype(int)] / 255.0
            rmse = float(np.sqrt(np.mean((_lstar_darkness(series, commanded) - t) ** 2)))
            # 8% of rows blanked on every channel is a stress test (a real head
            # clogs far less, and not on all channels); the robust measurement
            # keeps the linearisation usable rather than exact.
            assert rmse < 0.10, f"{series}: RMSE {rmse:.3f} under nozzle streaks"


def test_gray_balance_caps_the_strong_inks():
    # Under gray balance the C/M/Y LUTs share the weakest ink's dark end, so the
    # stronger inks cap below full coverage. In the device model Y is weakest
    # (rmin 0.25, i.e. lightest at full), C strongest (rmin 0.10).
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        scan_path, payload = _make_scan_and_payload(tmp)
        _, plain, _ = cal.calibrate(str(scan_path), str(payload))
        _, gb, _ = cal.calibrate(str(scan_path), str(payload), gray_balance=True)
        # Plain: every channel reaches ~full coverage at input 255.
        assert plain["C"][255] > 240 and plain["Y"][255] > 240
        # Gray-balanced: the weak ink (Y) stays highest, the strong ink (C) caps.
        assert gb["Y"][255] >= gb["C"][255] and gb["C"][255] < 220


def test_degenerate_guard_flags_bilevel_curve():
    # A near-bilevel curve (everything crammed to ~31% coverage) is flagged;
    # a healthy full-span curve is not.
    bad = {ch: [0] + [80 + (i * 4) // 255 for i in range(255)] for ch in cal.LUT_SERIES}
    flags = dict(cal.degenerate_channels(bad))
    assert set(flags) == set(cal.LUT_SERIES), "did not flag the degenerate curve"

    good = {ch: list(range(256)) for ch in cal.LUT_SERIES}
    assert not cal.degenerate_channels(good), "flagged a healthy identity curve"


if __name__ == "__main__":
    test_calibrate_recovers_linearising_lut()
    print("PASS test_calibrate_recovers_linearising_lut")
    test_calibrate_tolerates_scan_skew()
    print("PASS test_calibrate_tolerates_scan_skew")
    test_calibrate_survives_nozzle_streaks()
    print("PASS test_calibrate_survives_nozzle_streaks")
    test_gray_balance_caps_the_strong_inks()
    print("PASS test_gray_balance_caps_the_strong_inks")
    test_degenerate_guard_flags_bilevel_curve()
    print("PASS test_degenerate_guard_flags_bilevel_curve")
