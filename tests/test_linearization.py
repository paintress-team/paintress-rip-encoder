# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for applying measured linearisation LUTs in the RIP pipeline.

Covers LUT detection (measured_luts), the LUT/total-ink application helpers,
and that a profile carrying a complete DPI LUT switches the pipeline into the
linearised colour path (metadata provenance).
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))

import rip  # noqa: E402


def test_measured_luts_requires_full_cmyk_set():
    prof = rip.CalibrationProfile()                             # in-code: empty dpi_curves
    assert prof.measured_luts(630) is None                      # empty -> None

    prof.dpi_curves = {"630": {"C": list(range(256))}}          # incomplete
    assert prof.measured_luts(630) is None

    full = {ch: list(range(256)) for ch in ("C", "M", "Y", "K")}
    prof.dpi_curves = {"630": full}
    got = prof.measured_luts(630)
    assert got is not None and set(got) == {"C", "M", "Y", "K"}
    assert prof.measured_luts(720) is None                      # only 630 measured


def test_apply_linearization_luts_remaps_values():
    channels = {"C": np.array([[0, 128, 255]], dtype=np.uint8),
                "M": np.array([[0, 128, 255]], dtype=np.uint8),
                "Y": np.array([[10, 20, 30]], dtype=np.uint8),
                "K": np.array([[0, 0, 0]], dtype=np.uint8)}
    luts = {"C": [min(255, 2 * v) for v in range(256)],         # doubles (clamped)
            "M": list(range(256))}                              # identity
    out = rip.apply_linearization_luts(channels, luts)
    assert list(out["C"][0]) == [0, 255, 255]
    assert list(out["M"][0]) == [0, 128, 255]
    assert list(out["Y"][0]) == [10, 20, 30]                    # no LUT -> unchanged


def test_apply_total_ink_limit_caps_sum():
    channels = {ch: np.full((1, 1), 255, dtype=np.uint8) for ch in ("C", "M", "Y", "K")}
    out = rip.apply_total_ink_limit(channels, 0.90)
    total = sum(int(out[ch][0, 0]) for ch in ("C", "M", "Y", "K")) / 255.0
    assert abs(total - 0.90) < 0.02                             # 4x255 capped to 0.90


def test_pipeline_uses_linearised_path_when_lut_present():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # A small greyscale gradient input.
        grad = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (16, 1))
        img_path = tmp / "in.png"
        Image.fromarray(grad, mode="L").convert("RGB").save(img_path)

        prof = rip.CalibrationProfile.default()
        prof.dpi_curves = {"90": {ch: list(range(256)) for ch in ("C", "M", "Y", "K")}}

        out = tmp / "job.json"
        rip.process_image_for_printing(str(img_path), str(out), dpi=90, calibration=prof)
        header = json.loads(out.read_text())
        assert header["metadata"]["processing"]["colour_mode"] == "linearised"

        # Without a LUT it falls back to the heuristic path.
        prof.dpi_curves = {}
        out2 = tmp / "job2.json"
        rip.process_image_for_printing(str(img_path), str(out2), dpi=90, calibration=prof)
        header2 = json.loads(out2.read_text())
        assert header2["metadata"]["processing"]["colour_mode"] == "heuristic"


def test_murray_davies_modelled_lut():
    lut = rip.murray_davies_luts(720, 40.0)
    assert set(lut) == {"C", "M", "Y", "K"}
    arr = np.array(lut["C"])
    assert arr.shape == (256,) and np.all(np.diff(arr) >= -1)     # monotonic
    assert arr[0] <= 2 and arr[255] >= 250                        # spans the range
    # A bigger drop saturates faster -> less coverage for the same tone.
    small = np.array(rip.murray_davies_luts(720, 20.0)["C"])
    big = np.array(rip.murray_davies_luts(720, 80.0)["C"])
    assert big[128] < small[128]


def test_pipeline_uses_modelled_lut_from_drop_diameter():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        grad = np.tile(np.linspace(0, 255, 32, dtype=np.uint8), (16, 1))
        img_path = tmp / "in.png"
        Image.fromarray(grad, mode="L").convert("RGB").save(img_path)

        prof = rip.CalibrationProfile.default()
        prof.dpi_curves = {}
        prof.drop_diameter_um = 42.0        # no measured LUT for dpi 90 -> modelled
        out = tmp / "job.json"
        rip.process_image_for_printing(str(img_path), str(out), dpi=90, calibration=prof)
        proc = json.loads(out.read_text())["metadata"]["processing"]
        assert proc["colour_mode"] == "linearised" and proc["lut_source"] == "modelled"


if __name__ == "__main__":
    test_measured_luts_requires_full_cmyk_set()
    print("PASS test_measured_luts_requires_full_cmyk_set")
    test_apply_linearization_luts_remaps_values()
    print("PASS test_apply_linearization_luts_remaps_values")
    test_apply_total_ink_limit_caps_sum()
    print("PASS test_apply_total_ink_limit_caps_sum")
    test_pipeline_uses_linearised_path_when_lut_present()
    print("PASS test_pipeline_uses_linearised_path_when_lut_present")
    test_murray_davies_modelled_lut()
    print("PASS test_murray_davies_modelled_lut")
    test_pipeline_uses_modelled_lut_from_drop_diameter()
    print("PASS test_pipeline_uses_modelled_lut_from_drop_diameter")
