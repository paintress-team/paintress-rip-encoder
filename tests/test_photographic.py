# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for the R6 photographic front-end: sRGB transfer, 16-bit load, and
linear-light resampling."""

import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))

import rip  # noqa: E402


def test_srgb_linear_roundtrip():
    x = np.linspace(0, 1, 256, dtype=np.float32)
    back = rip.linear_to_srgb(rip.srgb_to_linear(x))
    assert np.max(np.abs(back - x)) < 1e-4
    # 50% sRGB is ~0.214 in linear light (the whole point of the correction).
    assert abs(float(rip.srgb_to_linear(np.array([0.5]))[0]) - 0.214) < 0.01


def test_load_preserves_16bit():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "g16.png"
        # A 16-bit ramp with values a plain 8-bit load would collapse.
        ramp = (np.linspace(0, 65535, 256).astype(np.uint16))[None, :].repeat(4, 0)
        Image.fromarray(ramp, mode="I;16").save(p)
        rgb, _icc, cmyk = rip.load_image_rgb_float(str(p))
        assert cmyk is None and rgb.shape == (4, 256, 3)
        # Distinct low values survive (would all be 0 after 8-bit truncation).
        assert rgb[0, 1, 0] > 0 and rgb[0, 1, 0] < rgb[0, 2, 0]


def test_gcr_black_start_suppresses_highlight_k():
    neutral = np.array([0.1, 0.3, 0.6, 0.9], dtype=np.float32)
    # black_start = 0 -> K from the start (== neutral * strength).
    assert np.allclose(rip.gcr_black(neutral, 1.0, 0.0), neutral)
    # black_start = 0.4 -> no K below 0.4 (highlights stay CMY), K above.
    k = rip.gcr_black(neutral, 1.0, 0.4)
    assert k[0] == 0.0 and k[1] == 0.0 and k[2] > 0.0 and k[3] > k[2]


def test_resample_is_linear_light():
    # A fine black/white checkerboard averaged down should land near the
    # *linear* midpoint (sRGB ~0.735), not the gamma midpoint (0.5).
    board = np.indices((32, 32)).sum(axis=0) % 2  # 0/1 checker
    rgb = np.repeat(board[..., None].astype(np.float32), 3, axis=2)
    out = rip.resample_rgb_linear(rgb, 1, 1)
    v = float(out[0, 0, 0])
    assert v > 0.65, f"resample not in linear light (got {v:.3f}, gamma would be ~0.5)"


if __name__ == "__main__":
    test_srgb_linear_roundtrip()
    print("PASS test_srgb_linear_roundtrip")
    test_load_preserves_16bit()
    print("PASS test_load_preserves_16bit")
    test_gcr_black_start_suppresses_highlight_k()
    print("PASS test_gcr_black_start_suppresses_highlight_k")
    test_resample_is_linear_light()
    print("PASS test_resample_is_linear_light")
