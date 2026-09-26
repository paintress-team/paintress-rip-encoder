# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Tests for R3 halftoning improvements: serpentine Floyd-Steinberg and
per-channel decorrelated blue noise."""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))

import rip  # noqa: E402


def test_floyd_steinberg_preserves_mean_and_endpoints():
    # Flat fields: dithered coverage should track the input level.
    for level in (0, 64, 128, 200, 255):
        field = np.full((64, 64), level, dtype=np.uint8)
        bitmap = rip.apply_floyd_steinberg(field)
        assert set(np.unique(bitmap)).issubset({0, 1})
        assert abs(bitmap.mean() - level / 255.0) < 0.03, f"level {level}"
    assert rip.apply_floyd_steinberg(np.zeros((32, 32), np.uint8)).sum() == 0
    assert rip.apply_floyd_steinberg(np.full((32, 32), 255, np.uint8)).min() == 1


def test_floyd_steinberg_is_serpentine():
    # A serpentine scan is not left-right symmetric row-to-row: reversing the
    # input columns should not simply reverse the output (a plain raster would
    # be far more symmetric). We check the two differ meaningfully.
    rng = np.random.default_rng(0)
    field = rng.integers(0, 256, size=(48, 48)).astype(np.uint8)
    out = rip.apply_floyd_steinberg(field)
    out_flipped_input = rip.apply_floyd_steinberg(field[:, ::-1])
    disagree = np.mean(out != out_flipped_input[:, ::-1])
    assert disagree > 0.05, "output looks direction-independent (not serpentine)"


def test_blue_noise_textures_differ_by_seed():
    a = rip.get_blue_noise_texture(64, seed=42)
    b = rip.get_blue_noise_texture(64, seed=1013)
    assert a.shape == b.shape == (64, 64)
    assert np.mean(a != b) > 0.5, "different seeds gave near-identical textures"


def test_blue_noise_decorrelated_across_channels():
    # Two channels with identical mid-grey input must not print dot-on-dot.
    mid = np.full((64, 64), 128, dtype=np.uint8)
    channels = {"C": mid.copy(), "M": mid.copy(), "Y": mid.copy(), "K": mid.copy()}
    cfg = rip.DitherConfig(method="blue_noise", blue_noise_size=64)
    bm = rip.convert_to_halftone_bitmaps(channels, cfg)
    disagree = np.mean(bm["C"] != bm["M"])
    # Same texture/phase would give 0 disagreement; decorrelated ~ 2*p*(1-p).
    assert disagree > 0.2, f"C and M too correlated ({disagree:.3f}), dot-on-dot"


if __name__ == "__main__":
    test_floyd_steinberg_preserves_mean_and_endpoints()
    print("PASS test_floyd_steinberg_preserves_mean_and_endpoints")
    test_floyd_steinberg_is_serpentine()
    print("PASS test_floyd_steinberg_is_serpentine")
    test_blue_noise_textures_differ_by_seed()
    print("PASS test_blue_noise_textures_differ_by_seed")
    test_blue_noise_decorrelated_across_channels()
    print("PASS test_blue_noise_decorrelated_across_channels")
