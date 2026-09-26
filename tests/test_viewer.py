# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""viewer.py must undo exactly what the RIP's pass slicing did.

The viewer is the tool used to trust a job before printing it, so it is tested
as a round trip: slice a halftone into passes with the RIP, reassemble every
channel with the viewer, and require the original bitmap back, on every head.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))

import rip  # noqa: E402
import viewer  # noqa: E402

CMYK = ("C", "M", "Y", "K")


def _halftone(height, width, seed):
    rng = np.random.default_rng(seed)
    return {ch: (rng.random((height, width)) < f).astype(np.uint8)
            for ch, f in zip(CMYK, (0.3, 0.4, 0.2, 0.5))}


def _round_trip(head, dpi, height, width, metadata_extra=None):
    config = rip.PrintheadConfig(dpi=dpi, head=head)
    halftone = _halftone(height, width, seed=dpi + height)
    passes = rip.generate_print_passes(halftone, config, CMYK)
    y_positions = rip.compute_pass_y_positions_mm(height / dpi * 25.4, config)
    metadata = {
        "dpi": dpi,
        "nozzle_count": config.geometry.max_group_nozzles,
        "image_height_px": height,
        "head_layout": config.geometry.to_metadata(),
    }
    metadata.update(metadata_extra or {})
    metadata = {k: v for k, v in metadata.items() if v is not None}

    for ci, channel in enumerate(CMYK):
        pitch_lines, first_nozzle, nozzle_count = viewer.channel_rows(metadata, channel)
        rebuilt = viewer.reconstruct_channel_from_passes(
            passes_data=passes,
            y_positions_mm=y_positions,
            channel_index=ci,
            dpi=dpi,
            nozzle_pitch_lines=pitch_lines,
            nozzle_count=nozzle_count,
            first_nozzle=first_nozzle,
            image_height_px=height,
        )
        assert np.array_equal(rebuilt, halftone[channel]), \
            f"{head} @ {dpi} dpi: channel {channel} did not survive the round trip"


def test_c6n90_round_trips():
    _round_trip("c6n90", dpi=630, height=700, width=16)


def test_c4n180_round_trips():
    """The case the old ``dpi // nozzle_count`` got wrong: 60 nozzles at
    180 npi, with slots stacked up the column and a lead-in below the image."""
    _round_trip("c4n180", dpi=720, height=500, width=16)


def test_a_payload_without_head_layout_reads_as_c6n90():
    _round_trip("c6n90", dpi=630, height=300, width=8,
                metadata_extra={"head_layout": None})


def test_an_unplumbed_plane_has_no_rows():
    config = rip.PrintheadConfig(dpi=630, head="c6n90",
                                 ink_map=("K", "Y", "-", "-", "C", "M"))
    metadata = {"dpi": 630, "nozzle_count": 90,
                "head_layout": config.geometry.to_metadata()}
    assert viewer.channel_rows(metadata, "LC") is None
    assert viewer.channel_rows(metadata, "C") == (7, 0, 90)
