# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Round-trip test for the RIP payload format (rip_payload.save_rip/load_rip).

Checks that bit-packed save + load reproduces the bitmaps exactly, that the
sidecar CRC is verified, and that JSON metadata survives. Runs under pytest or
as a plain script.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "rip"))

import rip_payload  # noqa: E402


def _sample_passes(n_passes=3, width=37, seed=1):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 2, size=(6, 90, width), dtype=np.uint8) for _ in range(n_passes)]


def test_round_trip(tmp_path=None):
    import tempfile
    out_dir = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    passes = _sample_passes()
    meta = {"dpi": 630, "nozzle_count": 90, "channel_order": ["C", "M", "Y", "K", "LC", "LM"]}
    y_pos = [0.0, 0.04, 0.08]
    y_del = [0.0, 0.04, 0.04]

    header = out_dir / "sample.json"
    rip_payload.save_rip(header, meta, y_pos, y_del, passes)

    loaded = rip_payload.load_rip(header)
    assert loaded["metadata"] == meta
    assert loaded["passes"]["y_positions_mm"] == y_pos
    assert loaded["passes"]["y_deltas_mm"] == y_del
    assert len(loaded["passes"]["data"]) == len(passes)
    for original, restored in zip(passes, loaded["passes"]["data"]):
        assert np.array_equal(original, restored)


def test_crc_mismatch_detected(tmp_path=None):
    import tempfile
    out_dir = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    header = out_dir / "sample.json"
    rip_payload.save_rip(header, {"dpi": 630}, [0.0], [0.0], _sample_passes(1))

    # Corrupt the sidecar; load must refuse it.
    bin_path = header.with_suffix(".bin")
    data = bytearray(bin_path.read_bytes())
    data[0] ^= 0xFF
    bin_path.write_bytes(bytes(data))

    try:
        rip_payload.load_rip(header)
    except ValueError:
        return
    raise AssertionError("expected a CRC mismatch error")


if __name__ == "__main__":
    test_round_trip()
    print("PASS test_round_trip")
    test_crc_mismatch_detected()
    print("PASS test_crc_mismatch_detected")
