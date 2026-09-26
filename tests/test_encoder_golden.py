# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Golden regression tests for the encoder.

Two pins on the packed binary output, so the bit-packing can be rewritten with
confidence that the bytes on the wire are unchanged:

* test_synthetic_golden: a small deterministic RIP payload built in-memory.
  Always runs (no external data), so it covers the packing in CI.
* test_lucy_golden: the real sample (rip/output/lucy.json). That directory is
  gitignored, so the test skips when the sample is absent.

Runs under pytest or as a plain script.
"""

import json
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "encoder"))
sys.path.insert(0, str(REPO / "rip"))

import encoder as enc  # noqa: E402
import rip_payload  # noqa: E402

LUCY_RIP = REPO / "rip" / "output" / "lucy.json"

# Golden values captured for lucy.json (default layout).
LUCY_GOLDEN = {
    "geometry_fingerprint": 0x35D25533,
    "bin_crc32": 0xF9173FCA,
    "line_count": 52437,
    "bytes_per_line": 147,
    "total_passes": 21,
}

# Golden CRC for the synthetic payload below (default layout).
SYNTHETIC_BIN_CRC32 = 0x24A97A9E


def _as_int(value):
    """The JSON header stores some integers as hex strings (e.g. '0x...')."""
    return int(value, 16) if isinstance(value, str) else int(value)


def _encode(rip_json: Path, tmp: str):
    out_json = Path(tmp) / "job.json"
    # Pin the layout explicitly: the goldens verify the bit-packing and must
    # not move when the calibrated default gaps do. Both CRCs were captured
    # with the original 0.9 / 7.0 layout.
    layout = enc.PrintheadLayout(column_gap_mm=0.9, group_gap_mm=7.0)
    enc.convert_rip_to_printer_format(str(rip_json), str(out_json), layout)
    header = json.loads(out_json.read_text())
    bin_bytes = out_json.with_suffix(".bin").read_bytes()
    return header, bin_bytes


def _make_synthetic(path: Path):
    """A small, deterministic RIP payload (2 passes, width 16)."""
    dpi, nozzle, ppb, width = 630, 90, 7, 16
    rng = np.random.default_rng(20260629)
    passes = [rng.integers(0, 2, size=(6, nozzle, width), dtype=np.uint8) for _ in range(2)]
    meta = {
        "dpi": dpi,
        "image_width_px": width,
        "image_height_px": nozzle * ppb,
        "print_width_mm": width / dpi * 25.4,
        "print_height_mm": nozzle * ppb / dpi * 25.4,
        "nozzle_count": nozzle,
        "passes_per_band": ppb,
    }
    rip_payload.save_rip(path, meta, [0.0, 0.04], [0.0, 0.04], passes)


def test_synthetic_golden():
    with tempfile.TemporaryDirectory() as tmp:
        rip_json = Path(tmp) / "synthetic.json"
        _make_synthetic(rip_json)
        header, bin_bytes = _encode(rip_json, tmp)

        assert _as_int(header["bytes_per_line"]) == 147
        assert _as_int(header["total_passes"]) == 2
        crc = zlib.crc32(bin_bytes) & 0xFFFFFFFF
        assert crc == SYNTHETIC_BIN_CRC32, f"synthetic CRC32 0x{crc:08X} != 0x{SYNTHETIC_BIN_CRC32:08X}"


def test_lucy_golden():
    if not LUCY_RIP.exists():
        print(f"SKIP test_lucy_golden: sample not present ({LUCY_RIP})")
        return

    with tempfile.TemporaryDirectory() as tmp:
        header, bin_bytes = _encode(LUCY_RIP, tmp)
        crc = zlib.crc32(bin_bytes) & 0xFFFFFFFF

        assert _as_int(header["geometry_fingerprint"]) == LUCY_GOLDEN["geometry_fingerprint"]
        assert _as_int(header["bytes_per_line"]) == LUCY_GOLDEN["bytes_per_line"]
        assert _as_int(header["total_passes"]) == LUCY_GOLDEN["total_passes"]
        assert _as_int(header["data"]["line_count"]) == LUCY_GOLDEN["line_count"]
        assert len(bin_bytes) == LUCY_GOLDEN["line_count"] * LUCY_GOLDEN["bytes_per_line"]
        assert crc == LUCY_GOLDEN["bin_crc32"], f"bin CRC32 0x{crc:08X} != golden 0x{LUCY_GOLDEN['bin_crc32']:08X}"


if __name__ == "__main__":
    test_synthetic_golden()
    print("PASS test_synthetic_golden")
    test_lucy_golden()
    print("PASS test_lucy_golden")
