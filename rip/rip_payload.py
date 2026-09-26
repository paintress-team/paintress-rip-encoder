# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""On-disk format for the RIP intermediate (rip.py output, encoder.py input).

Mirrors the print-job container: a human-readable JSON header plus a packed
binary sidecar. The per-pass nozzle bitmaps are 1-bit (0/1) data, so they are
bit-packed with numpy (np.packbits) and stored back to back, about 20-40x
smaller than the old uncompressed pickle, and safe/portable (no arbitrary-code
execution on load, inspectable header, not tied to a Python pickle version).

    <name>.json   header: metadata, per-pass Y positions, array shape + CRC
    <name>.bin    np.packbits of the (passes, channels, nozzles, width) uint8 array

The channel and nozzle counts come from the head: one plane per channel the
head declares, and as many nozzle rows as its largest plumbed slot. The header's
``metadata.head_layout`` says which slot feeds each channel and how many of
those rows it actually uses. On a head whose slots sit at different heights,
that is the only thing that says which rows an ink reached (format 0.2.0; a
payload without the block is a c6n90 one, six planes of 90).

load_rip() returns the same in-memory dict shape the old pickle did, so the
encoder and viewer only change how they read the file, not what they get.
"""

import json
import zlib
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

RIP_FORMAT = "paintress-rip"
RIP_FORMAT_VERSION = "0.2.0"


def save_rip(
    output_path,
    metadata: Dict,
    y_positions_mm: Sequence[float],
    y_deltas_mm: Sequence[float],
    passes: List[np.ndarray],
) -> Path:
    """Write a RIP payload: a JSON header at ``output_path`` plus a ``.bin``
    sidecar. ``passes`` is a list of (channels, nozzles, width) uint8 arrays
    holding 0/1, all the same shape. Returns the sidecar path.
    """
    output_path = Path(output_path)
    bin_path = output_path.with_suffix(".bin")

    stacked = np.ascontiguousarray(np.stack(passes), dtype=np.uint8)  # (P, C, N, W)
    if stacked.size and stacked.max() > 1:
        raise ValueError("RIP bitmaps must contain only 0/1 values")

    bin_bytes = np.packbits(stacked).tobytes()  # C-order, MSB-first
    bin_path.write_bytes(bin_bytes)

    header = {
        "format": RIP_FORMAT,
        "format_version": RIP_FORMAT_VERSION,
        "metadata": metadata,
        "passes": {
            "y_positions_mm": list(y_positions_mm),
            "y_deltas_mm": list(y_deltas_mm),
        },
        "array": {
            "shape": list(stacked.shape),
            "dtype": "uint8",
            "packing": "packbits",  # np.packbits, C-order, MSB-first
            "element_count": int(stacked.size),
        },
        "data": {
            "file": bin_path.name,
            "byte_count": len(bin_bytes),
            "crc32": f"0x{zlib.crc32(bin_bytes) & 0xFFFFFFFF:08X}",
        },
    }
    output_path.write_text(json.dumps(header, indent=2))
    return bin_path


def load_rip(path) -> Dict:
    """Read a RIP payload (header path) into the dict shape the encoder/viewer
    expect: ``{"metadata": ..., "passes": {"y_positions_mm", "y_deltas_mm",
    "data": [ (C, N, W) arrays ]}}``.
    """
    path = Path(path)
    header = json.loads(path.read_text())
    if header.get("format") != RIP_FORMAT:
        raise ValueError(f"Not a RIP payload (format={header.get('format')!r}): {path}")

    bin_path = path.with_name(header["data"]["file"])
    bin_bytes = bin_path.read_bytes()

    crc = zlib.crc32(bin_bytes) & 0xFFFFFFFF
    expected = int(header["data"]["crc32"], 16)
    if crc != expected:
        raise ValueError(
            f"RIP sidecar CRC mismatch: 0x{crc:08X} != {header['data']['crc32']}"
        )

    shape = tuple(header["array"]["shape"])
    count = int(header["array"]["element_count"])
    flat = np.unpackbits(np.frombuffer(bin_bytes, dtype=np.uint8))[:count]
    stacked = flat.reshape(shape).astype(np.uint8)

    return {
        "metadata": header["metadata"],
        "passes": {
            "y_positions_mm": header["passes"]["y_positions_mm"],
            "y_deltas_mm": header["passes"]["y_deltas_mm"],
            "data": [stacked[i] for i in range(stacked.shape[0])],
        },
    }
