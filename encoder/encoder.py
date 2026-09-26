# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Print data encoder for piezo inkjet printing.

Second stage of the Paintress pipeline: reads the RIP payload (a JSON
header plus a packed binary sidecar, see rip_payload) produced by the RIP
and converts it into the job format the daemon loads and streams to the
firmware.

Stages performed here:
    1. Route each payload colour plane to the physical head output
       (slot) its ink is plumbed into, per the ink map.
    2. Apply the physical pixel offsets between nozzle columns, so every
       ink registers on the same target point of the substrate as the
       printhead travels.
    3. Encode each printed column into the firmware bit layout.
    4. Pack the head's data buses into a contiguous 147-byte line
       (head_packer; what rides each bus is the only head-specific part).
    5. Write the versioned JSON header plus the packed binary sidecar
       (see paintress_job, the shared job-container definition).

Wire format: each printed column is expressed as 196 hardware clocks of
6 bits each (three data pins x two clock edges) and packed contiguously into
147 bytes (1176 bits) with no padding. A head with fewer buses than the frame
has pins leaves the spare ones at zero; see head_packer.
"""

from dataclasses import dataclass, replace
from typing import List, Dict, Optional
import numpy as np
import json
import math
import argparse
from pathlib import Path
import sys
import time

import paintress_job as job

# rip_payload (the RIP-payload container, the rip -> encoder interface)
# lives with the RIP stage; make it importable when encoder.py runs as a
# script (python encoder/encoder.py ...).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rip"))
import rip_payload

# The head geometry is generated, not hand-kept: head_profiles.py is emitted
# from paintress-protocol's profiles/*.yaml and vendored in. It carries EVERY
# head, not one: the encoder packs for whichever head the payload names, so
# there is no build target on the host either. What it shares with the firmware
# is the FRAME (line size, clocks, packing); which head is fitted is machine
# configuration the daemon checks.
from head_profiles import (
    EDGE_COUNT, CLOCK_COUNT, BYTES_PER_LINE, FRAME_FINGERPRINT, head,
)

# The wire layout per head: which nozzle rides each bus position, and the
# window-enable map. See head_packer for why that table is the whole of it.
import head_packer
import head_profiles

# The reference head's scalars, for callers that name no head (a payload
# predating the head-layout block is a c6n90 one).
_REFERENCE = head("c6n90")
NOZZLE_COUNT = _REFERENCE.nozzle_count
CHANNELS = _REFERENCE.channels
SLOT_INKS = _REFERENCE.slot_inks


# ---------------------------------------------------------------------------
# Protocol Constants
# ---------------------------------------------------------------------------

# The wire layout of each head (which nozzle rides each position of each data
# bus, and the window-enable map) lives in head_packer. It is the only
# head-specific part of the encoding; the frame itself (196 clocks of 6 bits,
# packed into 147 bytes) is the firmware's and does not move.
#
# NAMED FOR ITS HEAD, on purpose. An unqualified WINDOW_ENABLE_MAP here read
# like "the" map and was in fact the reference head's, which is the exact shape
# of the bug that put a c6n90 value on c4n180 and printed blank pages. Encoding
# takes the map from the packer chosen per job (packer_for), never from a
# module constant, so this one exists only for the c6n90 tests that assert the
# reference layout.
C6N90_WINDOW_ENABLE_MAP = head_packer.C6N90_WINDOW_MAP

# EDGE_COUNT (392) and CLOCK_COUNT (196) come from the head profile: one bus
# block spans EDGE_COUNT bits, clocked on both the rise and the fall edge, so the
# block is CLOCK_COUNT clocks (two bits each).

BITS_PER_CLOCK = head_packer.C6N90.bits_per_clock

# The head's outputs (nozzle columns) as physical *slots*, indexed the way you
# read the head facing it. Which ink feeds each slot is the *ink map* (setup
# configuration, default SLOT_INKS from the head profile); how each slot's
# nozzles reach the wire is the packer's table.
SLOT_COUNT = head_packer.C6N90.slot_count
GROUP_COUNT = len(head_packer.C6N90.buses)

# Ink-map entry for an unplumbed slot: it fires nothing and drops out of the
# padding calculation.
EMPTY_SLOT = "-"

# The packed line the algorithm produces (CLOCK_COUNT x BITS_PER_CLOCK bits =
# 1176 bits = 147 bytes) must equal the profile's bytes_per_line; the assert
# catches a profile/packer disagreement instead of silently shipping a bad job.
PACKED_LINE_BYTES = BYTES_PER_LINE
assert (CLOCK_COUNT * BITS_PER_CLOCK + 7) // 8 == BYTES_PER_LINE, (
    "packer geometry (clocks x bits) disagrees with head profile bytes_per_line"
)


# ---------------------------------------------------------------------------
# Printhead Geometry
# ---------------------------------------------------------------------------

@dataclass
class PrintheadLayout:
    """
    Physical layout of the printhead.

    Defines the distances between nozzle columns (converted to pixels at the
    target DPI) and the ink map: which ink is plumbed into each of the head's
    outputs. ``slot_columns`` and ``columns`` say which column each slot sits
    in and how far behind the leading column each column sits, both read from
    the RIP payload's head layout; the defaults describe c6n90, so a caller
    that names neither gets the reference head.
    """

    # Measured c6n90 geometry (col_align calibration): the colour columns sit
    # on the nozzle-pitch grid (1/90"), 3 pitches between the two columns of a
    # group and 25 pitches between groups: exactly 0.846667 / 7.055556 mm.
    # The defaults sit 5 um below the exact values because the offsets
    # quantise with ceil(): a gap exactly on a pixel boundary can overshoot
    # the intended offset by a whole pixel through float rounding.

    # Distance between adjacent columns (mm). None = the head's own default
    # (see head_packer.COLUMN_GEOMETRY), which is what the CLI passes unless
    # --column-gap says otherwise.
    column_gap_mm: Optional[float] = None

    # Distance between colour groups (mm). None as above; a head with a single
    # column gap (c4n180) does not use it at all.
    group_gap_mm: Optional[float] = None

    # Ink map: the ink feeding each physical slot, left to right as you face
    # the head (slot 0 is leftmost and trails the sweep; slot 5 is rightmost
    # and leads it). Entries name RIP payload channels; EMPTY_SLOT marks an
    # unplumbed slot. Each ink may appear at most once. Default: the head
    # profile's reference plumbing.
    ink_map: tuple = SLOT_INKS

    # The column each slot sits in, in slot order. On c6n90 every slot has its
    # own column; on c4n180 three of the four share the right-hand one.
    slot_columns: tuple = tuple(range(SLOT_COUNT))

    # How far behind the leading column each column sits, in whole named gaps.
    columns: head_packer.ColumnGeometry = head_packer.COLUMN_GEOMETRY["c6n90"]

    def __post_init__(self):
        if len(self.ink_map) != len(self.slot_columns):
            raise ValueError(
                f"ink map {self.ink_map} must assign all "
                f"{len(self.slot_columns)} slots "
                f"(use {EMPTY_SLOT!r} for an unplumbed slot)"
            )
        plumbed = [ink for ink in self.ink_map if ink != EMPTY_SLOT]
        duplicates = {ink for ink in plumbed if plumbed.count(ink) > 1}
        if duplicates:
            raise ValueError(
                f"ink map {self.ink_map} plumbs {', '.join(sorted(duplicates))} "
                f"into more than one slot; each ink may appear at most once"
            )

    def slot_offsets_px(self, dpi: int) -> List[int]:
        """Return the pixel offset of each physical slot, in slot order.

        The head sweeps left to right and data columns fire in order, so each
        offset is the firing delay that registers a slot's column with the
        leading one: the leading column needs none, the trailing one the
        largest. A slot's offset is its *column's*; slots sharing a column
        (the three colours of c4n180's right-hand column) share it, with no
        special case.
        """
        offsets = self.columns.offsets_px(
            {"group": self.group_gap_mm, "column": self.column_gap_mm}, dpi)
        return [offsets[column] for column in self.slot_columns]

    def calculate_channel_offsets_px(self, dpi: int) -> Dict[str, int]:
        """Return the pixel offset of each plumbed ink at the given DPI.

        With the default ink map the physical layout, left to right, is
        K -> Y -> LM -> LC -> C -> M, so M (leading) gets offset 0 and K
        (trailing) the largest.
        """
        offsets_px = self.slot_offsets_px(dpi)
        return {
            ink: offsets_px[slot]
            for slot, ink in enumerate(self.ink_map)
            if ink != EMPTY_SLOT
        }

    def calculate_total_padding_px(self, dpi: int) -> int:
        """Return the maximum plumbed-slot offset (total padding required)."""
        offsets = self.calculate_channel_offsets_px(dpi)
        return max(offsets.values(), default=0)


# The frame fingerprint is generated once, in paintress-protocol's codegen, and
# imported above: the same value the firmware bakes in as FRAME_HASH. The
# daemon refuses a job whose frame disagrees with the connected firmware, and
# separately refuses one packed for a head other than the machine's.


# ---------------------------------------------------------------------------
# Pass Processing
# ---------------------------------------------------------------------------

def resolve_ink_map(
    ink_map: tuple,
    channel_order: List[str],
) -> List[Optional[int]]:
    """Resolve the ink map into a payload plane index per physical slot.

    ``channel_order`` is the payload's plane order (its ``channel_order``
    metadata). Returns one entry per slot, left to right; ``None`` marks an
    unplumbed slot. An ink the payload does not carry is an error.
    """
    plane_index = {name: i for i, name in enumerate(channel_order)}

    plane_for_slot: List[Optional[int]] = []
    for slot, ink in enumerate(ink_map):
        if ink == EMPTY_SLOT:
            plane_for_slot.append(None)
        elif ink in plane_index:
            plane_for_slot.append(plane_index[ink])
        else:
            raise ValueError(
                f"ink map slot {slot} names ink {ink!r}, which the RIP payload "
                f"does not carry (payload channels: {', '.join(channel_order)})"
            )
    return plane_for_slot


def check_no_ink_dropped(
    passes: List[np.ndarray],
    channel_order: List[str],
    plane_for_slot: List[Optional[int]],
) -> None:
    """Fail loud if a payload plane with ink coverage has no slot to fire from.

    Silently dropping a plane would print the wrong image; a payload plane may
    only go unmapped when it is completely blank (as LC/LM are today).
    """
    mapped = {plane for plane in plane_for_slot if plane is not None}
    for plane, ink in enumerate(channel_order):
        if plane in mapped:
            continue
        drops = int(sum(int(pass_data[plane].sum()) for pass_data in passes))
        if drops:
            raise ValueError(
                f"payload ink {ink!r} carries {drops} drops but the ink map "
                f"gives it no slot; add it to --ink-map or re-RIP without it"
            )


def gather_slot_passes(
    passes: List[np.ndarray],
    plane_for_slot: List[Optional[int]],
) -> List[np.ndarray]:
    """Reorder each pass's payload planes into physical slot order.

    Each input pass is (channels, 90, width) in payload plane order; each
    output pass is (SLOT_COUNT, 90, width) where row s is the bitmap slot s
    fires. An unplumbed slot gets a zero plane (it prints nothing).
    """
    slot_passes = []
    for pass_data in passes:
        zero_plane = np.zeros_like(pass_data[0])
        slot_passes.append(np.stack([
            pass_data[plane] if plane is not None else zero_plane
            for plane in plane_for_slot
        ]))
    return slot_passes


def apply_slot_offsets(
    passes: List[np.ndarray],
    layout: PrintheadLayout,
    dpi: int,
) -> List[np.ndarray]:
    """Apply the physical slot offsets to every slot-ordered pass.

    Each pass is an array of shape (SLOT_COUNT, 90, width). The function
    pads the width dimension and shifts each slot's plane according to the
    slot's physical offset, so every ink registers on the same target
    point of the substrate as the printhead travels.
    """
    if not passes:
        return []

    slot_offsets = layout.slot_offsets_px(dpi)
    total_padding = layout.calculate_total_padding_px(dpi)

    passes_array = np.array(passes)

    padded = np.pad(
        passes_array,
        pad_width=((0, 0), (0, 0), (0, 0), (0, total_padding)),
        mode="constant",
        constant_values=0,
    )

    shifted = np.zeros_like(padded)

    for slot, offset in enumerate(slot_offsets):
        if 0 < offset <= total_padding:
            shifted[:, slot, :, offset:] = padded[:, slot, :, :-offset]
        elif offset == 0:
            shifted[:, slot] = padded[:, slot]
        # An offset beyond total_padding belongs to an unplumbed slot (its
        # plane is all zeros and was excluded from the padding); leave it zero.

    return list(shifted)


def encode_single_pass(slot_data: np.ndarray, packer=None) -> List[bytes]:
    """Encode a single pass ``(slot_count, nozzles, width)`` to packed lines.

    ``slot_data`` is in physical slot order (see gather_slot_passes): row s is
    the bitmap fired by slot s. The head's packer decides which bus position
    each slot's nozzles ride and appends the window-enable map; the frame
    itself is fixed (see head_packer).
    """
    return (packer or head_packer.C6N90).encode_pass(slot_data)


def encode_all_passes(
    passes: List[np.ndarray],
    show_progress: bool = True,
    packer=None,
) -> List[List[bytes]]:
    """Encode every pass to a list of packed column lines."""
    packer = packer or head_packer.C6N90
    encoded_passes = []
    total = len(passes)

    for index, pass_data in enumerate(passes):
        if show_progress and (index + 1) % 5 == 0:
            print(f"   Encoding pass {index + 1}/{total}...")

        encoded = encode_single_pass(pass_data, packer)
        encoded_passes.append(encoded)

    return encoded_passes


# ---------------------------------------------------------------------------
# Main Processing Pipeline
# ---------------------------------------------------------------------------

def load_rip_file(file_path: str) -> Dict:
    """Load a RIP payload (JSON header + packed .bin sidecar) from disk."""
    return rip_payload.load_rip(file_path)


def convert_rip_to_printer_format(
    input_path: str,
    output_path: str,
    layout: Optional[PrintheadLayout] = None,
    window_map: Optional[str] = None,
    bus_order: Optional[str] = None,
) -> job.Job:
    """Convert a RIP payload (JSON header + .bin sidecar) into a printer-ready job.

    Writes two sibling files via :mod:`paintress_job`: a versioned JSON header
    (geometry + per-pass index) and a packed binary sidecar holding the
    contiguous 147-byte column lines. ``output_path`` is the header path; the
    sidecar takes the same name with a ``.bin`` suffix.
    """
    if layout is None:
        layout = PrintheadLayout()

    print(f"Loading: {input_path}")
    start_time = time.perf_counter()
    rip_data = load_rip_file(input_path)
    elapsed = time.perf_counter() - start_time
    print(f"   Loaded ({elapsed:.3f}s)")

    metadata = rip_data["metadata"]
    # The head comes from the payload, not from assumption: a payload written
    # for another head has a different wire layout, and the encoder must refuse
    # rather than pack it as if it were this one.
    head_layout = metadata.get("head_layout")
    packer = head_packer.packer_for(head_layout, window_map, bus_order)
    # The layout was built before the payload was read (the CLI only knows the
    # ink map), so give it the head's real column geometry now.
    layout = replace(
        layout,
        slot_columns=head_packer.slot_columns_for(head_layout),
        columns=head_packer.column_geometry_for(head_layout),
    )
    passes_data = rip_data["passes"]["data"]
    y_positions_mm = rip_data["passes"]["y_positions_mm"]
    y_deltas_mm = rip_data["passes"]["y_deltas_mm"]
    dpi = metadata["dpi"]

    print(f"Metadata:")
    print(f"   Head: {packer.name} ({packer.describe()})")
    # A print that comes out wrong has to be traceable to the guess behind it.
    for note in head_packer.provisional_notes(packer):
        print(f"   PROVISIONAL: {note}")
    print(f"   DPI: {dpi}")
    print(f"   Passes: {len(passes_data)}")
    print(f"   Dimensions: {metadata['image_width_px']}x{metadata['image_height_px']} px")

    # Route payload planes to physical slots via the ink map. Older payloads
    # (and synthetic test ones) may lack channel_order; they were produced in
    # the head profile's plane order.
    channel_order = list(metadata.get("channel_order", CHANNELS))
    plane_count = passes_data[0].shape[0] if len(passes_data) else 0
    if plane_count != len(channel_order):
        raise ValueError(
            f"RIP payload carries {plane_count} planes but its channel order "
            f"names {len(channel_order)} ({', '.join(channel_order)})"
        )
    plane_for_slot = resolve_ink_map(layout.ink_map, channel_order)
    check_no_ink_dropped(passes_data, channel_order, plane_for_slot)
    print(f"   Ink map (left to right): {', '.join(layout.ink_map)}")

    # Reorder planes into slot order and apply physical slot offsets
    print(f"\nApplying slot offsets...")
    start_time = time.perf_counter()
    slot_passes = gather_slot_passes(passes_data, plane_for_slot)
    shifted_passes = apply_slot_offsets(slot_passes, layout, dpi)
    elapsed = time.perf_counter() - start_time

    total_padding = layout.calculate_total_padding_px(dpi)
    original_width = passes_data[0].shape[2] if passes_data else 0
    padded_width = original_width + total_padding

    print(f"   Padding: +{total_padding} px (width: {original_width} -> {padded_width})")
    print(f"   Offsets applied ({elapsed:.3f}s)")

    # Encode passes to packed lines
    print(f"\nEncoding passes (147 bytes/line)...")
    start_time = time.perf_counter()
    encoded_passes = encode_all_passes(shifted_passes, show_progress=True,
                                       packer=packer)
    elapsed = time.perf_counter() - start_time
    print(f"   {len(encoded_passes)} passes encoded ({elapsed:.3f}s)")

    # Compute total padded image size in mm
    px_to_mm = 25.4 / dpi
    padded_width_mm = padded_width * px_to_mm
    padded_height_mm = metadata["image_height_px"] * px_to_mm

    # The job carries two identities. The FRAME fingerprint is what the
    # firmware reports at IDENTIFY: the line size, clocks and packing it shifts
    # out, shared by every head, so it never depends on which head this is. The
    # HEAD fingerprint says which printhead the job was packed for, and is
    # checked host-side against the machine's configuration; the board cannot
    # know which head is bolted on.
    head_profile = head(packer.name)
    if metadata["nozzle_count"] != head_profile.nozzle_count:
        raise ValueError(
            f"RIP payload says {metadata['nozzle_count']} nozzles per channel "
            f"but head {packer.name!r} has {head_profile.nozzle_count}; re-RIP "
            f"for this head, or update its profile in paintress-protocol"
        )

    # Build the typed job header. Per-pass line_count is filled by save_job
    # from the data actually written, so the header can't disagree with the bin.
    pass_infos = [
        job.PassInfo(
            y_position_mm=y_positions_mm[i],
            y_delta_mm=(y_deltas_mm[i] if i < len(y_deltas_mm) else 0.0),
            line_count=0,
        )
        for i in range(len(encoded_passes))
    ]

    job_metadata = job.JobMetadata(
        dpi=dpi,
        image_width_px=metadata["image_width_px"],
        image_height_px=metadata["image_height_px"],
        print_width_mm=metadata["print_width_mm"],
        print_height_mm=metadata["print_height_mm"],
        padded_width_px=padded_width,
        padded_width_mm=round(padded_width_mm, 4),
        padded_height_mm=round(padded_height_mm, 4),
        swath_pass_width_mm=metadata["print_width_mm"],
        nozzle_count=metadata["nozzle_count"],
        bytes_per_line=packer.bytes_per_line,
        passes_per_band=metadata["passes_per_band"],
        channel_offsets_px=layout.calculate_channel_offsets_px(dpi),
        passes=pass_infos,
        geometry_fingerprint=FRAME_FINGERPRINT,
        head_name=packer.name,
        head_fingerprint=head_profile.head_fingerprint,
        packing="contiguous",
    )

    # Write header + binary sidecar
    print(f"\nSaving: {output_path} (+ .bin sidecar)")
    start_time = time.perf_counter()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    saved = job.save_job(output_path, job_metadata, encoded_passes)
    elapsed = time.perf_counter() - start_time

    bin_mb = saved.byte_count / (1024 * 1024)
    print(f"   Saved {saved.bin_path.name} ({bin_mb:.2f} MB) + header ({elapsed:.3f}s)")
    print(f"\nConversion complete! ({job_metadata.total_lines} total lines)")

    return saved


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def read_payload_head(payload_path: str) -> Optional[Dict]:
    """The head layout a RIP payload was written for, or None for a c6n90 one.

    Read before the layout is built, because how many slots a layout has (and
    so whether an ink map even fits it) is a property of the head.
    """
    try:
        metadata = json.loads(Path(payload_path).read_text()).get("metadata", {})
    except (OSError, ValueError):
        return None
    return metadata.get("head_layout")


def resolve_layout_ink_map(cli_ink_map: Optional[str], payload_path: str) -> tuple:
    """Resolve the ink map for this encode.

    Precedence: an explicit ``--ink-map`` override, else the plumbing the RIP
    stamped into the payload (itself from the calibration profile), else the
    head layout's own reference plumbing, else the reference head's.
    Deferring to the payload keeps one source of the plumbing, so the encoder
    routes each ink to the same slot the RIP's dead-nozzle compensation
    assumed, and falling back through the payload's *head layout* rather than
    straight to the reference head matters as soon as a payload is for another
    head, whose slots the reference plumbing does not even fit.
    """
    if cli_ink_map is not None:
        return tuple(ink.strip() for ink in cli_ink_map.split(","))
    try:
        metadata = json.loads(Path(payload_path).read_text()).get("metadata", {})
    except (OSError, ValueError):
        metadata = {}
    payload_ink_map = (metadata.get("ink_map")
                       or (metadata.get("head_layout") or {}).get("ink_map"))
    return tuple(payload_ink_map) if payload_ink_map else SLOT_INKS


def main():
    parser = argparse.ArgumentParser(
        description="Convert a RIP payload (JSON+bin) to a printer-ready job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python encoder.py print_job.json -o print_data.json
  python encoder.py print_job.json -o print_data.json --column-gap 1.0
  python encoder.py print_job.json -o print_data.json --group-gap 7.5
  python encoder.py print_job.json -o print_data.json --ink-map "K,Y,-,-,C,M"

Output format:
  A versioned JSON header (<output>.json) plus a packed binary sidecar
  (<output>.bin) holding contiguous 147-byte column lines.
        """
    )

    parser.add_argument("input", help="RIP payload JSON header (from rip.py)")
    parser.add_argument("-o", "--output", required=True, help="JSON output file")

    parser.add_argument(
        "--column-gap",
        type=float,
        default=None,
        help="Column spacing in mm. Default: the head's own, 0.842 on c6n90 "
             "(calibrated, 3 nozzle pitches less the ceil() guard), 6.0 on "
             "c4n180 (nominal, not yet measured with col_align)."
    )
    parser.add_argument(
        "--group-gap",
        type=float,
        default=None,
        help="Group spacing in mm. Default: the head's own, 7.051 on c6n90 "
             "(calibrated, 25 nozzle pitches less the ceil() guard). Unused on "
             "a head with a single column gap."
    )

    # Bench knobs for a head whose electricals are still being walked through.
    parser.add_argument(
        "--window-map",
        default=None,
        help="Override the 32-bit window-enable map for this encode. Only "
             "meaningful while a head's real map is unknown; the default is "
             "the head profile's."
    )
    parser.add_argument(
        "--bus-order",
        choices=["straight", "interleaved"],
        default=None,
        help="Override how a bus's 180 positions map to its column's nozzles: "
             "'straight' (position p is nozzle p) or 'interleaved' (two banks "
             "of 90, as c6n90 wires two columns into one bus). A bench escape "
             "hatch: c4n180 specifies the straight order, so this should not "
             "be needed."
    )
    parser.add_argument(
        "--ink-map",
        default=None,
        help="Which ink feeds each of the head's six outputs, as comma-"
             "separated RIP channel names left to right as you face the head "
             "(the rightmost slot leads the sweep and fires first); use "
             f"'{EMPTY_SLOT}' for an unplumbed slot. Each ink at most once. "
             "Overrides the plumbing carried in the RIP payload. Default: the "
             "payload's ink_map, or the head profile's reference plumbing "
             f"({','.join(SLOT_INKS)}) when the payload carries none."
    )

    args = parser.parse_args()

    ink_map = resolve_layout_ink_map(args.ink_map, args.input)
    head_layout = read_payload_head(args.input)

    layout = PrintheadLayout(
        column_gap_mm=args.column_gap,
        group_gap_mm=args.group_gap,
        ink_map=ink_map,
        slot_columns=head_packer.slot_columns_for(head_layout),
        columns=head_packer.column_geometry_for(head_layout),
    )

    convert_rip_to_printer_format(
        input_path=args.input,
        output_path=args.output,
        layout=layout,
        window_map=args.window_map,
        bus_order=args.bus_order,
    )


if __name__ == "__main__":
    main()
