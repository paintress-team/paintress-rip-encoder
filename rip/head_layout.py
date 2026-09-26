# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Head geometry: where the nozzles are, and which ink feeds each of them.

Two things are deliberately kept apart here:

* **The layout**: the head as built. A fixed set of *slots*, each slot a
  contiguous run of nozzles at a known place (which physical column, and how
  far up that column). This is a property of the hardware and never changes.
* **The plumbing** (``ink_map``): which ink is connected to each slot. This
  is machine setup: the same head can be re-plumbed, and the RIP must follow.

On the reference head (``c6n90``) every slot has identical geometry (six
columns of 90 nozzles, all covering the same rows), so the plumbing only ever
mattered for routing (which is the encoder's job) and for keeping dead-nozzle
records against physical slots. The RIP could ignore geometry entirely.

That stops being true on a head whose slots sit at *different heights*. On
``c4n180`` the right-hand column is split into three 60-nozzle blocks stacked
in Y, so the slot an ink is plumbed into decides which rows that ink can reach
in a given sweep. Geometry and plumbing therefore have to be resolved together
before any pass can be generated; that is what ``HeadGeometry`` is.

The consequence that shapes everything downstream: **the band step is the span
of the shortest plumbed slot, not the length of the column.** A head whose
smallest ink block covers 1/3" advances 1/3" per band, even though its columns
are a full inch long. ``c6n90`` is the special case where the two coincide.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# The generated head identities live beside the encoder (vendored from
# paintress-protocol); the encoder reaches rip/ the same way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "encoder"))
import head_profiles  # noqa: E402

# An unplumbed slot: it exists in the head but carries no ink.
EMPTY_SLOT = "-"

# The inks the colour pipeline actually renders. Anything else in a layout's
# channel order (LC/LM today) travels as a blank plane.
IMAGING_INKS: Tuple[str, ...] = ("C", "M", "Y", "K")


@dataclass(frozen=True)
class NozzleSlot:
    """One ink feed of the head: a run of nozzles inside one physical column.

    ``first_nozzle`` is the offset from the bottom of the column (nozzle index
    grows with +Y, matching the pass slicer). ``nozzle_count`` is how many
    nozzles the RIP fires; ``physical_nozzle_count`` is how many the slot
    actually has. The two differ when a slot is deliberately under-used: on
    ``c4n180`` the 180-nozzle column fires only its bottom 60, so that every
    slot advances at the same 1/3" band step (see the module docstring).
    """

    column: int
    first_nozzle: int
    nozzle_count: int
    physical_nozzle_count: int = 0

    def __post_init__(self):
        object.__setattr__(
            self, "physical_nozzle_count",
            self.physical_nozzle_count or self.nozzle_count,
        )

    @property
    def idle_nozzle_count(self) -> int:
        return self.physical_nozzle_count - self.nozzle_count


@dataclass(frozen=True)
class HeadLayout:
    """A head as built: its nozzle pitch, its columns, and its slots.

    ``slots`` are ordered the way you read the head facing it: left to right
    by column, and bottom to top within a column. ``default_inks`` is the
    reference plumbing in that same order (the profile's ``ink_map`` overrides
    it). ``channel_order`` is something else entirely: the order the channel
    planes take in the RIP payload, which is fixed by the head profile and does
    *not* move when the head is re-plumbed.
    """

    name: str
    nozzle_pitch_npi: int
    column_length: int
    channel_order: Tuple[str, ...]
    slots: Tuple[NozzleSlot, ...]
    default_inks: Tuple[str, ...]

    def __post_init__(self):
        if len(self.default_inks) != len(self.slots):
            raise ValueError(
                f"head {self.name}: default_inks has {len(self.default_inks)} "
                f"entries for {len(self.slots)} slots"
            )
        smallest = min(s.nozzle_count for s in self.slots)
        occupied: Dict[int, List[NozzleSlot]] = {}
        for i, slot in enumerate(self.slots):
            if slot.nozzle_count <= 0:
                raise ValueError(f"head {self.name}: slot {i} has no nozzles")
            if slot.first_nozzle + slot.physical_nozzle_count > self.column_length:
                raise ValueError(
                    f"head {self.name}: slot {i} runs past the end of a "
                    f"{self.column_length}-nozzle column"
                )
            # Every slot must start and span a whole number of the smallest
            # slot's blocks; otherwise no single band step tiles the image
            # for all inks at once.
            if slot.first_nozzle % smallest or slot.nozzle_count % smallest:
                raise ValueError(
                    f"head {self.name}: slot {i} (first={slot.first_nozzle}, "
                    f"count={slot.nozzle_count}) is not a whole multiple of the "
                    f"smallest slot ({smallest} nozzles); the inks cannot tile"
                )
            for other in occupied.setdefault(slot.column, []):
                if (slot.first_nozzle < other.first_nozzle + other.physical_nozzle_count
                        and other.first_nozzle < slot.first_nozzle + slot.physical_nozzle_count):
                    raise ValueError(
                        f"head {self.name}: slot {i} overlaps another slot in "
                        f"column {slot.column}"
                    )
            occupied[slot.column].append(slot)

    @property
    def slot_count(self) -> int:
        return len(self.slots)

    @property
    def column_count(self) -> int:
        return len({s.column for s in self.slots})

    @property
    def smallest_slot_nozzles(self) -> int:
        return min(s.nozzle_count for s in self.slots)

    @property
    def default_dpi(self) -> int:
        """The DPI the CLI offers when none is given: the lowest supported
        multiple of the pitch at or above 630 (the historical default)."""
        return int(ceil(630 / self.nozzle_pitch_npi) * self.nozzle_pitch_npi)

    def supports_dpi(self, dpi: int) -> bool:
        return dpi > 0 and dpi % self.nozzle_pitch_npi == 0

    def validate_ink_map(self, ink_map: Sequence[str]) -> Tuple[str, ...]:
        """Check a plumbing against this layout and return it as a tuple."""
        ink_map = tuple(ink_map)
        if len(ink_map) != self.slot_count:
            raise ValueError(
                f"ink_map {ink_map} must assign all {self.slot_count} slots of "
                f"head {self.name}"
            )
        seen = set()
        for ink in ink_map:
            if ink == EMPTY_SLOT:
                continue
            if ink not in self.channel_order:
                raise ValueError(
                    f"ink_map names {ink!r}, which head {self.name} does not "
                    f"carry (channels: {', '.join(self.channel_order)})"
                )
            if ink in seen:
                raise ValueError(f"ink_map plumbs {ink!r} into more than one slot")
            seen.add(ink)
        return ink_map


# ---------------------------------------------------------------------------
# The shipped layouts
# ---------------------------------------------------------------------------

# The reference head: six columns of 90 nozzles, every slot covering the same
# rows. Slot order is left to right facing the head; the plumbing below is the
# same reference wiring the generated head profile carries as SLOT_INKS.
C6N90 = HeadLayout(
    name="c6n90",
    nozzle_pitch_npi=90,
    column_length=90,
    channel_order=("C", "M", "Y", "K", "LC", "LM"),
    slots=tuple(NozzleSlot(column=c, first_nozzle=0, nozzle_count=90)
                for c in range(6)),
    default_inks=("K", "Y", "LM", "LC", "C", "M"),
)

# Two columns of 180 nozzles at 180 npi (a one-inch column). The left column is
# a single slot; the right one is split into three 60-nozzle blocks stacked in
# Y. The left slot fires only its bottom 60 nozzles, so all four inks share the
# same 1/3" band step; see https://paintress.dev/concepts/swaths-and-passes/
C4N180 = HeadLayout(
    name="c4n180",
    nozzle_pitch_npi=180,
    column_length=180,
    channel_order=("C", "M", "Y", "K"),
    slots=(
        NozzleSlot(column=0, first_nozzle=0, nozzle_count=60,
                   physical_nozzle_count=180),
        NozzleSlot(column=1, first_nozzle=0, nozzle_count=60),
        NozzleSlot(column=1, first_nozzle=60, nozzle_count=60),
        NozzleSlot(column=1, first_nozzle=120, nozzle_count=60),
    ),
    default_inks=("K", "C", "M", "Y"),
)

HEAD_LAYOUTS: Dict[str, HeadLayout] = {h.name: h for h in (C6N90, C4N180)}

# Decided in paintress-protocol (the profile marked `default: true`), not
# here: a RIP and a daemon started without --head must agree on the head, or
# the first job they share is refused with head_mismatch.
DEFAULT_HEAD = head_profiles.DEFAULT_HEAD
assert DEFAULT_HEAD in HEAD_LAYOUTS, (
    f"default head {DEFAULT_HEAD!r} has a profile but no layout here")


def get_layout(name: str) -> HeadLayout:
    try:
        return HEAD_LAYOUTS[name]
    except KeyError:
        raise ValueError(
            f"unknown head layout {name!r} (known: "
            f"{', '.join(sorted(HEAD_LAYOUTS))})"
        ) from None


# ---------------------------------------------------------------------------
# Layout + plumbing + DPI, resolved
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeadGeometry:
    """A layout with its plumbing and DPI resolved into printable numbers.

    Everything the pass generator needs comes from here, so that the schedule
    (``pass_schedule``) and the Y positions can never be derived from two
    different pieces of arithmetic.
    """

    layout: HeadLayout
    ink_map: Tuple[str, ...]
    dpi: int

    @classmethod
    def build(cls, layout: HeadLayout, dpi: int,
              ink_map: Optional[Sequence[str]] = None) -> "HeadGeometry":
        if not layout.supports_dpi(dpi):
            raise ValueError(
                f"DPI ({dpi}) must be a multiple of head {layout.name}'s nozzle "
                f"pitch ({layout.nozzle_pitch_npi} npi)"
            )
        resolved = layout.validate_ink_map(
            layout.default_inks if ink_map is None else ink_map)
        geometry = cls(layout=layout, ink_map=resolved, dpi=dpi)
        if not geometry.imaging_channels:
            raise ValueError(
                f"head {layout.name}: the ink map {resolved} plumbs none of "
                f"{', '.join(IMAGING_INKS)}: nothing to print"
            )
        return geometry

    # -- plumbing ---------------------------------------------------------

    def slot_of_ink(self, ink: str) -> Optional[int]:
        """Index of the slot ``ink`` is plumbed into, or None if unplumbed."""
        for i, plumbed in enumerate(self.ink_map):
            if plumbed == ink:
                return i
        return None

    def slot_for(self, ink: str) -> Optional[NozzleSlot]:
        """The slot geometry ``ink`` fires from, or None if unplumbed."""
        index = self.slot_of_ink(ink)
        return None if index is None else self.layout.slots[index]

    @property
    def imaging_channels(self) -> Tuple[str, ...]:
        """Plumbed inks the colour pipeline renders, in payload plane order."""
        return tuple(ch for ch in self.layout.channel_order
                     if ch in IMAGING_INKS and self.slot_of_ink(ch) is not None)

    @property
    def blank_channels(self) -> Tuple[str, ...]:
        """Planes the payload carries with no data (light inks today)."""
        rendered = set(self.imaging_channels)
        return tuple(ch for ch in self.layout.channel_order if ch not in rendered)

    @property
    def active_slots(self) -> Tuple[NozzleSlot, ...]:
        return tuple(self.layout.slots[self.slot_of_ink(ch)]
                     for ch in self.imaging_channels)

    # -- timing / geometry -------------------------------------------------

    @property
    def line_spacing_mm(self) -> float:
        return 25.4 / self.dpi

    @property
    def nozzle_pitch_lines(self) -> int:
        """Raster lines between two adjacent nozzles of a slot.

        This is the old ``passes_per_band``: it is also how many one-line-apart
        sweeps are needed to fill the gaps a single sweep leaves behind.
        """
        return self.dpi // self.layout.nozzle_pitch_npi

    @property
    def interleave_passes(self) -> int:
        return self.nozzle_pitch_lines

    @property
    def band_step_nozzles(self) -> int:
        """Nozzles of Y advance per band: the shortest plumbed slot's span."""
        return min(s.nozzle_count for s in self.active_slots)

    @property
    def band_step_lines(self) -> int:
        """Raster lines of Y advance per band, the old ``lines_per_band``."""
        return self.band_step_nozzles * self.nozzle_pitch_lines

    @property
    def band_step_mm(self) -> float:
        return self.band_step_lines * self.line_spacing_mm

    @property
    def max_group_nozzles(self) -> int:
        """Nozzle dimension of the payload array (the largest plumbed slot)."""
        return max(s.nozzle_count for s in self.active_slots)

    @property
    def max_slot_offset_lines(self) -> int:
        """How far up the column the highest plumbed slot starts, in lines."""
        return (max(s.first_nozzle for s in self.active_slots)
                * self.nozzle_pitch_lines)

    def lead_in_bands(self, band_overlap: int = 0) -> int:
        """Bands the head must start *below* the image.

        A slot sitting ``first_nozzle`` up the column cannot reach the bottom
        rows until the head has travelled that far below them, so the schedule
        has to begin under the image. Zero whenever every plumbed slot starts at
        the bottom of its column (``c6n90``).
        """
        offset = self.max_slot_offset_lines
        if not offset:
            return 0
        step = self.band_step_lines - self.clamp_band_overlap(band_overlap)
        return -(-offset // step)  # ceil, so the lowest slot still reaches line 0

    def lead_in_mm(self, band_overlap: int = 0) -> float:
        step = self.band_step_lines - self.clamp_band_overlap(band_overlap)
        return self.lead_in_bands(band_overlap) * step * self.line_spacing_mm

    def first_line_of(self, ink: str, head_line: int, nozzle: int) -> int:
        """Raster line that ``nozzle`` of ``ink`` prints with the head at
        ``head_line`` (the line the bottom of the column is aligned to)."""
        slot = self.slot_for(ink)
        if slot is None:
            raise ValueError(f"{ink!r} is not plumbed into head {self.layout.name}")
        return head_line + (slot.first_nozzle + nozzle) * self.nozzle_pitch_lines

    # -- the schedule ------------------------------------------------------

    def clamp_band_overlap(self, band_overlap: int) -> int:
        return max(0, min(int(band_overlap),
                          self.band_step_lines - self.nozzle_pitch_lines))

    def pass_schedule(self, height_lines: int, band_overlap: int = 0) -> List[int]:
        """Head positions, in raster lines, for every pass of the print.

        One entry per pass, in print order: bands advance by
        ``band_step_lines - band_overlap``, and within a band the head steps one
        line at a time to fill the gaps between nozzles. Entries before the
        image (negative) are the lead-in the taller slots need to reach the
        bottom rows.
        """
        step = self.band_step_lines - self.clamp_band_overlap(band_overlap)
        lead_in = self.lead_in_bands(band_overlap)

        schedule: List[int] = []
        for band in range(-lead_in, max(1, -(-height_lines // step))):
            base = band * step
            for offset in range(self.interleave_passes):
                head_line = base + offset
                if head_line >= height_lines:
                    break
                schedule.append(head_line)
        return schedule

    # -- description -------------------------------------------------------

    def describe(self) -> str:
        parts = []
        for ink, slot in zip(self.imaging_channels, self.active_slots):
            idle = f"+{slot.idle_nozzle_count} idle" if slot.idle_nozzle_count else ""
            parts.append(
                f"{ink}=col{slot.column}[{slot.first_nozzle}:"
                f"{slot.first_nozzle + slot.nozzle_count}]{idle}")
        return ", ".join(parts)

    def to_metadata(self, band_overlap: int = 0) -> Dict:
        """The ``head_layout`` block of the RIP payload header."""
        return {
            "name": self.layout.name,
            "nozzle_pitch_npi": self.layout.nozzle_pitch_npi,
            "column_length": self.layout.column_length,
            "channel_order": list(self.layout.channel_order),
            "ink_map": list(self.ink_map),
            "band_step_nozzles": self.band_step_nozzles,
            "lead_in_bands": self.lead_in_bands(band_overlap),
            "lead_in_mm": self.lead_in_mm(band_overlap),
            "slots": [
                {
                    "column": s.column,
                    "first_nozzle": s.first_nozzle,
                    "nozzle_count": s.nozzle_count,
                    "physical_nozzle_count": s.physical_nozzle_count,
                    "ink": ink,
                }
                for s, ink in zip(self.layout.slots, self.ink_map)
            ],
        }
