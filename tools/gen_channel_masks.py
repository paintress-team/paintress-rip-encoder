#!/usr/bin/env python3
"""Generate the firmware's channel_masks[] table: for purging, or for finding
a head's 32-bit window-enable map with PURGE.

WHY PURGE IS THE RIGHT INSTRUMENT
    PURGE fires with no external trigger and needs no swath transfer: the line
    it drives is ``channel_masks[]``, compiled into the firmware. That sidesteps
    both things that make the window map awkward to probe over the wire: the
    hardware start trigger and a 225 KB upload. The cost is that changing the
    line means a rebuild and a flash, so the search has to be worth the flash.

ONE CHANNEL PER BIT (--mode sweep)
    PURGE picks its channel at runtime, so a flash can carry as many lines as
    CH_COUNT allows, and sweep mode grows CH_COUNT to carry one line per
    window bit, 32 of them plus two controls. 34 purges, one flash, no decode
    step and no assumption about how many bits are live.

    Prefer it on any head that is not known to be one-hot: the cheaper group
    test below turned out to be unsound on c4n180: see the warning at the end
    of that section, which is exactly what happened on 2026-07-27. Record which COLOURS each channel
    ejects, not just whether it ejected; on a head whose bits select nozzle
    groups rather than gate firing globally, that is the whole answer.

FIVE PROBES, ONE FLASH (--mode probe)
    The shipped firmware exposes seven purge channels, and PURGE picks one at
    runtime. So a flash can carry seven different lines, and the search becomes
    group testing rather than one-candidate-per-flash:

        probe k (k = 0..4) sets EVERY window bit whose index has bit k set

    If exactly one window bit makes the head fire (which is what the reference
    head does, one code enabled in one window), then probe k ejects ink if and
    only if bit k of that index is 1. Run the five purges, write down which ones
    ejected, and read the index off in binary. 32 possibilities, five purges,
    one flash.

        probes that ejected: 1, 2, 4   ->  index = 2^1 + 2^2 + 2^4 = 22

    Two more channels earn their place as controls:
      * channel 0 sets ALL 32 bits. If this does not eject, nothing will, and
        the answer is that something else is wrong (ink, drive, plumbing),
        not that every candidate failed.
      * channel 6 sets only bit 14, the reference head's value, so the shipped
        guess is tested directly on the same flash.

    The decode assumes ONE bit is responsible. If two are, the five results
    describe a set, not an index, and will decode to nonsense, which is why
    the last step is always to flash the single decoded bit (--bits N) and
    confirm it ejects on its own.

    That is not hypothetical. On c4n180 the group test decoded to bit 14; the
    confirmation purge of bit 14 alone ejected NOTHING, while probes carrying
    other bits ejected, and ejected different numbers of colours from each
    other, under identical fire bits. Several bits are live and they address
    different nozzles. Use --mode sweep on any head that is not known to be
    one-hot; probe mode is a shortcut for confirming a head you already
    understand.

VERIFIED AGAINST THE FIRMWARE
    ``--mode purge --head c6n90`` regenerates the table now in
    engine/channels.c, all seven rows byte for byte. The generator is checked
    against that on every run (--self-test), so a c4n180 table it emits rests on
    a model proven against hand-written ground truth rather than on inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO / "encoder"))
sys.path.insert(0, str(REPO / "rip"))

import head_layout                      # noqa: E402
import head_packer as hp                # noqa: E402

# The purge channel enum, in index order. Fixed by engine/channels.h and the
# wire protocol; this tool refills the rows, never the enum.
CHANNEL_ENUM = ["CH_ALL_CHANNELS", "CH_YELLOW", "CH_BLACK", "CH_LIGHT_CYAN",
                "CH_LIGHT_MAGENTA", "CH_MAGENTA", "CH_CYAN"]
MASK_WORDS = 49
PROBE_BITS = 5            # 2^5 = 32 window-bit positions
# The colour column's three stacked channels, and the unit the black column's
# plumbed third is measured in.
NOZZLE_BLOCK = 60

# Which ink each enum slot purges, per head. c6n90 is the shipped mapping;
# c4n180 has four inks, so the two light channels have nothing to drive.
PURGE_INKS: Dict[str, Dict[str, Optional[str]]] = {
    "c6n90": {"CH_ALL_CHANNELS": None, "CH_YELLOW": "Y", "CH_BLACK": "K",
              "CH_LIGHT_CYAN": "LC", "CH_LIGHT_MAGENTA": "LM",
              "CH_MAGENTA": "M", "CH_CYAN": "C"},
    "c4n180": {"CH_ALL_CHANNELS": None, "CH_YELLOW": "Y", "CH_BLACK": "K",
               "CH_LIGHT_CYAN": "", "CH_LIGHT_MAGENTA": "",
               "CH_MAGENTA": "M", "CH_CYAN": "C"},
}


# ---------------------------------------------------------------------------
# Lines -> the 49-word rows the firmware stores
# ---------------------------------------------------------------------------

def line_to_words(line: bytes) -> List[int]:
    """A packed 147-byte line as the 49 words of 24 usable bits line_pack makes.

    Mirrors memory/line_pack.c: three consecutive bytes into the low 24 bits,
    the top 8 left zero. The table in the firmware is exactly this.
    """
    if len(line) != 3 * MASK_WORDS:
        raise ValueError(f"expected {3 * MASK_WORDS} bytes, got {len(line)}")
    return [line[i] | (line[i + 1] << 8) | (line[i + 2] << 16)
            for i in range(0, len(line), 3)]


# A split row sets this fraction of each slot to code 01; the rest stays 00.
# A THIRD, not a half, so the two combinations are told apart by size alone:
# with halves you have to know which end of the column is nozzle 0 before you
# can say which combination fired, and getting that backwards inverts the
# answer silently.
SPLIT_NUMERATOR, SPLIT_DENOMINATOR = 1, 3


def build_line(head: str, window_map: str, inks: Optional[str],
               ink_map: Optional[Sequence[str]],
               split: bool = False) -> bytes:
    """One packed line: ``inks`` firing every nozzle, under ``window_map``.

    ``inks`` None fires every plumbed slot; "" fires none (a channel this head
    does not have); otherwise it names one ink.

    ``split`` sets a THIRD of each slot to code 01 and leaves the other two
    thirds at 00, which is the only line shape that answers the question the
    window map is actually about.

    The 32 bits do not "enable codes"; they say, for each of the four 2-bit
    combinations, in which window that combination fires. So the goal is a map
    where 01 fires and 00 does not, and no row made entirely of one combination
    can measure that:

      * all-01: 00 never appears, so a bit that fires 00 looks identical to one
        that does not.
      * all-00: nothing is set to 01 either, so "00 did not fire" is
        indistinguishable from "no window was opened at all".

    A split row carries both at once, and the LENGTH of the streak names the
    combination that fired: one third means 01, two thirds means 00, the whole
    column means both. That is a reading that needs no assumption about nozzle
    order or about which end of the column is which.
    """
    layout = head_layout.get_layout(head)
    geometry = head_layout.HeadGeometry(
        layout=layout,
        ink_map=tuple(ink_map) if ink_map else layout.default_inks,
        dpi=layout.nozzle_pitch_npi,
    )
    meta = geometry.to_metadata()
    packer = hp.packer_for(meta, window_map=window_map)

    slots = meta["slots"]
    nozzles = max(s["nozzle_count"] for s in slots)
    data = np.zeros((len(slots), nozzles, 1), dtype=np.uint8)
    for index, slot in enumerate(slots):
        if inks == "" or (inks is not None and slot["ink"] != inks):
            continue
        count = slot["nozzle_count"]
        fired = (count * SPLIT_NUMERATOR // SPLIT_DENOMINATOR if split
                 else count)
        data[index, :fired, 0] = 1
    if inks and not data.any():
        raise SystemExit(f"head {head} has no {inks} slot; it is plumbed "
                         f"{[s['ink'] for s in slots]}")
    return packer.encode_pass(data)[0]


# ---------------------------------------------------------------------------
# Raw streams: for probes that deliberately leave the head model behind
# ---------------------------------------------------------------------------

def pack_streams(streams: Dict[int, np.ndarray]) -> bytes:
    """Pack per-bus 392-bit streams into one line, bypassing the head model.

    encode_pass() can only express what a head *is*: code 01, the wired buses,
    the plumbed nozzles. The diagnostic rows have to be able to say things the
    model calls impossible (a different drop code, a bus nothing is supposed
    to be wired to, every bit set at once), because those are exactly the
    assumptions under suspicion when nothing ejects at all.
    """
    clock_bits = np.zeros((hp.EDGE_COUNT // 2, 6), dtype=np.uint8)
    for group, stream in streams.items():
        clock_bits[:, group] = stream[0::2]
        clock_bits[:, group + 3] = stream[1::2]
    return np.packbits(clock_bits.reshape(-1), bitorder="little").tobytes()


def make_stream(positions: Sequence[int], code: str,
                window_map: str) -> np.ndarray:
    """One bus: ``positions`` carrying two-bit ``code``, then the window map.

    ``code`` is written as the head sees it: the first block's bit then the
    fire block's bit, so "01" is what a print emits today.
    """
    stream = np.zeros(hp.EDGE_COUNT, dtype=np.uint8)
    high, low = int(code[0]), int(code[1])
    for position in positions:
        stream[position] = high
        stream[hp.BUS_POSITIONS + position] = low
    stream[2 * hp.BUS_POSITIONS:] = [int(c) for c in window_map]
    return stream


def head_bus_positions(head: str) -> Dict[int, List[int]]:
    """Which bus positions each of a head's buses actually wires."""
    layout = head_layout.get_layout(head)
    geometry = head_layout.HeadGeometry(layout=layout,
                                        ink_map=layout.default_inks,
                                        dpi=layout.nozzle_pitch_npi)
    packer = hp.packer_for(geometry.to_metadata())
    return {bus.frame_group: [p for p, src in enumerate(bus.positions)
                              if src is not None]
            for bus in packer.buses}


# ---------------------------------------------------------------------------
# The tables
# ---------------------------------------------------------------------------

# Which purge channel a candidate flash reserves for a known-good line.
REFERENCE_CHANNEL = 4


def parse_bit_set(spec: str) -> Tuple[str, List[int], str]:
    """One `--bits` candidate: bits, then optionally how many nozzles fire.

    Bits: "all", "3,7", or "~30,31" for the complement. The complement form
    exists for the strongest test a candidate has: if {30,31} really is the
    whole live set, then EVERY OTHER BIT TOGETHER must eject nothing. A
    candidate only ever confirmed by what fires is confirmed by the half of the
    evidence that cannot refute it.

    Nozzles, as a ":suffix": "full" (default), ":half", or ":none".

    ":none" sets every nozzle to code 00 and is the only way to ask whether a
    map fires the blanks. Every purge row up to now drove all-01, so none of
    them could distinguish a map that fires drops from one that fires
    everything; the difference only showed up as ink in the white parts of a
    print, a whole flash-and-print cycle later. If a ":none" row ejects, the
    map fires 00.
    """
    spec = spec.strip()
    body, _, mode = spec.partition(":")
    mode = mode or "full"
    if mode not in ("full", "split", "none"):
        raise SystemExit(f"'{spec}': mode must be full, split or none")
    everything = list(range(hp.WINDOW_BITS))
    if body == "all":
        return spec, everything, mode
    if body.startswith("~"):
        excluded = {int(b) for b in body[1:].split(",") if b.strip()}
        return spec, [i for i in everything if i not in excluded], mode
    return spec, [int(b) for b in body.split(",") if b.strip()], mode


def bits_to_map(indices: Sequence[int]) -> str:
    bits = ["0"] * hp.WINDOW_BITS
    for i in indices:
        if not 0 <= i < hp.WINDOW_BITS:
            raise SystemExit(f"window bit {i} is outside 0..{hp.WINDOW_BITS - 1}")
        bits[i] = "1"
    return "".join(bits)


def probe_table(head: str, ink_map: Optional[Sequence[str]],
                baseline: str) -> List[Tuple[str, str, str]]:
    """(enum name, window map, comment) for the group-testing flash."""
    rows = [(CHANNEL_ENUM[0], "1" * hp.WINDOW_BITS,
             "control: every window bit set; if THIS does not eject, the "
             "problem is not the window map")]
    for k in range(PROBE_BITS):
        members = [i for i in range(hp.WINDOW_BITS) if (i >> k) & 1]
        rows.append((
            CHANNEL_ENUM[1 + k], bits_to_map(members),
            f"probe {k}: window bits with 2^{k} in their index "
            f"({members[0]},{members[1]},...,{members[-1]}): ejects iff "
            f"bit {k} of the answer is 1"))
    rows.append((CHANNEL_ENUM[6], baseline,
                 "control: the reference head's value, tested directly"))
    return rows


def sweep_table(head: str, baseline: str
                ) -> Tuple[List[Tuple], List[str]]:
    """One channel per window bit: the search that survives a multi-bit map.

    Group testing asks five yes/no questions and decodes an index from them,
    which is only sound if exactly ONE bit fires the head. c4n180 broke that on
    2026-07-27: probes carrying different bit sets ejected different *numbers of
    colours* under identical fire bits, so several bits are live and they do not
    all do the same thing. Five answers cannot describe that; thirty-two can.

    Costs one flash and 32 purges instead of one flash and 5, and returns the
    whole truth: which bits fire anything, and which nozzles each one fires.

    Every row sets only the LOWER HALF of each slot's nozzles (see build_line's
    ``half``), so each channel answers two questions at once:

        half a column wet  -> that bit fires code 01, the drops
        a full column wet  -> that bit also fires code 00, the blanks
        nothing            -> that bit does nothing

    The third case is what a search wants; the second is what c4n180 actually
    did on 2026-07-27 under bits 8-15/24-31, printing ink where the image was
    blank. A row with every nozzle at 01 cannot tell those two apart.

    Record the COLOURS too, not just yes/no; that is the signal that says
    whether a bit is a global enable or selects one 60-nozzle block.

    Channel index is bit index + 1, leaving 0 as the all-bits control so the
    "all" every other layer knows still means all.

    CHANNELS 0..6 ARE THE PROBE FLASH, UNCHANGED: same indices, same enum
    names, same bytes. That flash ejected on channels 0, 1 and 4, so those three
    are the reference this sweep is read against, and reproducing them at their
    original indices removes every way the comparison could be unfair: not the
    line, not the channel number, not the enum. The sweep is a strict superset
    of the flash that worked, so the bits start at channel 7.

    This matters because the first attempt got it wrong. Its "positive control"
    had all 32 bits but only half the nozzles (two changes at once against the
    row it was standing in for), so when all 34 channels came back silent there
    was no way to tell a broken table from a dry head. A control that does not
    reproduce a known result is not a control.

    The per-bit rows fire only the LOWER HALF of each slot (see build_line's
    ``half``), so each answers two questions at once:

        nothing            the bit does nothing
        half a column      the bit fires code 01, the drops. What we want
        a full column      the bit fires code 00 too: ink on the blanks,
                           which is what c4n180 did under bits 8-15/24-31

    Record the COLOURS too: a bit that fires one 60-nozzle block is a block
    selector, a bit that fires all four is a global enable, and they need
    different maps.
    """
    # (enum, window map, comment, ink, split)
    rows: List[Tuple] = []
    names: List[str] = []

    # 0..6: verbatim from probe_table, so the enum the shipped header already
    # declares keeps its meaning and the diff against it is purely additive.
    probe_names = ["probe_ctl_all", "probe_q0", "probe_q1", "probe_q2",
                   "probe_q3", "probe_q4", "probe_ctl_b14"]
    for (enum_name, window_map, comment), name in zip(
            probe_table(head, None, baseline), probe_names):
        rows.append((enum_name, window_map,
                     f"PROBE FLASH, VERBATIM: {comment}", None, False))
        names.append(name)

    # 7..38: the sweep proper.
    for bit in range(hp.WINDOW_BITS):
        rows.append((f"CH_BIT{bit:02d}", bits_to_map([bit]),
                     f"window bit {bit} alone, a THIRD of each slot at 01: "
                     f"purge channel {bit + 7}",
                     None, True))
        names.append(f"bit{bit:02d}")

    rows.append(("CH_NO_BITS", "0" * hp.WINDOW_BITS,
                 "NEGATIVE CONTROL, no window bit + every nozzle. MUST NOT "
                 "eject. If it does, the map does not gate firing and the "
                 "whole sweep is void",
                 None, False))
    names.append("ctl_no_bits")
    return rows, names


def check_controls(head: str, ink_map: Optional[Sequence[str]],
                   rows: List[Tuple[str, str, bytes]], baseline: str) -> None:
    """Channels 0..6 of a sweep must BE the probe flash, byte for byte.

    A reference that is merely "similar" to a known-good line is not a
    reference: if it goes silent you cannot say whether the head stopped
    working or the line changed. Assert the equality here rather than trusting
    that two code paths built the same bytes.
    """
    probe_rows = probe_table(head, ink_map, baseline)
    for index, (enum_name, window_map, _) in enumerate(probe_rows):
        expected = build_line(head, window_map, None, ink_map)
        actual = rows[index][2]
        if actual != expected or rows[index][0] != enum_name:
            raise SystemExit(
                f"sweep channel {index} is not byte-identical to the probe "
                f"flash's channel {index}. Channels 0..6 exist to reproduce a "
                f"result that is known, so emitting them changed would make "
                f"every comparison against that result meaningless.")


def sweep_manifest(head: str, triples: Sequence[Tuple], names: Sequence[str],
                   roles: Optional[Sequence[str]] = None,
                   expected: Optional[Sequence[Optional[bool]]] = None,
                   extra: Optional[Sequence[Dict]] = None) -> str:
    """What each purge channel is, for the runner to read back.

    The runner needs to know which channel is which bit and which are controls.
    Hard-coding that in two places is how a table grows a row and a tool keeps
    reporting the old layout, so the generator states it and the runner reads
    it.
    """
    # What the probe flash actually did, as reported from the bench.
    #
    # Only channel 4 is trustworthy. Everything else was observed at 10 pulses
    # during a session that ended with the DAC latched off by a safety fault,
    # and a latched DAC ejects nothing while the host still ACKs, so a
    # "silent" from that session cannot be told apart from a channel that never
    # ran. Channel 4 re-ran at 100 pulses after the latch was cleared and threw
    # THREE colours, not the two recorded at 10, which is the measure of how
    # much the pulse count was hiding.
    #
    # Channels 2, 3 and 5 were never called either way; guessing them would be
    # inventing a reference.
    OBSERVED = {
        0: "ejected, but at 10 pulses; re-run at 100",
        1: "1 colour at 10 pulses; likely an undercount, re-run at 100",
        4: "ejected 3 colours at 100 pulses (trustworthy)",
        6: "silent at 10 pulses, possibly with the DAC already latched: "
           "NOT a reliable negative, re-run at 100",
    }

    channels = []
    for index, (triple, name) in enumerate(zip(triples, names)):
        enum_name, window_map, comment = triple[0], triple[1], triple[2]
        bit = None
        if enum_name.startswith("CH_BIT"):
            bit = int(enum_name[len("CH_BIT"):])
        channels.append({
            "channel": index,
            "name": name,
            "bit": bit,
            "window_map": window_map,
            "split": bool(triple[4]) if len(triple) > 4 else False,
            "role": (roles[index] if roles else
                     "bit" if bit is not None else
                     "negative_control" if enum_name == "CH_NO_BITS"
                     else "reference"),
            # What the table PREDICTS for this channel, where it predicts
            # anything: True must eject, False must not, None is the open
            # question. A search whose rows have no prediction attached cannot
            # be surprised, and being surprised is the only way it learns.
            "must_eject": expected[index] if expected else None,
            "previously": OBSERVED.get(index) if bit is None and not roles
                          else None,
            "comment": comment,
        })
        if extra:
            channels[-1].update(extra[index])
    return json.dumps({"head": head, "window_bits": hp.WINDOW_BITS,
                       "channels": channels}, indent=2)


def emit_header(enum_names: Sequence[str]) -> str:
    """channels.h for a table whose length is not the shipped seven.

    The enum is normally hand-written and this tool only refills the rows. A
    sweep needs 34 of them, and CH_COUNT bounds-checks the wire's channel byte
    in two places, so the header has to grow with the table or every channel
    past the seventh is NACKed.
    """
    out = [
        "// SPDX-FileCopyrightText: 2026 paintress-team",
        "// SPDX-License-Identifier: GPL-3.0-or-later",
        "",
        "#pragma once",
        "",
        "// channels.h",
        "//",
        "// Ink channel identifiers and per-channel nozzle bit masks.",
        "//",
        "// GENERATED by tools/gen_channel_masks.py alongside channels.c.",
        "// This is a BENCH header: the enum is a window-bit sweep, not the",
        "// shipped ink channels. Restore the hand-written channels.h (git",
        "// checkout) together with a --mode purge channels.c.",
        "",
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "",
        f"#define CHANNEL_MASK_LEN {MASK_WORDS}",
        "",
        "typedef enum {",
    ]
    out += [f"    {name} = {i}," for i, name in enumerate(enum_names)]
    out += [
        "    CH_COUNT,  // number of channels; keep last (used to "
        "bounds-check input)",
        "} channel_mask_id_t;",
        "",
        "// Human-readable name of each channel, indexed by channel_mask_id_t.",
        "extern const char *const channel_mask_names[CH_COUNT];",
        "",
        "// Packed nozzle bit mask for each channel, indexed by "
        "channel_mask_id_t.",
        "extern const uint32_t channel_masks[CH_COUNT][CHANNEL_MASK_LEN];",
    ]
    return "\n".join(out) + "\n"


def purge_table(head: str, ink_map: Optional[Sequence[str]],
                window_map: str) -> List[Tuple[str, str, str]]:
    """The real per-ink purge table for a head."""
    mapping = PURGE_INKS[head]
    rows = []
    for name in CHANNEL_ENUM:
        ink = mapping[name]
        if ink is None:
            comment = "every plumbed nozzle"
        elif ink == "":
            comment = f"unused: {head} has no such ink"
        else:
            comment = f"{ink} only"
        rows.append((name, window_map, comment, ink))
    return rows


# ---------------------------------------------------------------------------
# The 32 bits, decoded: 8 windows x 4 codes
# ---------------------------------------------------------------------------

# A dump taken off a REAL PRINT of this head gives one value known to work:
#
#     0000 0000 0000 0000 1100 1010 1100 0001
#     w0   w1   w2   w3   w4   w5   w6   w7
#
# Read as eight nibbles (one per firing window) with four bits inside each,
# one per 2-bit nozzle code in the order 11, 10, 01, 00, that value says:
#
#     code 11    w4, w5, w6    3 pulses
#     code 10    w4, w6        2 pulses
#     code 01    w5            1 pulse
#     code 00    w7            1 pulse, alone in a window of its own
#
# which is a greyscale drop ladder: 11 the large drop, 10 the medium, 01 the
# small, and 00 a lone pulse nothing else shares: the non-ejecting tickle that
# keeps an idle meniscus moving. Monotonic in pulse count, which is the shape a
# drop ladder has to have and which no other grouping of these 32 bits produces.
#
# The cross-check the model did not get to choose: the reference head sets bit
# 14 and nothing else. 14 = window 3, slot 2 = code 01. One code in one window,
# which is exactly how that head prints. A model fitted to a dump from one head
# lands on the shipped value of the other.
#
# It also retrodicts the failure. {30,31}, the value this profile carried on
# 2026-07-27, is window 7 slots 2 AND 3: codes 01 and 00 firing in the same
# window. A print under it put ink where the image was blank, which is what the
# bench reported. The map was not "half right"; it explicitly enabled the blanks.
WINDOW_COUNT = 8
CODES_PER_WINDOW = 4
# Slot order inside a window's nibble, most significant bit of the nibble first.
WINDOW_SLOT_CODES = ("11", "10", "01", "00")
# The dump. Not a guess and not a measurement of ours: a value the head printed
# with, which makes it the one honest known-good a candidate can be read against.
DUMPED_MAP = "00000000000000001100101011000001"


def window_bit(window: int, code: str) -> int:
    """Index of the bit that fires ``code`` in ``window``, per the nibble model."""
    if not 0 <= window < WINDOW_COUNT:
        raise SystemExit(f"window {window} is outside 0..{WINDOW_COUNT - 1}")
    return window * CODES_PER_WINDOW + WINDOW_SLOT_CODES.index(code)


def describe_map(window_map: str) -> str:
    """A 32-bit map read back as (window, code) pairs."""
    firing: Dict[str, List[int]] = {code: [] for code in WINDOW_SLOT_CODES}
    for index, bit in enumerate(window_map):
        if bit == "1":
            firing[WINDOW_SLOT_CODES[index % CODES_PER_WINDOW]].append(
                index // CODES_PER_WINDOW)
    return "; ".join(
        f"{code} -> w{','.join(map(str, windows))}"
        for code, windows in firing.items() if windows) or "nothing fires"


def windows_table(head: str, ink_map: Optional[Sequence[str]]
                  ) -> Tuple[List[Tuple[str, str, bytes]], List[str],
                             List[Dict]]:
    """Which of the eight windows can OUR board open? One row per window.

    The dump settles the SLOT axis and settles it well: four codes, a monotonic
    drop ladder, and the reference head's shipped bit landing on code 01 without
    being asked to. What it cannot settle is the WINDOW axis, because a window is
    a firing trigger and the dump comes from a controller that is not ours.

    That is the whole difficulty. The 32 bits say *where* a code fires (after
    the latch, after a counter, and so on), and the original controller drives
    inputs this board does not have. Its print uses windows 4, 5, 6 and 7 and
    never touches 0..3. Ours cannot assume any of those four are reachable.

    The evidence that they may not be is already in hand: the REFERENCE HEAD
    PRINTS ON THIS BOARD WITH BIT 14, which is window 3, a window the dumped
    controller never uses. The most economical reading is that window 3 is the
    one a bare latch opens, which is exactly the "right after the latch" firing
    the bench asked for, and that windows 4..7 need a counter nobody wired.

    So this sweeps the axis that is open rather than betting on it:

        channel 0        every window bit set, all nozzles at 01
        channels 1..8    code 01's bit in window 0..7, all nozzles at 01
        channels 9..16   code 00's bit in the same window, all nozzles at 00
        channel 17       no bits at all

    Channel 0 is the control, and it is deliberately NOT the dumped map: the
    dumped map only lights windows 4..7, so if those are the unreachable ones it
    would be silent too and a silent flash would prove nothing. Every bit set
    can fire through whichever window this board does open, whatever it is. If
    channel 0 ejects nothing, the map is not what is wrong.

    The 01/00 pair per window is what makes a result mean something. A window
    that opens shows up on ONE of its two rows (the 01 row if the slot order
    is as decoded, the 00 row if it is inverted), so the pair separates "this
    window is shut" from "this window is open and I have the slots backwards",
    which a single row per window cannot do.

    Every row drives the head's own wired positions and differs from every other
    in nothing but (code, window map). None goes through the head model, which
    can only express code 01.
    """
    wired = head_bus_positions(head)

    def line(code: str, bits: Sequence[int]) -> bytes:
        window_map = bits_to_map(bits)
        return pack_streams({group: make_stream(positions, code, window_map)
                             for group, positions in wired.items()})

    reference_window = 3        # where the reference head fires on this board
    dumped_windows = sorted({i // CODES_PER_WINDOW
                             for i, b in enumerate(DUMPED_MAP) if b == "1"})

    # (name, code, bits, must_eject, role, comment)
    plan = [
        ("ctl_all_bits", "01", list(range(hp.WINDOW_BITS)), True, "control",
         "CONTROL: every window bit set, every nozzle at 01. Whatever window "
         "this board opens, something fires through it. MUST EJECT; if it "
         "does not, the window map is not what is wrong and nothing else on "
         "this flash was tested. Read it FIRST"),
    ]
    windows: List[Optional[int]] = [None]
    for window in range(WINDOW_COUNT):
        note = (": the window the REFERENCE HEAD fires in on this board, so "
                "the likeliest to be reachable" if window == reference_window
                else ": the dumped print fires in this window"
                if window in dumped_windows else
                ": the dumped print never uses this window, which says "
                "nothing about whether this board can open it")
        plan.append(
            (f"w{window}_at01", "01", [window_bit(window, "01")], None, "window",
             f"window {window}, code 01's bit ({window_bit(window, '01')}), "
             f"every nozzle at 01{note}"))
        windows.append(window)
    # The disqualifier, and the reason this pairing is what it is.
    #
    # The obvious second row is code 00's OWN bit driven at code 00, and it is
    # the wrong one: it asks whether the slot order is as decoded, which window
    # 7 answered on 2026-07-28 (both of its rows fired, each with its bit and
    # its code matched, which no inverted ordering produces). What a print turns
    # on is narrower and is not tested by any uniform row that came before:
    # setting code 01's bit ALONE, does the head fire nozzles that are at 00?
    #
    # It should not (code 00 then has no window anywhere in the map), but
    # "should not" is the assumption that put ink on the white parts of a page
    # twice. A map is only usable when the SAME bit that fires the drops has
    # been seen to leave the blanks alone.
    for window in range(WINDOW_COUNT):
        plan.append(
            (f"w{window}_blanks", "00", [window_bit(window, "01")], False,
             "window",
             f"window {window}, code 01's bit ({window_bit(window, '01')}) "
             f"AGAIN, but every nozzle at 00. MUST NOT EJECT: with only this "
             f"bit set code 00 has no window at all. Pairs with channel "
             f"{1 + window}; that one says window {window} can print, this "
             f"one says it can leave the white white"))
        windows.append(window)
    plan.append(
        ("ctl_no_bits", "01", [], False, "negative_control",
         "NEGATIVE CONTROL: no window bit at all, every nozzle at 01. MUST NOT "
         "EJECT. If it does, the 32 bits do not gate firing and this whole "
         "flash is void"))
    windows.append(None)

    rows, names, detail = [], [], []
    for index, (name, code, bits, must_eject, role, comment) in enumerate(plan):
        enum_name = (CHANNEL_ENUM[index] if index < len(CHANNEL_ENUM)
                     else f"CH_{name.upper()}")
        rows.append((enum_name, f"code {code}: {comment}", line(code, bits)))
        names.append(name)
        detail.append({"window_map": bits_to_map(bits), "code": code,
                       "must_eject": must_eject, "role": role,
                       "window": windows[index]})
    return rows, names, detail


def drop_table(head: str, ink_map: Optional[Sequence[str]]
               ) -> Tuple[List[Tuple[str, str, bytes]], List[str], List[Dict]]:
    """Why does every bit set eject MORE than code 01's bit alone?

    Measured on a new head, 2026-07-28, same waveform and same pulse count:
    all 32 bits ejects strongly on all four inks; bit 30 alone ejects, but
    cyan and magenta come out visibly lighter. Every other single window bit
    ejects nothing, so window 7 is the only one open and BOTH rows fire through
    it. The difference has to come from the other 31 bits.

    Under the nibble model it should not exist. With every nozzle at code 01,
    only code 01's slot can match, so bits 28, 29 and 31 (window 7's other
    three slots) ought to be inert, and the seven other windows are measured
    shut. Something in that chain is wrong, and an unexplained result on this
    particular question is not one to print through: the last two window maps
    that looked right and were not both put ink where the image was white.

    This bisects it. Every row drives every nozzle at code 01 and differs only
    in which window bits are set:

        ch0  all 32          the strong result, reproduced byte for byte
        ch1  bit 30 alone    the weak result, reproduced byte for byte
        ch2  bits 28-31      window 7's whole nibble
        ch3  bits 28,30      + the code 11 slot
        ch4  bits 29,30      + the code 10 slot
        ch5  bits 30,31      + the code 00 slot, the 2026-07-27 map
        ch6  bits 0..27      everything EXCEPT window 7

    ch6 is the one that can break the model outright: those bits belong to
    windows measured shut, so it must eject NOTHING. If it ejects, some bit
    outside code 01's slots reaches nozzles that carry code 01, and the whole
    decode has to be redone rather than patched.

    ch0 and ch1 are not fresh rows, they are the previous flash's ch0 and ch8
    rebuilt, and asserted identical below. A bisection whose endpoints are
    merely similar to the results it is bisecting measures nothing.

    Whatever wins here still has to be driven at code 00 before it can go in a
    map: more ink is worth nothing if it also reaches the blanks, which is
    exactly what ch5's {30,31} did in July.
    """
    wired = head_bus_positions(head)
    everything = list(range(hp.WINDOW_BITS))
    drop = window_bit(7, "01")                       # 30
    nibble = [7 * CODES_PER_WINDOW + s for s in range(CODES_PER_WINDOW)]
    outside = [b for b in everything if b not in nibble]

    def line(bits: Sequence[int]) -> bytes:
        window_map = bits_to_map(bits)
        return pack_streams({group: make_stream(positions, "01", window_map)
                             for group, positions in wired.items()})

    plan = [
        ("ctl_all_bits", everything, True,
         "REFERENCE, STRONG: every window bit, every nozzle at 01. Byte for "
         "byte the last flash's ch0, which ejected all four inks"),
        ("b30_alone", [drop], True,
         f"REFERENCE, WEAK: bit {drop} alone. Byte for byte the last flash's "
         f"ch8, which ejected with cyan and magenta visibly lighter"),
        ("w7_nibble", nibble, None,
         "window 7's whole nibble (28,29,30,31). If this is as strong as ch0, "
         "the extra ink comes from inside window 7 and the other seven windows "
         "are irrelevant"),
        ("b28_30", [28, drop], None,
         "bits 28,30: code 11's slot added to code 01's"),
        ("b29_30", [29, drop], None,
         "bits 29,30: code 10's slot added"),
        ("b30_31", [drop, 31], None,
         "bits 30,31: code 00's slot added. This is the map that printed ink "
         "on the white in July, so if it is the strong one, strength is not "
         "what a map should be chosen for"),
        ("outside_w7", outside, False,
         "bits 0..27: EVERYTHING EXCEPT WINDOW 7. Those windows are measured "
         "shut, so this MUST EJECT NOTHING. If it ejects, a bit outside code "
         "01's slots reaches code 01 nozzles and the decode is wrong, not "
         "incomplete"),
    ]

    rows, names, detail = [], [], []
    for enum_name, (name, bits, must_eject, comment) in zip(CHANNEL_ENUM, plan):
        rows.append((enum_name, f"code 01: {comment}", line(bits)))
        names.append(name)
        detail.append({"window_map": bits_to_map(bits), "code": "01",
                       "must_eject": must_eject, "role": "candidate",
                       "window": None})

    # The two endpoints must BE the rows they stand for, not resemble them.
    previous, _, _ = windows_table(head, ink_map)
    for here, there, what in ((0, 0, "ch0"), (1, 8, "ch8")):
        if rows[here][2] != previous[there][2]:
            raise SystemExit(
                f"drop table's ch{here} is not byte-identical to the window "
                f"sweep's {what}. Those two rows exist to reproduce measured "
                f"results; emitting them changed would make the bisection "
                f"between them meaningless.")
    return rows, names, detail


def blocks_table(head: str, window_map: str
                 ) -> Tuple[List[Tuple[str, str, bytes]], List[str],
                            List[Dict]]:
    """Which physical ink is on each (data bus, 60-nozzle block)?

    Every mapping this encoder applies above the wire (which frame group is
    the black column, which ink sits in which third of the colour column, which
    end of a column nozzle 0 is) has been ASSUMED from looking at the head,
    never measured. A print then came out with three of its four combs in black
    ink at wrong heights and the fourth in yellow, which no single one of those
    assumptions explains on its own.

    So this stops inferring. Each row drives exactly ONE 60-position block on
    ONE bus, at the code and window that are now known to fire. Purge it and
    write down the INK that comes out. Six rows give the whole physical map:

        bus 0, positions   0- 59      bus 1, positions   0- 59
        bus 0, positions  60-119      bus 1, positions  60-119
        bus 0, positions 120-179      bus 1, positions 120-179

    From that table, three separate questions answer themselves at once:

      * whichever bus ejects ONE ink from all three blocks is the black column,
        and that fixes C4N180_BUS_WIRING;
      * the other bus ejects three different inks, and their order down the
        blocks IS slot_inks, including whether it runs C,M,Y or Y,M,C, which
        is the "does nozzle 0 sit at the bottom" assumption nobody has tested;
      * a block that ejects nothing is a block this head does not plumb.

    None of those is in the fingerprint, so whatever this measures is a config
    change: no firmware rebuild, no re-RIP, just an ink map and a wiring tuple.

    Nothing here goes through the head model. The model IS the thing under
    test: asking it to describe the buses would only echo the assumption.
    """
    everything = list(range(hp.BUS_POSITIONS))

    def line(streams: Dict[int, Sequence[int]]) -> bytes:
        return pack_streams({group: make_stream(positions, "01", window_map)
                             for group, positions in streams.items()})

    plan = [("ctl_both_buses", {0: everything, 1: everything}, True,
             "CONTROL: every position of both buses. Every ink this head has "
             "must appear. If one is missing here it is missing everywhere, "
             "and no row below can be read as 'that block is unplumbed'")]
    for group in (0, 1):
        for block in range(hp.BUS_POSITIONS // NOZZLE_BLOCK):
            first = block * NOZZLE_BLOCK
            positions = list(range(first, first + NOZZLE_BLOCK))
            plan.append((
                f"g{group}_b{block}", {group: positions}, None,
                f"bus (frame group) {group}, positions {first}-"
                f"{first + NOZZLE_BLOCK - 1} ONLY. Write down WHICH INK comes "
                f"out: that is the whole measurement"))

    rows, names, detail = [], [], []
    for enum_name, (name, streams, must_eject, comment) in zip(CHANNEL_ENUM,
                                                               plan):
        rows.append((enum_name, f"code 01: {comment}", line(streams)))
        names.append(name)
        detail.append({"window_map": window_map, "code": "01",
                       "must_eject": must_eject, "role": "mapping",
                       "window": None})
    return rows, names, detail


def confirm_table(head: str, ink_map: Optional[Sequence[str]]
                  ) -> Tuple[List[Tuple[str, str, bytes]], List[str],
                             List[Dict]]:
    """Does {29,30} leave the blanks dry? And what does bit 29 do alone?

    The bisection at 10 pulses (new head, 2026-07-28) came out clean. Nothing
    outside window 7 ejects at all, and inside it the strength of a row follows
    one rule with no exception across six rows: code 01's slot ejects fully when
    an ADJACENT slot is also enabled, and weakly when it sits alone.

        {30}          slot 2 alone        CMY weak
        {28,30}       slots 0 and 2       CMY weak
        {29,30}       slots 1 and 2       full
        {30,31}       slots 2 and 3       full, but black weak
        {28,29,30,31} all four            full
        all 32 bits                       full

    That makes {29,30} the one to want: full strength on all four inks and the
    only such set that excludes bit 31, code 00's own slot. Bit 29 is code 10's
    slot, and we never emit code 10, so it cannot select a nozzle of ours; it
    appears to shape the pulse rather than choose who fires.

    "Appears to" is why this flash exists. The property a print depends on has
    never been measured for {29,30}, and the two maps that shipped before it
    were both believed on exactly this kind of reasoning.

        ch0  {29,30} at code 00     MUST NOT EJECT: the decision
        ch1  {30,31} at code 00     MUST EJECT: bit 31 IS code 00's slot
        ch2  {30}    at code 00     must not eject, already measured here
        ch3  {29,30} at code 01     full, reproduces the bisection's ch4
        ch4  {30}    at code 01     weak, reproduces the bisection's ch1
        ch5  {29}    at code 01     bit 29 alone: does it fire our code?
        ch6  {29}    at code 00     bit 29 alone: does it fire the blanks?

    CH1 IS WHAT MAKES CH0 MEAN ANYTHING. A row of nozzles at code 00 that
    ejects nothing is also what a dead head, a de-primed head and a mis-flashed
    table all look like. Bit 31 is code 00's own slot in the one window this
    board opens, so ch1 is the same question asked where the answer must be
    yes. If ch1 is silent too, ch0 measured nothing and the flash is void.

    Bit 29 alone (ch5, ch6) is the diligence the earlier maps skipped: do not
    add a bit to a shipped map without having seen what it does by itself.
    """
    wired = head_bus_positions(head)

    def line(code: str, bits: Sequence[int]) -> bytes:
        window_map = bits_to_map(bits)
        return pack_streams({group: make_stream(positions, code, window_map)
                             for group, positions in wired.items()})

    drop = window_bit(7, "01")      # 30
    shape = window_bit(7, "10")     # 29: the neighbour that restores strength
    blank = window_bit(7, "00")     # 31: code 00's own slot

    plan = [
        ("cand_blanks", "00", [shape, drop], False,
         f"THE DECISION: {{{shape},{drop}}} with every nozzle at code 00. MUST "
         f"NOT EJECT. Read it against ch1"),
        ("leak_control", "00", [drop, blank], True,
         f"POSITIVE CONTROL for ch0: {{{drop},{blank}}} at code 00. Bit {blank} "
         f"IS code 00's slot in window 7, so this MUST EJECT. If it does not, "
         f"a silent ch0 proves nothing: it would look the same on a dry head"),
        ("b30_blanks", "00", [drop], False,
         f"bit {drop} at code 00, the map in the profile today. Measured silent "
         f"on this head already; here to show the run reproduces it"),
        ("cand_drops", "01", [shape, drop], True,
         f"{{{shape},{drop}}} at code 01: FULL strength on all four inks in "
         f"the bisection. Byte for byte that flash's ch4"),
        ("b30_drops", "01", [drop], True,
         f"bit {drop} at code 01: ejects, cyan and magenta weak. Byte for "
         f"byte the bisection's ch1. The pair 3/4 is the strength difference "
         f"this map change is for"),
        ("b29_drops", "01", [shape], None,
         f"bit {shape} ALONE at code 01. Code 10's slot, and we emit code 01: "
         f"if this ejects, the slots are not an exact match on the code and "
         f"adding {shape} to a map is not the harmless shaping it looks like"),
        ("b29_blanks", "00", [shape], None,
         f"bit {shape} ALONE at code 00. The other half of characterising a bit "
         f"before shipping it"),
    ]

    rows, names, detail = [], [], []
    for enum_name, (name, code, bits, must_eject, comment) in zip(CHANNEL_ENUM,
                                                                 plan):
        rows.append((enum_name, f"code {code}: {comment}", line(code, bits)))
        names.append(name)
        detail.append({"window_map": bits_to_map(bits), "code": code,
                       "must_eject": must_eject, "role": "candidate",
                       "window": 7})

    # ch3 and ch4 stand in for measured rows; they must BE them.
    previous, _, _ = drop_table(head, ink_map)
    for here, there, what in ((3, 4, "ch4"), (4, 1, "ch1")):
        if rows[here][2] != previous[there][2]:
            raise SystemExit(
                f"confirm table's ch{here} is not byte-identical to the drop "
                f"bisection's {what}, which is the result it stands for.")
    return rows, names, detail


def codes_table(head: str, window_map: str, ink_map: Optional[Sequence[str]]
                ) -> Tuple[List[Tuple[str, str, bytes]], List[str]]:
    """Which 2-bit combination is the drop, and which window bits fire it?

    The encoder emits 01 for a drop and 00 for a blank, and every search so far
    took that for granted. The measurements say it cannot be right:

      * every bit except 30 and 31, all set at once, with every nozzle at 01:
        NOTHING. So no other bit fires the combination we emit.
      * bit 31 alone, with only a third of each block at 01: the whole band.
        So bit 31 fires the 00 nozzles too, and bit 30 does the same over the
        rest of the column.

    Together those leave no bit that fires our 01 without also firing 00, which
    is not a map that exists. The remaining assumption is the combination
    itself, so this varies THAT instead: the same split line written as 10 and
    as 11, against the bits that are known to do something and against the
    rest. Whichever (combination, bits) pair lights only the coded nozzles (
    a gapped streak rather than a solid one) is what the encoder should emit.
    """
    everything = list(range(hp.WINDOW_BITS))
    known = [30, 31]
    rest = [i for i in everything if i not in known]

    # A third of each 60-nozzle block, so a solid streak reads as "00 fired
    # too" and a gapped one as "only the coded nozzles fired".
    coded = [p for p in range(hp.BUS_POSITIONS) if p % 60 < 20]

    def line(code, bits):
        stream = make_stream(coded, code, bits_to_map(bits))
        return pack_streams({0: stream, 1: stream})

    rows = [
        (CHANNEL_ENUM[0], "combination 10, ALL 32 window bits, a third of each "
                          "block coded", line("10", everything)),
        (CHANNEL_ENUM[1], "combination 11, ALL 32 window bits, a third of each "
                          "block coded", line("11", everything)),
        (CHANNEL_ENUM[2], "combination 10, bits 30+31 only",
         line("10", known)),
        (CHANNEL_ENUM[3], "combination 11, bits 30+31 only",
         line("11", known)),
        (CHANNEL_ENUM[4], "REFERENCE, head model, every nozzle at 01 under the "
                          "current map; ejects. Read it first",
         build_line(head, window_map, None, ink_map)),
        (CHANNEL_ENUM[5], "combination 10, every bit EXCEPT 30 and 31",
         line("10", rest)),
        (CHANNEL_ENUM[6], "combination 11, every bit EXCEPT 30 and 31",
         line("11", rest)),
    ]
    names = ["c10_all", "c11_all", "c10_3031", "c11_3031", "reference",
             "c10_rest", "c11_rest"]
    return rows, names


def buswire_table(head: str, window_map: str,
                  ink_map: Optional[Sequence[str]]
                  ) -> Tuple[List[Tuple[str, str, bytes]], List[str]]:
    """Which frame group is the silent column actually wired to?

    c4n180 rides a three-pin frame with two buses, so one pin is shifted out as
    zeros and assumed to be connected to nothing. If the column that never
    ejects is in fact on THAT pin, it would behave exactly as observed: its data
    goes to a pin nobody wired, and the pin it is wired to carries zeros
    forever. No window map can fix that, which is why the column stayed silent
    through a search that explained every other result.

    The test copies the colour bus's stream (known to eject, since it is what
    channel 0 drives) onto each group in turn. Whichever copy makes the silent
    column fire is the group it is on. Sending a stream known to work removes
    the other explanation: if a copy is silent, it is the wiring and not the
    data.
    """
    colours = make_stream(range(hp.BUS_POSITIONS), "01", window_map)
    black = make_stream(range(60), "01", window_map)
    quiet = make_stream([], "01", "0" * hp.WINDOW_BITS)

    def line(streams):
        return pack_streams(streams)

    rows = [
        (CHANNEL_ENUM[0],
         "colour stream on group 1 only: the wiring as configured. "
         "Every colour, no black",
         line({1: colours, 0: quiet, 2: quiet})),
        (CHANNEL_ENUM[1],
         "colour stream COPIED ONTO GROUP 2, the pin assumed unconnected. "
         "If black fires here, that is where the black column is",
         line({1: colours, 2: colours, 0: quiet})),
        (CHANNEL_ENUM[2],
         "colour stream copied onto group 0, where black is configured. "
         "If black fires here the wiring is right and the fault is elsewhere",
         line({1: colours, 0: colours, 2: quiet})),
        (CHANNEL_ENUM[3],
         "colour stream on all three groups at once; if nothing fires "
         "beyond channel 0's colours, no group reaches the black column",
         line({0: colours, 1: colours, 2: colours})),
        # Built through the head model rather than raw streams, deliberately:
        # every other row here bypasses it, so if this and channel 0 eject
        # identically the raw-stream path is validated against the model on the
        # same flash. Note the map is the head's own measured one: passing
        # the REFERENCE head's (bit 14) would make this row silent, because bit
        # 14 fires code 00 and every nozzle here is 01.
        (CHANNEL_ENUM[4],
         "REFERENCE, built by the head model under the measured map: ejects "
         "3 colours. Read it first: if it is silent the head is the variable "
         "and nothing here was tested",
         build_line(head, window_map, None, ink_map)),
        (CHANNEL_ENUM[5],
         "black's own 60 positions on group 2",
         line({2: black, 0: quiet, 1: quiet})),
        (CHANNEL_ENUM[6],
         "black's own 60 positions on group 0: the current wiring, already "
         "known silent. The negative this flash is measured against",
         line({0: black, 1: quiet, 2: quiet})),
    ]
    names = ["colour_g1", "colour_g1g2", "colour_g1g0", "colour_all3",
             "reference_q3", "black_g2", "black_g0"]
    return rows, names


def diagnose_table(head: str, baseline: str) -> List[Tuple[str, str, bytes]]:
    """(enum name, comment, packed line) for the "nothing ejects" flash.

    A search over window maps only asks *which* map fires while taking three
    things for granted: that the drop code is 01, that the head's buses are on
    frame groups 0 and 1, and that the window bits gate firing at all. If the
    head ejects nothing under any map (including all 32 bits set), one of
    those is likelier wrong than every candidate being wrong. Each row here
    breaks exactly one of them, so a single flash says which.
    """
    wired = head_bus_positions(head)
    all_bits = "1" * hp.WINDOW_BITS
    every_position = list(range(hp.BUS_POSITIONS))

    def line(streams):
        return pack_streams(streams)

    def normal(code, window_map):
        return {g: make_stream(p, code, window_map) for g, p in wired.items()}

    rows = [
        (CHANNEL_ENUM[0],
         "code 01, every window bit, the head's own buses: the reference "
         "assumption with the window wide open",
         line(normal("01", all_bits))),
        (CHANNEL_ENUM[1],
         "DROP CODE 11 instead of 01. If only this ejects, the head fires on a "
         "code the encoder never emits, and the window map was never the issue",
         line(normal("11", all_bits))),
        (CHANNEL_ENUM[2],
         "DROP CODE 10: the other unexplored code",
         line(normal("10", all_bits))),
        (CHANNEL_ENUM[3],
         "NEGATIVE CONTROL: no window bit set at all. If this ejects, the "
         "window bits do not gate firing and the whole search was misdirected",
         line(normal("01", "0" * hp.WINDOW_BITS))),
        (CHANNEL_ENUM[4],
         "the shipped guess (bit 14) on its own, for comparison",
         line(normal("01", baseline))),
        (CHANNEL_ENUM[5],
         "WIRING DOUBT: all three frame groups driven, every position. If only "
         "this ejects, the head's buses are not on groups 0 and 1",
         line({g: make_stream(every_position, "01", all_bits)
               for g in range(3)})),
        (CHANNEL_ENUM[6],
         "EVERY BIT OF THE LINE SET: all three groups, both code blocks, all "
         "32 window bits. The last thing to try before concluding the head is "
         "not being driven at all",
         line({g: np.ones(hp.EDGE_COUNT, dtype=np.uint8) for g in range(3)})),
    ]
    return rows


# ---------------------------------------------------------------------------
# Emitting C
# ---------------------------------------------------------------------------

def format_row(name: str, words: List[int], comment: str) -> str:
    body = []
    for i in range(0, MASK_WORDS, 7):
        body.append("            "
                    + " ".join(f"0x{w:06X}," for w in words[i:i + 7]))
    return (f"    // {comment}\n"
            f"    [{name}] =\n        {{\n" + "\n".join(body) + "\n        },\n")


def emit(head: str, rows: List[Tuple[str, str, bytes]], names: Sequence[str],
         banner: str) -> str:
    """``rows`` are (enum name, comment, packed 147-byte line)."""
    out = [
        "// SPDX-FileCopyrightText: 2026 paintress-team",
        "// SPDX-License-Identifier: GPL-3.0-or-later",
        "//",
        "// channels.c",
        "//",
        "// Per-channel nozzle bit masks (declared in channels.h).",
        "//",
        f"// GENERATED by tools/gen_channel_masks.py for head {head}.",
        "// Do not hand-edit: regenerate instead, so the table and the",
        "// encoder's wire layout cannot drift apart.",
        "//",
    ]
    out += [f"// {line}" for line in banner.strip().splitlines()]
    out += [
        "",
        '#include "channels.h"',
        "#include <stdint.h>",
        "",
        "const char* const channel_mask_names[CH_COUNT] = {",
        "    " + ", ".join(f'"{n}"' for n in names) + ",",
        "};",
        "",
        "const uint32_t channel_masks[CH_COUNT][CHANNEL_MASK_LEN] = {",
        "",
    ]
    for name, comment, line in rows:
        out.append(format_row(name, line_to_words(line), comment))
    out.append("};")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Self-test against the firmware's hand-written table
# ---------------------------------------------------------------------------

# SHA-256 (first 128 bits) of each row of the ORIGINAL hand-written c6n90
# table, taken from engine/channels.c as first committed. Frozen here on
# purpose: this tool overwrites that file, so a self-test that read it back
# would start comparing the generator against its own last output and go quiet
# exactly when it mattered. These hashes are the ground truth the whole wire
# model rests on (two blocks of 180, fire bits at 180+p, window at 360..391)
# and the generator reproduces them byte for byte.
C6N90_SHIPPED_ROWS = {
    "CH_ALL_CHANNELS": "973ae973f817f5344d659b3f11466ce5",
    "CH_YELLOW": "6c262ab77ecb066ad03ce41b828feaf5",
    "CH_BLACK": "d4847be9ead11d80b73ea44e12d98896",
    "CH_LIGHT_CYAN": "27f47e1f5d423969d56dc5a120a80d6e",
    "CH_LIGHT_MAGENTA": "cfc33439ee5674191d17de8cc4c7fb13",
    "CH_MAGENTA": "5a0a94c04cd142edcb535db9608fd51d",
    "CH_CYAN": "0e6a8917ec5ecc8c671fa23bb5de784e",
}


def row_digest(words: List[int]) -> str:
    blob = b"".join(w.to_bytes(4, "little") for w in words)
    return hashlib.sha256(blob).hexdigest()[:32]


def self_test() -> bool:
    """Regenerate c6n90's original table and check it against frozen hashes."""
    import head_profiles
    baseline = head_profiles.head("c6n90").window_enable_map
    ok = True
    for name, window_map, _comment, ink in purge_table("c6n90", None, baseline):
        ours = row_digest(line_to_words(build_line("c6n90", window_map, ink,
                                                   None)))
        if ours != C6N90_SHIPPED_ROWS[name]:
            ok = False
            print(f"  MISMATCH {name}: {ours} != {C6N90_SHIPPED_ROWS[name]}")
    print(f"self-test: {'all 7 rows match the shipped c6n90 table' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the firmware's channel_masks[] table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # The flash that finds the window map: 5 group-testing probes + 2 controls.
  python tools/gen_channel_masks.py --mode probe --head c4n180 \\
      -o ../paintress-firmware/engine/channels.c

  # Confirm a decoded answer on its own before believing it.
  python tools/gen_channel_masks.py --mode probe --head c4n180 --bits 22

  # Back to a normal per-ink purge table once the map is known.
  python tools/gen_channel_masks.py --mode purge --head c4n180 \\
      --window-map 00000000000000000000010000000000
""")
    parser.add_argument("--mode",
                        choices=["windows", "drop", "confirm", "blocks",
                                 "sweep", "probe", "purge", "diagnose",
                                 "buswire", "codes"],
                        default="windows",
                        help="confirm the 8-windows x 4-codes model against a "
                             "dumped map, 7 rows, no header change (windows, "
                             "default); "
                             "one channel per window bit (sweep: the "
                             "search that survives a multi-bit map); the "
                             "5-probe group test (probe, only sound if exactly "
                             "one bit fires); the real per-ink purge table "
                             "(purge); the assumption-breaking table for a "
                             "head that ejects nothing at all (diagnose); or "
                             "the frame-group hunt for a column that never "
                             "ejects (buswire)")
    parser.add_argument("--manifest",
                        default=str(Path(__file__).with_name(
                            "sweep_manifest.json")),
                        help="sweep mode: where to describe the channel "
                             "layout for purge_sweep_runner.py "
                             "(default: alongside this tool)")
    parser.add_argument("--header",
                        help="sweep mode: also write channels.h here. REQUIRED "
                             "for a sweep: 34 rows need CH_COUNT to grow or "
                             "every channel past the seventh is NACKed "
                             "(typically paintress-firmware/engine/channels.h)")
    parser.add_argument("--head", default="c4n180", choices=sorted(PURGE_INKS))
    parser.add_argument("--ink-map",
                        help="comma-separated inks per slot; default is the "
                             "head's own plumbing")
    parser.add_argument("--window-map",
                        help="purge mode: the 32-bit map to bake in "
                             "(default: the head profile's)")
    parser.add_argument("--bits",
                        help="probe mode: skip the group test and test "
                             "candidate bit sets instead, one per channel. A "
                             "set is comma-separated (30,31), 'all', or "
                             "'~30,31' for every OTHER bit; separate sets with "
                             "';'. Channel 4 is always kept as a known-good "
                             "reference. E.g. --bits '30,31;30;31;~30,31'")
    parser.add_argument("-o", "--output",
                        help="write here instead of stdout (typically "
                             "paintress-firmware/engine/channels.c)")
    parser.add_argument("--no-self-test", action="store_true")
    args = parser.parse_args(argv)

    if not args.no_self_test and not self_test():
        print("\nrefusing to generate: the model disagrees with the firmware's "
              "own table, so anything emitted here would be a guess",
              file=sys.stderr)
        return 1

    import head_profiles
    # Two different maps, and conflating them silently rewrote a flash once.
    #
    # head_map is what THIS head's profile currently claims: a moving value,
    # since bringing a head up is precisely the business of changing it. It is
    # what a purge table should bake in.
    #
    # reference_map is the c6n90 value the probe flash's last row means by "the
    # reference head's value, tested directly". It must NOT follow the target
    # head's profile: when c4n180's profile moved off bit 14 on 2026-07-27, a
    # regenerated probe flash would have quietly stopped testing bit 14 while
    # still calling itself the same flash, and the row it is compared against
    # would have moved with it.
    head_map = head_profiles.head(args.head).window_enable_map
    reference_map = head_profiles.head("c6n90").window_enable_map
    ink_map = args.ink_map.split(",") if args.ink_map else None

    def with_lines(triples):
        """(name, window_map, comment[, ink[, split]]) -> (name, comment, line)."""
        return [(row[0], row[2],
                 build_line(args.head, row[1],
                            row[3] if len(row) > 3 else None, ink_map,
                            row[4] if len(row) > 4 else False))
                for row in triples]

    header = manifest = None
    if args.mode == "windows":
        rows, names, detail = windows_table(args.head, ink_map)
        manifest = sweep_manifest(
            args.head,
            [(row[0], d["window_map"], row[1]) for row, d in zip(rows, detail)],
            names,
            roles=[d["role"] for d in detail],
            expected=[d["must_eject"] for d in detail],
            extra=[{"code": d["code"], "window": d["window"]} for d in detail])
        header = emit_header([row[0] for row in rows])
        banner = (
            "WINDOW SWEEP. The 32 bits are 8 windows x 4 codes: a nibble per\n"
            "window, and inside it the codes 11, 10, 01, 00. The dump settles\n"
            "the CODE axis; it cannot settle which windows THIS board opens,\n"
            "because a window is a firing trigger and the dump is another\n"
            "controller's.\n"
            f"  dumped map {DUMPED_MAP}\n"
            f"  -> {describe_map(DUMPED_MAP)}, i.e. windows 4-7 only.\n"
            "The reference head prints on this board in WINDOW 3, which that\n"
            "controller never uses, so window 3 is the likeliest to be the\n"
            "one a bare latch opens. Channel 4 is that row.\n"
            "\n"
            "CHANNEL 0 IS THE CONTROL: every bit set. Read it FIRST; if it\n"
            "is silent the map is not what is wrong.\n"
            "Channels 1-8 are code 01 in window 0-7; channels 9-16 are code\n"
            "00 in the same windows, paired. Channel 17 must stay silent.")
    elif args.mode == "drop":
        rows, names, detail = drop_table(args.head, ink_map)
        manifest = sweep_manifest(
            args.head,
            [(row[0], d["window_map"], row[1]) for row, d in zip(rows, detail)],
            names,
            roles=[d["role"] for d in detail],
            expected=[d["must_eject"] for d in detail],
            extra=[{"code": d["code"], "window": d["window"]} for d in detail])
        banner = (
            "WHY DOES EVERY BIT EJECT MORE THAN CODE 01'S BIT ALONE?\n"
            "Same waveform, same pulse count, one open window (7), and all 32\n"
            "bits still ejects more than bit 30 alone. Under the nibble model\n"
            "that cannot happen, so this bisects the other 31 bits.\n"
            "Every row is every nozzle at code 01; only the window bits move.\n"
            "ch0 and ch1 ARE the last flash's ch0 and ch8, byte for byte.\n"
            "ch6 MUST BE SILENT; if it ejects, the decode is wrong.\n"
            "Whatever wins still has to be driven at code 00 before it is a map.")
    elif args.mode == "blocks":
        rows, names, detail = blocks_table(args.head, head_map)
        manifest = sweep_manifest(
            args.head,
            [(row[0], d["window_map"], row[1]) for row, d in zip(rows, detail)],
            names,
            roles=[d["role"] for d in detail],
            expected=[d["must_eject"] for d in detail],
            extra=[{"code": d["code"], "window": d["window"]} for d in detail])
        banner = (
            "WHICH INK IS ON EACH (BUS, 60-NOZZLE BLOCK)?\n"
            "One row per block, one block per row: purge it and write down\n"
            "the INK that comes out. That table fixes three things nobody has\n"
            "measured: which frame group is the black column, which ink sits\n"
            "in which third of the colour column, and which end of a column\n"
            "nozzle 0 is. All three are host config, none is fingerprinted.\n"
            f"Window map {head_map}, code 01: the pair measured to fire.\n"
            "ch0 is the control: every ink must appear there.")
    elif args.mode == "confirm":
        rows, names, detail = confirm_table(args.head, ink_map)
        manifest = sweep_manifest(
            args.head,
            [(row[0], d["window_map"], row[1]) for row, d in zip(rows, detail)],
            names,
            roles=[d["role"] for d in detail],
            expected=[d["must_eject"] for d in detail],
            extra=[{"code": d["code"], "window": d["window"]} for d in detail])
        banner = (
            "DOES {29,30} LEAVE THE BLANKS DRY?\n"
            "ch0 is the decision: {29,30} with every nozzle at code 00, which\n"
            "MUST NOT EJECT. ch1 is the positive control that makes ch0 mean\n"
            "something: bit 31 IS code 00's slot, so ch1 MUST eject. A silent\n"
            "ch0 with a silent ch1 measured nothing.\n"
            "ch3/ch4 reproduce the strength difference, byte for byte.\n"
            "ch5/ch6 characterise bit 29 on its own before it ships.")
    elif args.mode == "sweep":
        triples, names = sweep_table(args.head, reference_map)
        rows = with_lines(triples)
        header = emit_header([t[0] for t in triples])
        check_controls(args.head, ink_map, rows, reference_map)
        manifest = sweep_manifest(args.head, triples, names)
        banner = ("WINDOW-BIT SWEEP FLASH.\n"
                  "Channels 0..6 ARE THE PROBE FLASH, VERBATIM: same "
                  "indices, same bytes.\nChannel 4 is the one that ejected two "
                  "colours; 0 and 1 also ejected.\nIf those three are silent "
                  "here, the table is not what changed.\n"
                  "Channels 7..38 drive window bit N-7 ALONE, lower half of "
                  "each slot only,\nso a full-length streak means that bit "
                  "fires code 00 too (ink on blanks).\nChannel 39 is the "
                  "negative control: it must NOT eject.\n"
                  "Record WHICH COLOURS and HOW MUCH of the column, not just "
                  "whether it ejected.")
    elif args.mode == "codes":
        rows, names = codes_table(args.head, head_map, ink_map)
        manifest = sweep_manifest(
            args.head, [(r[0], head_map, r[1]) for r in rows], names)
        banner = ("DROP-COMBINATION HUNT. Every row codes only a THIRD of "
                  "each\n60-nozzle block, so a GAPPED streak means only the "
                  "coded nozzles fired\nand a SOLID one means 00 fired too.\n"
                  "CHANNEL 4 is the reference; read it first.")
    elif args.mode == "buswire":
        rows, names = buswire_table(args.head, head_map, ink_map)
        manifest = sweep_manifest(
            args.head, [(r[0], head_map, r[1]) for r in rows], names)
        banner = ("FRAME-GROUP HUNT for the column that never ejects.\n"
                  "The colour bus's stream (known to eject) is copied onto "
                  "each group in turn.\nWhichever copy makes the silent column "
                  "fire is the group it is wired to.\n"
                  "CHANNEL 4 is the reference; read it first.")
    elif args.mode == "diagnose":
        rows = diagnose_table(args.head, reference_map)
        names = ["open_window", "code_11", "code_10", "no_window",
                 "baseline_b14", "all_three_buses", "every_bit"]
        banner = ("DIAGNOSTIC FLASH, for a head that ejects nothing under any "
                  "window map.\nEach row breaks a DIFFERENT assumption the "
                  "search took for granted.")
    elif args.mode == "purge":
        window_map = args.window_map or head_map
        rows = with_lines(purge_table(args.head, ink_map, window_map))
        names = ["all_channels", "yellow", "black", "light_cyan",
                 "light_magenta", "magenta", "cyan"]
        banner = (f"Per-ink purge table, window map {window_map}.")
    elif args.bits:
        candidates = [parse_bit_set(spec) for spec in args.bits.split(";")]

        # Channel 4 stays the probe-3 row, verbatim. A confirmation flash whose
        # every channel is the candidate has no way to distinguish "the
        # candidate is wrong" from "the head stopped ejecting", which cost a
        # day on 2026-07-27, when a safety fault latched the DAC mid-search and
        # every channel of every flash went quiet with nothing to compare
        # against. One reference channel is cheap insurance.
        reference = probe_table(args.head, ink_map, reference_map)[4]
        slots = [i for i in range(len(CHANNEL_ENUM)) if i != REFERENCE_CHANNEL]
        triples, names = [], []
        lines = []
        for index, enum_name in enumerate(CHANNEL_ENUM):
            if index == REFERENCE_CHANNEL:
                triples.append((enum_name, reference[1],
                                "REFERENCE, probe q3 verbatim: known to "
                                "eject 3 colours. If this is silent the flash "
                                "proves nothing about any candidate"))
                names.append("reference_q3")
                continue
            spec, indices, mode = candidates[
                slots.index(index) % len(candidates)]
            shown = (indices if len(indices) < 9
                     else f"{len(indices)} of them")
            nozzles = {"full": "every nozzle at 01",
                       "split": "a THIRD at 01, two thirds at 00",
                       "none": "EVERY NOZZLE AT 00: must not eject"}[mode]
            triples.append((enum_name, bits_to_map(indices),
                            f"candidate '{spec}': window bit(s) {shown}, "
                            f"{nozzles}",
                            "" if mode == "none" else None,
                            mode == "split"))
            names.append(f"cand_{spec}".replace(",", "_").replace("~", "not")
                         .replace(":", "_"))
            lines.append(f"    channel {index}  {spec:<14} {nozzles}")
        rows = with_lines(triples)
        manifest = sweep_manifest(args.head, triples, names)
        banner = ("CANDIDATE FLASH. CHANNEL 4 is the probe q3 reference, "
                  "which ejects 3 colours;\nread it FIRST; if it is silent "
                  "the head is the variable and nothing else\nhere was "
                  "tested. The other channels carry:\n" + "\n".join(lines))
    else:
        triples = probe_table(args.head, ink_map, reference_map)
        rows = with_lines(triples)
        names = ["control_all_bits", "probe_q0", "probe_q1", "probe_q2",
                 "probe_q3", "probe_q4", "control_baseline"]
        # The runner drives whichever flash was emitted last, so probe mode
        # describes itself too. Without this it would keep reading a sweep
        # manifest and calling channel 4 "window bit -3".
        manifest = sweep_manifest(args.head, triples, names)
        banner = ("WINDOW-MAP SEARCH FLASH. Every row fires every plumbed "
                  "nozzle;\nthe rows differ only in their 32 window bits.\n"
                  "Channels 0, 1 and 4 ejected on 2026-07-27; channel 6 did "
                  "not. This table is\nbyte-identical to the one that produced "
                  "those results.")

    if header is not None and not args.header:
        print("--mode sweep needs --header: its table is longer than the "
              "shipped enum, and without a matching channels.h the firmware "
              "still has CH_COUNT=7, so channels 7..33 are NACKed and the "
              "sweep silently tests only its first seven bits.", file=sys.stderr)
        return 1

    code = emit(args.head, rows, names, banner)

    if args.output:
        Path(args.output).write_text(code, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(code)

    if header is not None:
        Path(args.header).write_text(header, encoding="utf-8")
        print(f"wrote {args.header}")

    if manifest is not None:
        Path(args.manifest).write_text(manifest, encoding="utf-8")
        print(f"wrote {args.manifest}  (purge_sweep_runner.py reads this)")

    if args.mode == "diagnose":
        print("\n".join([
            "",
            "Flash this, then purge each channel and note which eject:",
            "",
            "    0  window wide open, code 01   the search's assumption",
            "    1  code 11                     a code the encoder never emits",
            "    2  code 10                     the other one",
            "    3  no window bits at all       do they gate firing?",
            "    4  bit 14 only                 the shipped guess",
            "    5  all three data buses        is the wiring what we think?",
            "    6  every bit of the line set   last resort",
            "",
            "If 1 or 2 eject, the drop code is the answer, not the window map.",
            "If 3 ejects, the window bits do not gate firing.",
            "If 5 ejects but 0 does not, the buses are on other frame groups.",
            "If NONE eject (including 6), the head is not being driven:",
            "look at ink priming, the drive rail, and the DAC waveform before",
            "any more bit patterns.",
        ]), file=sys.stderr)
    if args.mode == "probe" and not args.bits:
        print("\n".join([
            "",
            "Flash this, then purge each channel and note which eject ink:",
            "",
            "    channel 0  control: all 32 bits    must eject, or stop here",
            "    channel 1  probe q0                adds 1 to the answer",
            "    channel 2  probe q1                adds 2",
            "    channel 3  probe q2                adds 4",
            "    channel 4  probe q3                adds 8",
            "    channel 5  probe q4                adds 16",
            "    channel 6  control: bit 14 only    the shipped guess",
            "",
            "Add up the probes that ejected: that sum is the window bit.",
            "None ejecting (with channel 0 working) means bit 0.",
            "Then confirm it alone:  --bits <answer>",
        ]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
