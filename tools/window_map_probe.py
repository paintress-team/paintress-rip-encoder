#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Walk candidate 32-bit window-enable maps on a head over USB.

The last 32 bits of every bus's 392-bit stream tell the head which drop codes
fire in which window after the latch. They are an electrical property that
cannot be derived, so this finds it by trying candidates on the real head.
The c4n180 value in its profile was measured this way on 2026-07-28.

WHY THIS DRIVES THE BOARD DIRECTLY, NOT THE DAEMON
    The only path that puts *host* bytes on the head is a swath: BEGIN_SWATH ->
    lines -> END_SWATH -> ARM. PURGE looks tempting and is not usable here:
    it drives ``channel_masks[]``, a table compiled into the firmware, so it
    would test the firmware's window bits rather than ours. The daemon adds a
    job file, a TCP hop and a head check on top of that same sequence, none of
    which a bench probe wants, so this imports the daemon's transport
    (SerialManager + SwathHandler) and speaks to the board itself.

THE ONE THING YOU MUST WIRE
    ARM does not fire. It arms and waits for a RISING EDGE on TRIGGER_IN_PIN
    (GPIO 45), giving up after 10 s with a TRIGGER_TIMEOUT event. On a printer
    the motion controller raises it mid-sweep; on a bench, touch the pin to 3V3
    through a pull-up-friendly wire, or drive it from any spare output. This
    tool tells you when it is waiting.

TWO WAYS TO READ THE ANSWER

  --mode sweep (default)
    Every printed column carries its own copy of the 32 bits, so one swath can
    carry every candidate: N columns of candidate 0, a blank gap, N columns of
    candidate 1, and so on. One trigger, one pass of the gantry, and the
    candidate that works is the block that ejected ink, read off the strip
    against the manifest this writes. 32 candidates in one print.

  --mode step
    One candidate per ARM, pausing between them. For a stationary head being
    watched or scoped, where all the columns would land on the same spot.

SAFETY
    The head ejects ink and the piezo stack is energized. Any exit path (
    finished, failed, Ctrl-C) goes through ABORT before closing the port. The
    frame fingerprint is checked at IDENTIFY, because a firmware built for a
    different frame would take these 147-byte lines and shift them out wrong.
    Use --dry-run to build and print the plan without opening the port.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO / "encoder"))
sys.path.insert(0, str(REPO / "rip"))

import head_layout                      # noqa: E402
import head_packer as hp                # noqa: E402
import head_profiles                    # noqa: E402

WINDOW_BITS = hp.WINDOW_BITS            # 32

# A blank separator between candidate blocks: no code enabled in any window, so
# nothing can fire whatever the head makes of it.
BLANK_MAP = "0" * WINDOW_BITS

# Refuse to build a probe longer than this. A slot holds far more, but a sweep
# this long is a sign of a mistyped block size, not an intention.
MAX_PROBE_LINES = 20000


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def _one_hot(index: int) -> str:
    return "0" * index + "1" + "0" * (WINDOW_BITS - index - 1)


def _reverse_bytes(bits: str) -> str:
    chunks = [bits[i:i + 8] for i in range(0, WINDOW_BITS, 8)]
    return "".join(reversed(chunks))


def _reverse_within_bytes(bits: str) -> str:
    chunks = [bits[i:i + 8] for i in range(0, WINDOW_BITS, 8)]
    return "".join(c[::-1] for c in chunks)


def candidate_maps(which: str, baseline: str,
                   explicit: Optional[Sequence[str]] = None) -> List[str]:
    """The candidate ladder, cheapest and likeliest first.

    2^32 is not searchable, and does not need to be: the reference head enables
    exactly one code in one window, so the map is one-hot. ``single`` walks that
    one bit through all 32 positions, which is the search that actually ends
    this question. ``order`` is the other way to be wrong (the right map read
    out backwards), and costs a handful of trials. ``pair`` is the fallback if
    the head turns out to want two codes enabled.
    """
    if explicit:
        return list(explicit)

    if which == "baseline":
        return [baseline]
    if which == "single":
        return [_one_hot(i) for i in range(WINDOW_BITS)]
    if which == "order":
        seen, out = set(), []
        for candidate in (baseline,
                          baseline[::-1],
                          _reverse_bytes(baseline),
                          _reverse_within_bytes(baseline),
                          _reverse_bytes(baseline[::-1]),
                          _reverse_within_bytes(baseline)[::-1]):
            if candidate not in seen:
                seen.add(candidate)
                out.append(candidate)
        return out
    if which == "pair":
        return ["".join("1" if k in (i, j) else "0" for k in range(WINDOW_BITS))
                for i in range(WINDOW_BITS) for j in range(i + 1, WINDOW_BITS)]
    raise ValueError(f"unknown candidate set {which!r}")


def describe_map(bits: str) -> str:
    """A map as its set bit positions: how you actually think about a one-hot."""
    set_bits = [i for i, c in enumerate(bits) if c == "1"]
    if not set_bits:
        return "(none set)"
    if len(set_bits) <= 4:
        return "bit " + ", ".join(str(b) for b in set_bits)
    return f"{len(set_bits)} bits set"


# ---------------------------------------------------------------------------
# Building the lines
# ---------------------------------------------------------------------------

def build_layout(head: str, ink_map: Optional[Sequence[str]], dpi: int) -> Dict:
    layout = head_layout.get_layout(head)
    geometry = head_layout.HeadGeometry(
        layout=layout,
        ink_map=tuple(ink_map) if ink_map else layout.default_inks,
        dpi=dpi,
    )
    return geometry.to_metadata()


def slot_pattern(layout_meta: Dict, channel: Optional[str], width: int,
                 fire: bool = True) -> np.ndarray:
    """The nozzle bitmap a probe block fires.

    Solid by default (every plumbed nozzle on every line), because the
    question being asked is "did anything come out at all", and a solid bar is
    the loudest possible yes. ``channel`` narrows it to one ink, which is how
    you tell the two buses apart: on c4n180 K is the only thing on bus 0.
    ``fire=False`` is the separator: not one nozzle bit set anywhere.
    """
    slots = layout_meta["slots"]
    nozzles = max(s["nozzle_count"] for s in slots)
    data = np.zeros((len(slots), nozzles, width), dtype=np.uint8)
    if not fire:
        return data
    for index, slot in enumerate(slots):
        if channel and slot["ink"] != channel:
            continue
        data[index, :slot["nozzle_count"], :] = 1
    if not data.any():
        raise ValueError(
            f"no slot carries ink {channel!r}; this head is plumbed "
            f"{[s['ink'] for s in slots]}")
    return data


def block_lines(layout_meta: Dict, window_map: str, channel: Optional[str],
                width: int, bus_order: Optional[str],
                fire: bool = True) -> List[bytes]:
    packer = hp.packer_for(layout_meta, window_map=window_map,
                           bus_order=bus_order)
    return packer.encode_pass(
        slot_pattern(layout_meta, channel, width, fire))


def build_sweep(layout_meta: Dict, maps: Sequence[str], channel: Optional[str],
                width: int, gap: int, bus_order: Optional[str],
                dpi: int, index_offset: int = 0) -> Tuple[List[bytes], List[Dict]]:
    """One swath carrying every candidate, with blank gaps between them.

    The separator is silent by DATA (every nozzle bit zero), not merely by a
    zero window map. Using a window value to guarantee silence would be
    circular: what the window bits do to the head is the very thing being
    probed, so a separator resting on them could bleed into its neighbours and
    smear the answer. Both are zeroed; only the data zeroing is load-bearing.
    """
    lines: List[bytes] = []
    manifest: List[Dict] = []
    blank = (block_lines(layout_meta, BLANK_MAP, channel, gap, bus_order,
                         fire=False)
             if gap else [])

    for index, bits in enumerate(maps):
        first = len(lines)
        lines.extend(block_lines(layout_meta, bits, channel, width, bus_order))
        manifest.append({
            "index": index_offset + index,
            "window_map": bits,
            "set_bits": [i for i, c in enumerate(bits) if c == "1"],
            "first_line": first,
            "last_line": len(lines) - 1,
            "start_mm": round(first * 25.4 / dpi, 3),
            "end_mm": round((len(lines) - 1) * 25.4 / dpi, 3),
        })
        lines.extend(blank)

    return lines, manifest


# ---------------------------------------------------------------------------
# The board
# ---------------------------------------------------------------------------

class Board:
    """The board over USB: connect, send a swath, arm, wait for the outcome."""

    def __init__(self, port: str, baudrate: int, arm_timeout: float,
                 chunk_lines: int = 64, pace_ms: float = 0.0):
        sys.path.insert(0, str(ROOT / "paintress-daemon"))
        from paintress_daemon.serial_manager import SerialManager
        from paintress_daemon.swath_handler import SwathHandler
        from paintress_daemon.protocol import encode_data_frame
        from paintress_daemon.paintress_protocol import Event

        self._Event = Event
        self._encode_data_frame = encode_data_frame
        self.arm_timeout = arm_timeout
        self.chunk_lines = chunk_lines
        self.pace_ms = pace_ms
        self.serial = SerialManager(port=port, baudrate=baudrate)
        if not self.serial.connect():
            raise SystemExit(f"could not open {port}")
        self.handler = SwathHandler(self.serial)
        self.serial.set_frame_callback(self.handler.handle_frame)
        self.handler.set_event_callback(self._on_event)
        self.handler.set_log_callback(
            lambda level, text: print(f"   [fw] {text}"))
        self._events: List[Tuple[int, dict]] = []

    def _on_event(self, event, data):
        self._events.append((event, data))

    def check_identity(self, force: bool) -> dict:
        identity = self.handler.identify()
        if identity is None:
            raise SystemExit("firmware did not answer IDENTIFY")
        expected = head_profiles.FRAME_FINGERPRINT
        actual = identity["profile_hash"]
        print(f"   firmware frame 0x{actual:08X}, build {identity['fw_build']}")
        if actual != expected:
            message = (f"firmware frame 0x{actual:08X} != this host's "
                       f"0x{expected:08X}: it would shift these lines out "
                       f"wrong, and nothing you read off the head would mean "
                       f"anything")
            if not force:
                raise SystemExit(message + " (--force to override)")
            print(f"   WARNING: {message}")
        return identity

    def send_paced(self, swath_id: int, lines: List[bytes]) -> Optional[str]:
        """Send a swath in bounded chunks, waiting for each to drain.

        The daemon's own send_lines() queues the whole swath at once and lets
        the TX thread coalesce it into 64 KB writes. Against a CDC device that
        is not draining that fast, one write blows the 1 s write timeout, the
        port is declared lost, and a probe dies before the head ever fires.
        Sending a bounded chunk and waiting for the queue to empty caps every
        write at chunk_lines * 147 bytes, so a link that sustains even a
        fraction of the rate finishes instead of failing.

        Returns None on success, or a message describing the failure.
        """
        if not lines:
            return "nothing to send"
        resp = self.handler.begin_swath(swath_id, len(lines))
        if not resp.success:
            return f"BEGIN_SWATH refused: {resp.error_msg}"

        started = time.monotonic()
        for start in range(0, len(lines), self.chunk_lines):
            chunk = lines[start:start + self.chunk_lines]
            blob = b"".join(self._encode_data_frame(line) for line in chunk)
            if not self.serial.send_bytes(blob):
                return f"TX queue rejected lines {start}..{start + len(chunk)}"
            if not self.serial.wait_tx_empty(timeout=10.0):
                why = ("the port was declared lost"
                       if not self.serial.is_connected()
                       else "the queue never drained")
                return (f"link stalled after {start + len(chunk)}/{len(lines)} "
                        f"lines: {why}")
            if self.pace_ms:
                time.sleep(self.pace_ms / 1000.0)

        elapsed = time.monotonic() - started
        sent = len(lines) * len(lines[0])
        print(f"   sent {len(lines)} lines ({sent / 1024:.0f} KB) in "
              f"{elapsed:.2f} s, {sent / 1024 / max(elapsed, 1e-6):.0f} KB/s")

        resp = self.handler.end_swath()
        if not resp.success:
            return f"END_SWATH failed: {resp.error_msg}"
        return None

    def fire(self, swath_id: int, lines: List[bytes],
             line_delay_us: int) -> str:
        """Send one swath, arm it, and wait for the head to report an outcome."""
        self._events.clear()
        failure = self.send_paced(swath_id, lines)
        if failure:
            return f"transfer failed: {failure}"

        resp = self.handler.print_swath(swath_id, line_delay_us)
        if not resp.success:
            return f"arm refused: {resp.error_msg}"

        print(f"   armed ({len(lines)} lines); waiting for the rising edge "
              f"on the trigger pin")
        deadline = time.monotonic() + self.arm_timeout
        while time.monotonic() < deadline:
            for event, data in list(self._events):
                if event == self._Event.PRINT_COMPLETE:
                    return "printed"
                if event == self._Event.TRIGGER_TIMEOUT:
                    return "no trigger arrived"
                if event == self._Event.PRINT_ERROR:
                    return f"print error: {data}"
            time.sleep(0.02)
        return "timed out waiting for the firmware to report anything"

    def stop(self, emergency: bool):
        """Leave the board safe, and usable next time.

        On an abnormal exit ABORT comes first, because it is the immediate
        electrical kill. It also latches: every later ARM is NACKed DAC_LATCHED
        until the chip reboots, so RESET follows it; otherwise this probe
        would leave the board refusing to print and the next run would look
        like a mysterious failure. A clean finish needs neither: the swath
        already ran to its end.
        """
        if emergency:
            for step in (self.handler.abort, self.handler.reset):
                try:
                    step()
                except Exception:
                    pass
        try:
            self.serial.disconnect()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def report_plan(maps: Sequence[str], manifest: Optional[List[Dict]],
                line_count: int, line_delay_us: int, dpi: int):
    print(f"\n{len(maps)} candidate map(s), {line_count} lines, "
          f"{line_delay_us} us/line "
          f"({line_count * line_delay_us / 1e6:.2f} s, "
          f"{line_count * 25.4 / dpi:.1f} mm at {dpi} dpi)")
    if manifest:
        print("\n  #   window map                          set bits      "
              "lines          mm")
        for entry in manifest:
            print(f"  {entry['index']:<3} {entry['window_map']}  "
                  f"{str(entry['set_bits']):<13} "
                  f"{entry['first_line']:>5}-{entry['last_line']:<5} "
                  f"{entry['start_mm']:>7.1f}-{entry['end_mm']:.1f}")


def run_sweep(board: Optional[Board], lines: List[bytes],
              manifest: List[Dict], line_delay_us: int, swath_id: int):
    if board is None:
        print("\n(dry run: nothing sent)")
        return
    outcome = board.fire(swath_id, lines, line_delay_us)
    print(f"\n   -> {outcome}")
    if outcome == "printed":
        print("\nRead the strip against the table above: the block that "
              "ejected ink names the map.\nIf several did, the head accepts "
              "more than one; if none did, try --set order, then --set pair.")


def run_step(board: Optional[Board], layout_meta: Dict, maps: Sequence[str],
             channel: Optional[str], width: int, bus_order: Optional[str],
             line_delay_us: int, pause: Optional[float]):
    for index, bits in enumerate(maps):
        lines = block_lines(layout_meta, bits, channel, width, bus_order)
        print(f"\n[{index + 1}/{len(maps)}] {bits}  ({describe_map(bits)})")
        if board is None:
            print(f"   (dry run: {len(lines)} lines built)")
            continue
        outcome = board.fire((index % 65535) + 1, lines, line_delay_us)
        print(f"   -> {outcome}")
        if outcome != "printed":
            # Not a verdict on this candidate: the setup did not fire at all.
            # Carrying on would burn the whole ladder on the same fault, and a
            # swath that never printed keeps holding its slot: after two of
            # these the firmware has no slot left and starts refusing outright.
            print("\nStopping: that is a setup failure, not a candidate "
                  "failure.\nA missing trigger is the usual cause: ARM waits "
                  "for a rising edge on GPIO 45.")
            return
        if index + 1 == len(maps):
            break
        if pause is None:
            try:
                input("   press Enter for the next candidate (Ctrl-C to stop) ")
            except EOFError:
                return
        else:
            time.sleep(pause)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Walk candidate 32-bit window-enable maps on a head.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  # See the plan for the 32 one-hot candidates without touching hardware.
  python tools/window_map_probe.py --dry-run

  # One sweep carrying all 32; trigger it once and read the strip.
  python tools/window_map_probe.py --port COM5 --speed-mm-s 100

  # Stationary head, one candidate at a time, black only (bus 0).
  python tools/window_map_probe.py --port COM5 --mode step --channel K

  # Prove the method before trusting it: run the same sweep on the reference
  # head, whose answer is known. Block 14 (and only block 14) should eject.
  python tools/window_map_probe.py --port COM5 --head c6n90 --dpi 630
""")
    parser.add_argument("--port", help="serial port (COM5, /dev/ttyACM0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and report the plan; open nothing")
    parser.add_argument("--mode", choices=["sweep", "step"], default="sweep",
                        help="one swath carrying every candidate (sweep, "
                             "default), or one ARM per candidate (step)")
    parser.add_argument("--head", default="c4n180",
                        help="head to probe (default: c4n180)")
    parser.add_argument("--ink-map",
                        help="comma-separated inks per slot; default is the "
                             "head's own plumbing")
    parser.add_argument("--set", dest="candidate_set", default="single",
                        choices=["baseline", "single", "order", "pair"],
                        help="candidate ladder (default: single, the 32 "
                             "one-hot maps)")
    parser.add_argument("--maps",
                        help="explicit candidates: comma-separated 32-bit "
                             "strings, overriding --set")
    parser.add_argument("--skip", type=int, default=0,
                        help="drop the first N candidates; with --take, how "
                             "a set too long for one sweep is run in batches")
    parser.add_argument("--take", type=int,
                        help="probe at most N candidates")
    parser.add_argument("--channel",
                        help="fire only this ink (K, C, M, Y); on c4n180 K "
                             "isolates bus 0 and any colour isolates bus 1")
    parser.add_argument("--block-lines", type=int, default=32,
                        help="columns per candidate (default: 32)")
    parser.add_argument("--gap-lines", type=int, default=16,
                        help="blank columns between candidates in sweep mode "
                             "(default: 16)")
    parser.add_argument("--dpi", type=int, default=720,
                        help="column pitch, for the mm column of the manifest "
                             "(default: 720)")
    parser.add_argument("--speed-mm-s", type=float,
                        help="gantry cruise speed; sets the firing interval to "
                             "match --dpi")
    parser.add_argument("--line-delay-us", type=int,
                        help="firing interval, overriding --speed-mm-s "
                             "(default: 1000)")
    parser.add_argument("--bus-order", choices=["straight", "interleaved"],
                        help="override the bus order (specified; a last resort)")
    parser.add_argument("--swath-id", type=int, default=1)
    parser.add_argument("--baudrate", type=int, default=2000000)
    parser.add_argument("--chunk-lines", type=int, default=64,
                        help="lines per serial write; each chunk is waited out "
                             "before the next is queued, so this caps the write "
                             "size against the port's 1 s write timeout "
                             "(default: 64, about 9 KB)")
    parser.add_argument("--pace-ms", type=float, default=0.0,
                        help="idle milliseconds between chunks, if the link "
                             "needs more breathing room than draining gives it")
    parser.add_argument("--purge", type=int, metavar="CHANNEL",
                        help="run PURGE on this channel and exit; the "
                             "companion to tools/gen_channel_masks.py, which "
                             "bakes candidate window maps into the firmware's "
                             "purge table. Needs no trigger and no transfer.")
    parser.add_argument("--purge-pulses", type=int, default=10,
                        help="pulses per purge (default: 10)")
    parser.add_argument("--link-test", action="store_true",
                        help="send one short swath and report the throughput, "
                             "then stop; run this first when a transfer fails")
    parser.add_argument("--arm-timeout", type=float, default=30.0,
                        help="seconds to wait for the trigger and the outcome "
                             "(default: 30)")
    parser.add_argument("--pause", type=float,
                        help="step mode: seconds between candidates instead of "
                             "waiting for Enter")
    parser.add_argument("--manifest", default="window_map_probe.json",
                        help="where to write the candidate/position table")
    parser.add_argument("--force", action="store_true",
                        help="probe even if the firmware reports a different "
                             "frame")
    args = parser.parse_args(argv)

    if not args.dry_run and not args.port:
        parser.error("--port is required unless --dry-run")

    if args.line_delay_us:
        line_delay_us = args.line_delay_us
    elif args.speed_mm_s:
        # One column is 25.4/dpi mm; at speed_mm_s that is
        # 25.4e6 / (dpi * speed) microseconds.
        line_delay_us = max(1, round(25.4e6 / (args.dpi * args.speed_mm_s)))
    else:
        line_delay_us = 1000
    if line_delay_us < 20:
        print(f"WARNING: {line_delay_us} us per column is faster than the head "
              f"is likely to fire; check --dpi/--speed-mm-s.")

    ink_map = args.ink_map.split(",") if args.ink_map else None
    layout_meta = build_layout(args.head, ink_map, args.dpi)

    explicit = None
    if args.maps:
        explicit = [m.strip() for m in args.maps.split(",")]
        for bits in explicit:
            if len(bits) != WINDOW_BITS or set(bits) - {"0", "1"}:
                parser.error(f"{bits!r} is not {WINDOW_BITS} bits of 0/1")

    baseline = head_profiles.head(args.head).window_enable_map
    maps = candidate_maps(args.candidate_set, baseline, explicit)
    total = len(maps)
    maps = maps[args.skip:]
    if args.take is not None:
        maps = maps[:args.take]
    if not maps:
        parser.error(f"--skip {args.skip} leaves nothing of {total} candidates")
    if len(maps) != total:
        print(f"batch: candidates {args.skip}..{args.skip + len(maps) - 1} "
              f"of {total}")

    print(f"Head    : {args.head}  (plumbed "
          f"{[s['ink'] for s in layout_meta['slots']]})")
    print(f"Firing  : {args.channel or 'every plumbed nozzle'}")
    print(f"Baseline: {baseline}  ({describe_map(baseline)}), the shipped "
          f"guess, carried over from the reference head")

    manifest = None
    lines: List[bytes] = []
    # A purge drives a line the firmware already holds, so there is nothing
    # here to build for it.
    if args.purge is not None:
        pass
    elif args.mode == "sweep":
        lines, manifest = build_sweep(layout_meta, maps, args.channel,
                                      args.block_lines, args.gap_lines,
                                      args.bus_order, args.dpi, args.skip)
        if len(lines) > MAX_PROBE_LINES:
            fits = MAX_PROBE_LINES // (args.block_lines + args.gap_lines)
            parser.error(
                f"{len(lines)} lines is more than the {MAX_PROBE_LINES}-line "
                f"cap. Run it in batches (--take {fits}, then --skip {fits} "
                f"--take {fits}, ...), or shrink --block-lines/--gap-lines.")
        report_plan(maps, manifest, len(lines), line_delay_us, args.dpi)
        Path(args.manifest).write_text(
            json.dumps({"head": args.head, "channel": args.channel,
                        "line_delay_us": line_delay_us, "dpi": args.dpi,
                        "blocks": manifest}, indent=2), encoding="utf-8")
        print(f"\nmanifest -> {args.manifest}")
    else:
        report_plan(maps, None, args.block_lines * len(maps), line_delay_us,
                    args.dpi)

    if args.dry_run:
        if args.mode == "sweep":
            run_sweep(None, lines, manifest or [], line_delay_us, args.swath_id)
        else:
            run_step(None, layout_meta, maps, args.channel, args.block_lines,
                     args.bus_order, line_delay_us, args.pause)
        return 0

    print(f"\nopening {args.port} ...")
    board = Board(args.port, args.baudrate, args.arm_timeout,
                  args.chunk_lines, args.pace_ms)
    emergency = True
    try:
        board.check_identity(args.force)
        if args.purge is not None:
            print(f"\npurging channel {args.purge}, "
                  f"{args.purge_pulses} pulses ...")
            resp = board.handler.purge(args.purge, args.purge_pulses)
            print(f"   -> {'done' if resp.success else resp.error_msg}")
            emergency = not resp.success
            return 0 if resp.success else 1
        if args.link_test:
            # Transfer only: no ARM, and every nozzle bit zero, so this cannot
            # eject ink even if something did trigger the board.
            probe = block_lines(layout_meta, BLANK_MAP, None, 256,
                                args.bus_order, fire=False)
            print(f"\nlink test: {len(probe)} silent lines, "
                  f"{args.chunk_lines} per write")
            failure = board.send_paced(args.swath_id, probe)
            print(f"   -> {failure or 'transfer OK'}")
            if failure:
                print("\nThe link cannot carry a swath yet. Try a smaller "
                      "--chunk-lines (16, then 8) and --pace-ms 2.")
            emergency = False
            return 0 if failure is None else 1
        if args.mode == "sweep":
            run_sweep(board, lines, manifest or [], line_delay_us,
                      args.swath_id)
        else:
            run_step(board, layout_meta, maps, args.channel, args.block_lines,
                     args.bus_order, line_delay_us, args.pause)
        emergency = False
    except KeyboardInterrupt:
        print("\ninterrupted; killing the head, then resetting the board")
        return 130
    finally:
        board.stop(emergency)
    return 0


if __name__ == "__main__":
    sys.exit(main())
