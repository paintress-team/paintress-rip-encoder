#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 paintress-team
# SPDX-License-Identifier: GPL-3.0-or-later

"""Fire a purge sweep over USB and keep the score.

WHAT THIS IS FOR
    ``gen_channel_masks.py`` flashes one purge channel per candidate. Reading
    the answer off means firing a lot of channels and keeping a lot of results
    straight while you are looking at a printhead instead of a screen. This does
    the firing and the bookkeeping.

FIRING NEVER PROMPTS
    Enter fires the next channel and comes straight back. You look at the head,
    and if you want the result kept you type ONE character; it attaches to the
    channel that was fired last. Nothing blocks on you: the earlier version
    stopped for an answer after every purge, which is the wrong trade when the
    thing you actually want is to walk the whole sweep quickly and only write
    down the rows that did something.

        sweep> <enter>     fires the next channel
        sweep> -           nothing came out (of the one just fired)
        sweep> + CMY       ink came out, these colours
        sweep> 4           fires channel 4
        sweep> 4 +         records channel 4 without firing it

    Results are written to JSON after every entry, so a session survives being
    interrupted, which it will be, because you will be at the head.

THE FLASH DESCRIBES ITSELF
    Which channel is which, and what each one is PREDICTED to do, comes from the
    manifest the generator wrote for the flash. So the runner can tell you when
    a result contradicts the prediction instead of leaving you to notice, and it
    cannot drift out of step with the table on the board the way a hard-coded
    channel list does.

SAFETY
    A purge fires with no trigger and no motion. Put something under the head.
    ABORT latches the DAC off and every later purge is refused until RESET;
    ``reset`` here does that (the board reboots and reconnects).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
sys.path.insert(0, str(REPO / "encoder"))
sys.path.insert(0, str(ROOT / "paintress-daemon"))

import head_profiles                    # noqa: E402

# What an observation can be. The key is what you type; the value is what it
# means and whether it counts as the head having ejected.
OBSERVATIONS = {
    "-": ("nothing", False, "no ink"),
    "+": ("ink", True, "ink came out"),
    "?": ("unclear", None, "saw something, could not judge"),
}


class Layout:
    """Which purge channel is which, read from the flash's own manifest.

    Hard-coding the channel map here is how a table grows a row and a tool keeps
    reporting the old one, so gen_channel_masks.py writes what it emitted and
    this reads it back.
    """

    def __init__(self, path: Path):
        if not path.exists():
            raise SystemExit(
                f"no manifest at {path}. Generate the flash first:\n"
                f"    python tools/gen_channel_masks.py --mode windows "
                f"--head c4n180 \\\n"
                f"        -o ../paintress-firmware/engine/channels.c \\\n"
                f"        --header ../paintress-firmware/engine/channels.h")
        data = json.loads(path.read_text(encoding="utf-8"))
        self.head = data["head"]
        self.window_bits = data["window_bits"]
        self.channels: Dict[int, dict] = {c["channel"]: c
                                          for c in data["channels"]}
        self.count = len(self.channels)

    def entry(self, channel: int) -> dict:
        return self.channels.get(channel, {})

    def label(self, channel: int) -> str:
        entry = self.entry(channel)
        if not entry:
            return f"ch{channel} (not in the manifest)"
        parts = [f"ch{channel}", entry.get("name", "?")]
        if entry.get("code"):
            parts.append(f"code {entry['code']}")
        bits = [i for i, b in enumerate(entry.get("window_map", "")) if b == "1"]
        parts.append("all 32 bits" if len(bits) > 8 else
                     f"bit {','.join(map(str, bits))}" if bits else "no bits")
        must = entry.get("must_eject")
        if must is not None:
            parts.append("MUST EJECT" if must else "must stay silent")
        return "  ".join(parts)

    def by_role(self, role: str) -> List[int]:
        return sorted(c for c, e in self.channels.items()
                      if e.get("role") == role)


class Board:
    """The board over USB: identify, purge, reset."""

    # The firmware reports a purge's outcome as a LOG line, not in the ACK: the
    # ACK goes out when the command is queued to core 1, and core 1 answers
    # later over the inter-core event queue. Waiting for it is the difference
    # between "the head was driven and no ink came out" and "the head was never
    # driven", which look identical at the nozzle plate.
    PURGE_OK = "purge done"
    PURGE_REFUSED = "purge refused"

    def __init__(self, port: str, baudrate: int, quiet: bool = False):
        from paintress_daemon.serial_manager import SerialManager
        from paintress_daemon.swath_handler import SwathHandler

        self.quiet = quiet
        self.serial = SerialManager(port=port, baudrate=baudrate)
        if not self.serial.connect():
            raise SystemExit(f"could not open {port}")
        self.handler = SwathHandler(self.serial)
        self.serial.set_frame_callback(self.handler.handle_frame)
        self._logs: List[str] = []
        self.handler.set_log_callback(self._on_log)
        self.handler.set_event_callback(self._on_event)

    def _on_log(self, level, text):
        self._logs.append(text)
        if not self.quiet:
            print(f"   [fw] {text}")

    def _on_event(self, event, data):
        if not self.quiet:
            print(f"   [fw event] {event.name} {data}")

    def check_identity(self, force: bool) -> dict:
        identity = self.handler.identify()
        if identity is None:
            raise SystemExit("firmware did not answer IDENTIFY")
        actual = identity["profile_hash"]
        expected = head_profiles.FRAME_FINGERPRINT
        print(f"   firmware frame 0x{actual:08X}, build {identity['fw_build']}")
        if actual != expected:
            message = (f"firmware frame 0x{actual:08X} != this host's "
                       f"0x{expected:08X}: it shifts lines out differently, so "
                       f"nothing you read off the head would mean anything")
            if not force:
                raise SystemExit(message + " (--force to override)")
            print(f"   WARNING: {message}")
        return identity

    def purge(self, channel: int, pulses: int, timeout: float = 5.0) -> str:
        """Fire one channel and wait for the firmware to say what it did.

        Returns "driven" / "refused: ..." / "no outcome". The last one is the
        one worth stopping for: the command was accepted and core 1 never
        answered, so the head was never driven and no amount of staring at the
        nozzles will say why.
        """
        self._logs.clear()
        response = self.handler.purge(channel, pulses)
        if not response.success:
            # A channel past CH_COUNT means the flashed firmware is not the
            # build this manifest describes: the commonest way to run against
            # the wrong flash.
            return f"refused: {response.error_msg or 'no reason given'}"

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for text in list(self._logs):
                if text.startswith(self.PURGE_REFUSED):
                    return f"refused: {text}"
                if text.startswith(self.PURGE_OK):
                    return "driven"
            time.sleep(0.02)
        return "no outcome"

    def status(self) -> Optional[dict]:
        return self.handler.get_status()

    def reset(self) -> bool:
        return self.handler.reset()

    def close(self):
        try:
            self.serial.disconnect()
        except Exception:
            pass


class Session:
    """The record of what each channel did, persisted after every entry."""

    def __init__(self, path: Path, layout: Layout, pulses: int,
                 fresh: bool = False, note: str = ""):
        self.path = path
        self.layout = layout
        self.head = layout.head
        self.pulses = pulses
        self.note = note
        self.results: Dict[int, dict] = {}
        if path.exists() and fresh:
            # Archive rather than overwrite: a superseded session is still the
            # record of what a head did, and the reason a result was thrown out
            # is usually only obvious later.
            stamp = time.strftime("%Y%m%d-%H%M%S")
            archived = path.with_suffix(f".{stamp}.json")
            path.rename(archived)
            print(f"   archived the previous session to {archived.name}")
        elif path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("head") == layout.head:
                self.results = {int(k): v for k, v in saved["results"].items()}
                print(f"   resuming {path.name}: {len(self.results)} "
                      f"channel(s) already recorded"
                      + (f" ({saved['note']})" if saved.get("note") else ""))
                print("   !! if this is a DIFFERENT PHYSICAL HEAD, stop and "
                      "rerun with --fresh.\n      Nothing distinguishes one "
                      "head from another in here, and a silent\n      channel "
                      "measured on a tired head reads exactly like a shut "
                      "window.")
            else:
                # Results from another head's flash would be silently read as
                # this one's, and channel numbers mean different things.
                print(f"   ignoring {path.name}: it is a {saved.get('head')} "
                      f"session, not {layout.head}")

    def record(self, channel: int, key: str, colours: str, note: str) -> str:
        observation, ejected, _ = OBSERVATIONS[key]
        self.results[channel] = {
            "observation": observation,
            "ejected": ejected,
            "colours": colours,
            "note": note,
            "pulses": self.pulses,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.save()
        return self.verdict(channel)

    def verdict(self, channel: int) -> str:
        """Whether a recorded result matches what the flash predicted."""
        entry = self.results.get(channel)
        must = self.layout.entry(channel).get("must_eject")
        if entry is None or must is None or entry["ejected"] is None:
            return ""
        if entry["ejected"] == must:
            return "as predicted"
        return ("!! SURPRISE: this channel was predicted to "
                + ("EJECT and did not" if must else "stay silent and ejected"))

    def save(self):
        self.path.write_text(json.dumps({
            "head": self.head,
            "note": self.note,
            "results": {str(k): v for k, v in sorted(self.results.items())},
        }, indent=2), encoding="utf-8")

    def next_unrecorded(self, after: int = -1) -> Optional[int]:
        for channel in sorted(self.layout.channels):
            if channel > after and channel not in self.results:
                return channel
        return None

    def table(self) -> str:
        lines = [f"  {'ch':>3}  {'name':<14} {'code':<5} {'saw':<8} "
                 f"{'colours':<8} note",
                 "  " + "-" * 62]
        for channel in sorted(self.layout.channels):
            entry = self.layout.entry(channel)
            got = self.results.get(channel)
            lines.append(
                f"  {channel:>3}  {entry.get('name', '?'):<14} "
                f"{entry.get('code', ''):<5} "
                f"{(got['observation'] if got else '-- not run'):<8} "
                f"{(got['colours'] if got else ''):<8} "
                f"{got['note'] if got else ''}")
        return "\n".join(lines)

    # -- reading the sweep -------------------------------------------------

    def _ejected(self, channel: int) -> Optional[bool]:
        entry = self.results.get(channel)
        return entry["ejected"] if entry else None

    def summary(self) -> str:
        out = ["", "SUMMARY", ""]

        # Controls first: without them the rest means nothing.
        for channel in self.layout.by_role("control"):
            saw = self._ejected(channel)
            if saw is False:
                out += [f"  !! THE CONTROL (ch{channel}) WAS SILENT.",
                        "",
                        "     It sets every window bit with every nozzle at 01,",
                        "     so it fires through whichever window this board",
                        "     opens, whatever that is. If it ejects nothing the",
                        "     window map is not what is wrong; look at the",
                        "     head (prime, ink, wiring) or at whether the flash",
                        "     really took. Nothing below is interpretable.", ""]
            elif saw:
                out += [f"  control ch{channel} ejected: the head fires and the",
                        "  rows below are readable.", ""]

        for channel in self.layout.by_role("negative_control"):
            if self._ejected(channel):
                out += ["  !! the no-bits control EJECTED. The 32 bits do not",
                        "     gate firing at all, so nothing else here means",
                        "     anything.", ""]

        # Per window: the 01 row and the 00 row, paired.
        rows: Dict[int, Dict[str, int]] = {}
        for channel, entry in self.layout.channels.items():
            if entry.get("role") == "window" and entry.get("window") is not None:
                rows.setdefault(entry["window"], {})[entry["code"]] = channel

        # Both rows drive the SAME bit (code 01's) and differ only in what
        # the nozzles carry. So "ink" on the 01 row means the window opens for
        # our code, and "ink" on the 00 row means that bit reaches the blanks
        # as well, which disqualifies it however well it prints.
        usable, leaks, shut, partial, untested = [], [], [], [], []
        for window in sorted(rows):
            drop = self._ejected(rows[window].get("01", -1))
            blank = self._ejected(rows[window].get("00", -1))
            if drop is None and blank is None:
                untested.append(window)
            elif drop and blank is False:
                usable.append(window)
            elif blank:
                leaks.append(window)
            elif drop is False and blank is False:
                shut.append(window)
            else:
                partial.append(window)

        out += ["  window   fires 01   fires 00   reading",
                "  " + "-" * 56]
        for window in sorted(rows):
            drop, blank = (self._ejected(rows[window].get(code, -1))
                           for code in ("01", "00"))

            def mark(value):
                return "ink" if value else "--" if value is False else "?"

            reading = ("USABLE: prints, and leaves the white white"
                       if window in usable else
                       "leaks: this bit fires the blanks too"
                       if window in leaks else
                       "shut" if window in shut else
                       "half tested" if window in partial else "not tested")
            out.append(f"  {window:>6}   {mark(drop):<10} {mark(blank):<10} "
                       f"{reading}")
        out.append("")

        if usable:
            bits = sorted(int(b) for window in usable
                          for b, flag in enumerate(
                              self.layout.entry(rows[window]["01"])["window_map"])
                          if flag == "1")
            candidate = "".join("1" if i in bits else "0"
                                for i in range(self.layout.window_bits))
            out += [f"  Window(s) {usable} fire code 01 and not code 00 --",
                    "  which is exactly what a print needs.", "",
                    f"      {candidate}", "",
                    "      python encoder/encoder.py <rip.json> -o <job.json> \\",
                    f"          --window-map {candidate}"]
            if len(usable) > 1:
                out += ["",
                        "  More than one window works. They are separate firing",
                        "  opportunities for the same code, so setting all of",
                        "  them fires each nozzle once per window: a bigger",
                        "  drop. Start with one and add if it under-inks."]
        elif leaks:
            out += [f"  Window(s) {leaks} fire the blanks as well, so no subset",
                    "  of them can be the map however well they print. Keep",
                    "  going through the untested windows."]
        else:
            out += ["  No window has fired code 01 alone yet."]
        if partial:
            out += ["", f"  Window(s) {partial} have only one of their two rows"
                        f" run --",
                    "  a window that prints is not usable until the blanks row",
                    "  has been seen to stay dry."]

        if untested:
            out += ["", f"  windows not tested: {untested}"]
        return "\n".join(out)


HELP = """
firing: never prompts, comes straight back
  <enter>      fire the next channel with no result yet
  <n>          fire channel n

recording: applies to the channel fired last
  -            nothing came out
  +            ink came out           (+ CMY to name the colours)
  ?            saw ink, could not judge
  <n> -        record channel n without firing it
  -- text      anything after -- is a free note

other
  pulses <n>   change the pulse count (currently {pulses})
  table        every channel and what was recorded
  summary      what the results add up to, and the map they imply
  status       ask the board its state: dac_latched above all
  reset        reboot the board (clears a latched DAC after an ABORT)
  quiet        stop echoing firmware log lines
  help         this
  quit         save and exit
"""


def parse_entry(command: str, last: Optional[int], layout: Layout):
    """A recording command -> (channel, key, colours, note), or None.

    Accepts "+", "+ CMY", "4 -", "+ CM -- weak on yellow". Returns None when
    the command is not a recording at all, so the caller can treat it as a
    channel to fire.
    """
    body, _, note = command.partition("--")
    tokens = body.split()
    if not tokens:
        return None
    channel = last
    if tokens[0].isdigit() and len(tokens) > 1:
        channel, tokens = int(tokens[0]), tokens[1:]
    if tokens[0] not in OBSERVATIONS:
        return None
    if channel is None:
        print("   nothing has been fired yet; give a channel, e.g. '4 +'")
        return "handled"
    if channel not in layout.channels:
        print(f"   channel must be 0..{layout.count - 1}")
        return "handled"
    colours = "".join(tokens[1:]).upper()
    return channel, tokens[0], colours, note.strip()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Fire a purge sweep and keep the score.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""example:
  # flash the sweep first
  python tools/gen_channel_masks.py --mode windows --head c4n180 \\
      -o ../paintress-firmware/engine/channels.c \\
      --header ../paintress-firmware/engine/channels.h

  # then, with the board on USB and something under the head
  python tools/purge_sweep_runner.py --port COM3
""")
    parser.add_argument("--port", required=True, help="serial port, e.g. COM3")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--manifest",
                        default=str(Path(__file__).with_name(
                            "sweep_manifest.json")),
                        help="the channel layout gen_channel_masks.py wrote "
                             "for the flash now on the board")
    parser.add_argument("--pulses", type=int, default=100,
                        help="pulses per purge (default 100). Only the first "
                             "is window-synchronised; the rest are bare DAC "
                             "pulses that re-fire whatever the head latched, "
                             "so they make a result easier to SEE without "
                             "re-testing the map")
    parser.add_argument("--results", default="window_sweep_results.json",
                        help="where to record results (resumed if it exists)")
    parser.add_argument("--fresh", action="store_true",
                        help="start a new session, archiving any existing "
                             "results file. USE THIS WHEN THE PHYSICAL HEAD "
                             "CHANGED: results carry the head's name but not "
                             "its identity, and a silent channel from a tired "
                             "head reads exactly like a shut window")
    parser.add_argument("--note", default="",
                        help="what this session is, e.g. 'new head 2026-07-28'. "
                             "Stored with the results")
    parser.add_argument("--quiet", action="store_true",
                        help="do not echo firmware log lines between purges")
    parser.add_argument("--force", action="store_true",
                        help="run even if the firmware's frame fingerprint "
                             "disagrees with this host's")
    args = parser.parse_args(argv)

    layout = Layout(Path(args.manifest))
    session = Session(Path(args.results), layout, args.pulses,
                      fresh=args.fresh, note=args.note)
    print(f"   flash: {layout.head}, {layout.count} channels"
          + (f", {args.note}" if args.note else ""))

    print(f"connecting to {args.port}...")
    board = Board(args.port, args.baudrate, quiet=args.quiet)
    try:
        board.check_identity(args.force)

        # Before anything else: a latched DAC makes every purge below a silent
        # no-op, and it is the one failure that looks exactly like a dry head.
        state = board.status()
        if state is None:
            print("   WARNING: the board did not answer GET_STATUS")
        elif state.get("dac_latched"):
            print("   !! dac_latched is SET: an ABORT or a safety fault killed "
                  "the DAC and the\n      latch is sticky. NOTHING will eject "
                  "until you reboot. Type 'reset'.")
        else:
            print("   dac_latched clear: the head can fire")

        print(HELP.format(pulses=args.pulses))
        controls = layout.by_role("control")
        if controls:
            print(f"   START WITH CHANNEL {controls[0]}: the control. If it "
                  f"is silent,\n   stop: the map is not the variable and "
                  f"nothing else would mean anything.")
        print("   Put something under the head. A purge fires immediately.\n")

        last: Optional[int] = None
        current = -1
        while True:
            try:
                command = input("sweep> ").strip()
            except EOFError:
                break
            if command in ("quit", "q", "exit"):
                break
            if command == "help":
                print(HELP.format(pulses=session.pulses))
                continue
            if command == "table":
                print(session.table())
                continue
            if command == "summary":
                print(session.summary())
                continue
            if command == "quiet":
                board.quiet = not board.quiet
                print(f"   firmware log echo {'off' if board.quiet else 'on'}")
                continue
            if command == "status":
                state = board.status()
                if state is None:
                    print("   the board did not answer GET_STATUS")
                    continue
                for key, value in state.items():
                    flag = ("   <-- the head cannot fire until a reset"
                            if key == "dac_latched" and value else "")
                    print(f"   {key:<20} {value}{flag}")
                continue
            if command == "reset":
                print("   rebooting the board..." if board.reset()
                      else "   RESET was not acknowledged")
                continue
            if command.startswith("pulses"):
                _, _, value = command.partition(" ")
                try:
                    session.pulses = max(1, min(255, int(value)))
                except ValueError:
                    print("   pulses needs a number, e.g. 'pulses 20'")
                    continue
                print(f"   pulses = {session.pulses}")
                continue

            # A recording, applied to whatever was fired last.
            entry = parse_entry(command, last, layout)
            if entry == "handled":
                continue
            if entry is not None:
                channel, key, colours, note = entry
                verdict = session.record(channel, key, colours, note)
                print(f"   ch{channel}: {OBSERVATIONS[key][2]}"
                      + (f", {colours}" if colours else "")
                      + (f"   {verdict}" if verdict else ""))
                continue

            # Otherwise it is a channel to fire.
            if not command:
                nxt = session.next_unrecorded(current)
                if nxt is None:
                    print("   every channel has a result. 'summary' to read "
                          "it, 'quit' to stop.")
                    continue
                channel = nxt
            else:
                try:
                    channel = int(command)
                except ValueError:
                    print(f"   '{command}' is not a command, a channel or a "
                          f"result. 'help' for the list.")
                    continue
                if channel not in layout.channels:
                    print(f"   channel must be 0..{layout.count - 1}")
                    continue

            print(f"   firing {layout.label(channel)} "
                  f"[{session.pulses} pulses]")
            outcome = board.purge(channel, session.pulses)

            if outcome.startswith("refused"):
                print(f"   REFUSED: {outcome}")
                if "LATCHED" in outcome.upper():
                    print("   The DAC is latched off: an ABORT or a safety "
                          "fault killed it and\n   the latch is sticky until a "
                          "reboot. Type 'reset', then retry.")
                elif channel >= 7:
                    print("   (channels above 6 need this flash AND its "
                          "--header; if the board\n   still has the shipped "
                          "7-channel table, CH_COUNT is 7 and the rest\n   are "
                          "NACKed)")
                continue

            if outcome == "no outcome":
                print("   !! THE FIRMWARE NEVER REPORTED BACK. The command was "
                      "accepted and core 1\n      never answered, so the head "
                      "was NOT driven. Try 'status', then 'reset'.")
                continue

            last = current = channel

        print(session.summary())
        print(f"\nresults in {session.path}")
    finally:
        board.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
