#!/usr/bin/env python3
"""
Standing test: stand 10 s, step off 5 s, stand 10 s. Works over USB or Bluetooth.

  py -3.13 standing_test.py --port COM9

Put the insoles on the floor (or in the shoes), feet OFF, before you press Enter.
It tells you when to STAND and when to STEP OFF, saves the raw data to recordings/, and prints per pad:
  unpressed level (from the step-off break), loaded level in each stand, drop, repeatability, saturation.
Close the Serial Monitor first. This checks that every pad responds and repeats under body weight.
It does NOT give calibration gains: each pad carries a different share of your weight.
"""
import argparse
import csv
import datetime
import os
import statistics
import sys
import time

import serial

CHANNELS = ["L_toe", "L_mid", "L_heel", "R_toe", "R_mid", "R_heel"]
N_COLS = 2 + len(CHANNELS)
STAND_S, BREAK_S, MARGIN_S = 10, 5, 2
MIN_DROP = 300        # smaller drop = no real response
SATURATED = 20        # at or below this = bottomed out
REPEAT_TOL = 0.25     # two stands should agree within 25 % of the drop


def parse_line(raw):
    try:
        parts = raw.decode("ascii", errors="ignore").strip().split(",")
        if len(parts) != N_COLS:
            return None
        return [int(p) for p in parts][2:]
    except ValueError:
        return None


def wait_for_data(ser, timeout=20):
    end = time.time() + timeout
    ser.reset_input_buffer()
    while time.time() < end:
        if parse_line(ser.readline()) is not None:
            return True
    return False


def record_phase(ser, seconds, label, beep_every=1):
    """Read for `seconds`, printing a countdown. Returns list of (t_since_phase_start, row)."""
    ser.reset_input_buffer()
    ser.readline()
    rows, t0 = [], time.perf_counter()
    shown = None
    while True:
        t = time.perf_counter() - t0
        if t >= seconds:
            break
        left = int(seconds - t) + 1
        if left != shown:
            print(f"   {label}  {left:2d} s", end="\r", flush=True)
            shown = left
        row = parse_line(ser.readline())
        if row is not None:
            rows.append((t, row))
    print(" " * 40, end="\r")
    return rows


def window(rows, seconds):
    """Rows from the middle of a phase (drop MARGIN_S at both ends)."""
    return [r for t, r in rows if MARGIN_S <= t <= seconds - MARGIN_S]


def col(rows, i):
    return [r[i] for r in rows]


def analyse(brk, s1, s2):
    out = []
    for i in range(6):
        base = statistics.median(col(brk, i))
        base_sd = statistics.pstdev(col(brk, i))
        l1, l2 = statistics.median(col(s1, i)), statistics.median(col(s2, i))
        sd1, sd2 = statistics.pstdev(col(s1, i)), statistics.pstdev(col(s2, i))
        d1, d2 = base - l1, base - l2
        low = min(min(col(s1, i)), min(col(s2, i)))
        mean_d = (d1 + d2) / 2
        rep = abs(d1 - d2) / mean_d if mean_d > 0 else None
        notes = []
        ok = True
        if mean_d < MIN_DROP:
            notes.append("NO RESPONSE: check wire, resistor, or the pad is not being loaded")
            ok = False
        if base < 3500:
            notes.append("unpressed level low: pad preloaded or shorted")
            ok = False
        if base_sd > 80:
            notes.append("noisy when unloaded: loose contact")
            ok = False
        if low <= SATURATED:
            notes.append("bottomed out: reading hits 0 under your weight (clipped)")
        if rep is not None and rep > REPEAT_TOL and mean_d >= MIN_DROP:
            notes.append(f"stands differ by {rep*100:.0f}%: repeat with the same posture")
        out.append(dict(ch=CHANNELS[i], base=base, l1=l1, l2=l2, d1=d1, d2=d2, rep=rep,
                        sd=max(sd1, sd2), low=low, ok=ok, notes=notes))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--subject", default="subject")
    ap.add_argument("--out", default="recordings")
    args = ap.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        sys.exit(f"Could not open {args.port}: {e}\nClose the Serial Monitor and try again.")
    print("Connecting (Bluetooth can take up to 20 s)...", flush=True)
    if not wait_for_data(ser):
        ser.close()
        sys.exit("No data from the ESP32. Check the port, the sketch and the power.")

    print("\nFeet OFF the insoles. Stand next to them, ready to step on with both feet together.")
    print("Plan: 3 s get ready, STAND 10 s, STEP OFF 5 s, STAND 10 s, STEP OFF 3 s.")
    input("Press Enter to begin...")
    all_rows = []
    phases = []
    for name, secs in [("GET READY (feet off)", 3), ("STAND ON BOTH INSOLES", STAND_S),
                       ("STEP OFF", BREAK_S), ("STAND ON BOTH INSOLES", STAND_S), ("STEP OFF", 3)]:
        print(f"-> {name}")
        rows = record_phase(ser, secs, name)
        phases.append((name, secs, rows))
        all_rows.extend(r for _, r in rows)
    ser.close()

    s1 = window(phases[1][2], STAND_S)
    brk = window(phases[2][2], BREAK_S)
    s2 = window(phases[3][2], STAND_S)
    if min(len(s1), len(brk), len(s2)) < 50:
        sys.exit("Too little data in a phase. Check the connection and run again.")

    res = analyse(brk, s1, s2)
    print("\n--- Standing test result ---")
    print(f"{'Pad':7s} {'unpressed':>9s} {'stand 1':>8s} {'stand 2':>8s} {'drop 1':>7s} {'drop 2':>7s} {'agree':>6s}  note")
    for r in res:
        agree = "-" if r["rep"] is None else f"{(1 - r['rep'])*100:.0f}%"
        note = "OK" if r["ok"] and not r["notes"] else "; ".join(r["notes"])
        print(f"{r['ch']:7s} {r['base']:9.0f} {r['l1']:8.0f} {r['l2']:8.0f} {r['d1']:7.0f} {r['d2']:7.0f} {agree:>6s}  {note}")

    dl = sum((r["d1"] + r["d2"]) / 2 for r in res[:3])
    dr = sum((r["d1"] + r["d2"]) / 2 for r in res[3:])
    if dl + dr > 0:
        print(f"\nShare of total drop: left {100*dl/(dl+dr):.0f}%, right {100*dr/(dl+dr):.0f}%  "
              "(rough guide to weight balance, not a force measurement)")
    bad = [r["ch"] for r in res if not r["ok"]]
    print("Fix before recording walks: " + ", ".join(bad) if bad else "All six pads responded under body weight.")

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.out, f"{args.subject}_standing_{stamp}.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "t_s"] + CHANNELS)
        for name, secs, rows in phases:
            for t, row in rows:
                w.writerow([name, f"{t:.3f}"] + row)
    print(f"\nSaved raw data: {path}")


if __name__ == "__main__":
    main()
