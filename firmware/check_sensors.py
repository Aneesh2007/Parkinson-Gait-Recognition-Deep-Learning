#!/usr/bin/env python3
"""
Check that all 6 FSR channels work, one at a time (run after wiring or replacing a sensor).

  python check_sensors.py --port COM5

For each channel it records 1 s with nothing pressed, then 4 s while you press and hold
that sensor with a finger, and reports whether the reading dropped. It also tells you if a
different channel moved instead (wires swapped) or if a channel is noisy (loose wire).
Close the Arduino Serial Monitor first.
"""
import argparse
import statistics
import sys
import time

import serial

CHANNELS = ["L_toe", "L_mid", "L_heel", "R_toe", "R_mid", "R_heel"]
LABELS = ["LEFT TOE", "LEFT MIDFOOT", "LEFT HEEL", "RIGHT TOE", "RIGHT MIDFOOT", "RIGHT HEEL"]
N_COLS = 2 + len(CHANNELS)
MIN_DROP = 400        # counts the reading must fall to count as "pressed"
WEAK_DROP = 150       # between this and MIN_DROP = responds but barely
CROSS_DROP = 300      # another channel falling this much while you press = suspicious
NOISY_STD = 80        # unpressed standard deviation above this = loose or floating wire


def parse_line(raw):
    try:
        parts = raw.decode("ascii", errors="ignore").strip().split(",")
        if len(parts) != N_COLS:
            return None
        return [int(p) for p in parts][2:]
    except ValueError:
        return None


def capture(ser, seconds):
    """Read channel values for `seconds`; returns a list of 6-value rows."""
    ser.reset_input_buffer()
    ser.readline()  # drop a partial line
    rows, end = [], time.time() + seconds
    while time.time() < end:
        row = parse_line(ser.readline())
        if row is not None:
            rows.append(row)
    return rows


def pct(values, q):
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def analyse(base, press, ch):
    """base, press: lists of 6-value rows. ch: index of the sensor that was pressed.
    Returns (verdict, details dict)."""
    col = lambda rows, i: [r[i] for r in rows]
    base_med = [statistics.median(col(base, i)) for i in range(6)]
    base_std = [statistics.pstdev(col(base, i)) for i in range(6)]
    low = [pct(col(press, i), 0.10) for i in range(6)]       # robust "pressed" level
    drop = [base_med[i] - low[i] for i in range(6)]
    info = dict(base=base_med[ch], low=low[ch], drop=drop[ch], std=base_std[ch], drops=drop)
    others = [(drop[i], i) for i in range(6) if i != ch and base_std[i] <= NOISY_STD] or [(0, ch)]
    big_other = max(others)
    wobble = base_std[ch] > NOISY_STD
    note = f" (unpressed reading wobbled, std {base_std[ch]:.0f}: keep hands off the pad before pressing, or check the joint)" if wobble else ""
    if drop[ch] >= MIN_DROP:
        if big_other[0] >= CROSS_DROP:
            return f"OK, but {CHANNELS[big_other[1]]} also moved (touching two pads, or crosstalk)" + note, info
        return "OK" + note, info
    if big_other[0] >= MIN_DROP:
        return f"WRONG CHANNEL: pressing this moved {CHANNELS[big_other[1]]} instead (swapped wires?)", info
    if wobble:
        return "NOISY (loose or floating wire)", info
    if base_med[ch] < 200:
        return "STUCK LOW (shorted to ground, or wrong pin)", info
    if drop[ch] >= WEAK_DROP:
        return "WEAK: responds but barely. Press the exact centre harder; if still weak check this pad's resistor and contact", info
    return "NO RESPONSE (not pressed hard enough, broken FSR, or wire not connected)", info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--press", type=float, default=2.0, help="seconds to hold each pad (default 2)")
    ap.add_argument("--reaction", type=float, default=1.0, help="seconds allowed to get your finger on the pad")
    args = ap.parse_args()
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        sys.exit(f"Could not open {args.port}: {e}\nClose the Serial Monitor and try again.")
    time.sleep(2.0)
    print("Pad check. Keep the insole flat on a table and press each pad with your thumb for 2 s.\n")

    results = []
    for ch in range(6):
        input(f"[{ch + 1}/6] {LABELS[ch]}: hands off, then press Enter...")
        print("      Hands OFF, wait...", flush=True)
        time.sleep(1.0)
        base = capture(ser, 1.5)
        if len(base) < 20:
            ser.close()
            sys.exit("Almost no data from the ESP32. Check the port, the cable and that the sketch is uploaded.")
        print(f"      PRESS and HOLD the {LABELS[ch]} pad firmly NOW, keep holding until I say release...", flush=True)
        time.sleep(args.reaction)
        press = capture(ser, args.press)
        print("      RELEASE.", flush=True)
        verdict, info = analyse(base, press, ch)
        results.append((ch, verdict, info))
        print(f"      unpressed {info['base']:.0f}, pressed {info['low']:.0f}, drop {info['drop']:.0f}  ->  {verdict}\n")
    ser.close()

    print("--- Summary ---")
    bad = 0
    for ch, verdict, info in results:
        print(f"{LABELS[ch]:14s} {verdict}")
        bad += not verdict.startswith("OK") or "wobbled" in verdict
    print("\nAll 6 channels OK." if bad == 0 else f"\n{bad} channel(s) need attention. Fix them and run this again.")


if __name__ == "__main__":
    main()
