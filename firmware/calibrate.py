#!/usr/bin/env python3
"""
Per-sensor calibration for the insole. Run after wiring, before recording walks.

  py -3.13 calibrate.py --port COM9

Step 1: insole flat on a hard table, nothing on it. Measures each sensor's unpressed level.
Step 2: one sensor at a time, load it with the SAME standard load (best: a full 1 L bottle
standing on a coin-sized piece of cardboard centred on the round pad; a thumb press is less
repeatable). Measures how far each sensor's reading drops.
Result: calibration.json with a gain per sensor, so a weak and a sensitive sensor can be compared.
Works over USB (COM5) or Bluetooth (the Outgoing port). Close the Serial Monitor first.
"""
import argparse
import datetime
import json
import statistics
import sys
import time

import serial

CHANNELS = ["L_toe", "L_mid", "L_heel", "R_toe", "R_mid", "R_heel"]
LABELS = ["LEFT TOE", "LEFT MIDFOOT", "LEFT HEEL", "RIGHT TOE", "RIGHT MIDFOOT", "RIGHT HEEL"]
N_COLS = 2 + len(CHANNELS)
NOISY_STD = 80          # unpressed std above this = loose or floating wire
MIN_DROP = 300          # smaller drop than this = no real response
SATURATED = 20          # reading at or below this = bottomed out at zero


def parse_line(raw):
    try:
        parts = raw.decode("ascii", errors="ignore").strip().split(",")
        if len(parts) != N_COLS:
            return None
        return [int(p) for p in parts][2:]
    except ValueError:
        return None


def wait_for_data(ser, timeout=20):
    """Bluetooth can take several seconds to connect after the port is opened."""
    end = time.time() + timeout
    ser.reset_input_buffer()
    while time.time() < end:
        if parse_line(ser.readline()) is not None:
            return True
    return False


def capture(ser, seconds):
    ser.reset_input_buffer()
    ser.readline()
    rows, end = [], time.time() + seconds
    while time.time() < end:
        row = parse_line(ser.readline())
        if row is not None:
            rows.append(row)
    return rows


def col(rows, i):
    return [r[i] for r in rows]


def plateau_level(values, base):
    """Pressed level: median of the readings near the lowest point (the held part of the press)."""
    k = 10
    smooth = [sum(values[i:i + k]) / k for i in range(max(len(values) - k, 1))] or values
    low = min(smooth)
    thr = low + 0.15 * max(base - low, 1)
    plateau = [v for v in values if v <= thr]
    return statistics.median(plateau) if len(plateau) >= 10 else low


def measure(base, press, ch):
    """base: unpressed medians (6). press: rows while sensor ch was loaded. Returns dict."""
    level = plateau_level(col(press, ch), base[ch])
    drop = base[ch] - level
    others = []
    for i in range(6):
        if i != ch:
            lo = sorted(col(press, i))[int(0.10 * len(press))]
            others.append((base[i] - lo, i))
    cross = max(others)
    return dict(level=level, drop=drop, saturated=level <= SATURATED,
                cross_drop=cross[0], cross_ch=cross[1])


def verdict(m):
    notes = []
    if m["drop"] < MIN_DROP:
        if m["cross_drop"] >= 400:
            return f"WRONG CHANNEL? {CHANNELS[m['cross_ch']]} moved instead", False
        return "NO RESPONSE (not loaded, wire not connected, or bad FSR)", False
    if m["cross_drop"] >= 0.5 * m["drop"] and m["cross_drop"] >= MIN_DROP:
        notes.append(f"{CHANNELS[m['cross_ch']]} also moved (two sensors loaded, or crosstalk)")
    if m["saturated"]:
        notes.append("bottomed out at 0: gain is only a lower bound, use a lighter load")
    strength = "strong" if m["drop"] >= 2000 else "medium" if m["drop"] >= 800 else "weak"
    return f"OK ({strength})" + ("; " + "; ".join(notes) if notes else ""), True


def compute_gains(drops):
    good = [d for d in drops if d is not None and d >= MIN_DROP]
    if not good:
        return [None] * 6, None
    ref = statistics.median(good)
    return [None if (d is None or d < MIN_DROP) else round(max(0.25, min(4.0, ref / d)), 3)
            for d in drops], ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        sys.exit(f"Could not open {args.port}: {e}\nClose the Serial Monitor and try again.")
    print("Connecting (Bluetooth can take up to 20 s)...", flush=True)
    if not wait_for_data(ser):
        ser.close()
        sys.exit("No data from the ESP32. Check the port, that the sketch is running and it is powered.")

    print("\nStep 1: insole flat on a hard table, NOTHING on it.")
    input("Press Enter, then keep hands off for 3 s...")
    base_rows = capture(ser, 3.0)
    if len(base_rows) < 50:
        ser.close()
        sys.exit("Too little data. Check the connection and run again.")
    base = [statistics.median(col(base_rows, i)) for i in range(6)]
    std = [statistics.pstdev(col(base_rows, i)) for i in range(6)]
    for i in range(6):
        flag = "   <-- NOISY: loose or floating wire" if std[i] > NOISY_STD else ""
        low = "   <-- LOW: shorted, wrong pin, or sensor already pressed" if base[i] < 3000 else ""
        print(f"  {CHANNELS[i]:7s} unpressed {base[i]:.0f}  (noise {std[i]:.0f}){flag}{low}")

    print("\nStep 2: put the SAME standard load on each sensor in turn (centre of the round pad).")
    results = [None] * 6
    for ch in range(6):
        while True:
            input(f"\n[{ch + 1}/6] {LABELS[ch]}: get the load ready, press Enter, then place it and HOLD still for 4 s...")
            print("      Hold...", flush=True)
            press = capture(ser, 4.0)
            if len(press) < 100:
                print("      Too little data, try again.")
                continue
            m = measure(base, press, ch)
            text, ok = verdict(m)
            print(f"      unpressed {base[ch]:.0f}, loaded {m['level']:.0f}, drop {m['drop']:.0f}  ->  {text}")
            if ok:
                results[ch] = m
                break
            if input("      Type r to repeat this sensor, or just Enter to skip it: ").strip().lower() != "r":
                results[ch] = m
                break
    ser.close()

    drops = [None if r is None else r["drop"] for r in results]
    gains, ref = compute_gains(drops)
    print("\n--- Calibration summary ---")
    print(f"{'Sensor':8s} {'unpressed':>9s} {'loaded':>7s} {'drop':>6s} {'gain':>6s}  note")
    for i in range(6):
        r = results[i]
        g = "-" if gains[i] is None else f"{gains[i]:.2f}"
        note = "" if r is None else verdict(r)[0]
        print(f"{CHANNELS[i]:8s} {base[i]:9.0f} {r['level']:7.0f} {r['drop']:6.0f} {g:>6s}  {note}")
    if ref:
        print(f"\nReference drop (median of working sensors): {ref:.0f}. gain = reference / own drop.")
        print("Normalised signal = (unpressed - reading) / own drop, so every sensor reads about 1.0 at the standard load.")
    bad = [CHANNELS[i] for i in range(6) if gains[i] is None]
    if bad:
        print("Needs attention before recording:", ", ".join(bad))
    sat = [CHANNELS[i] for i in range(6) if results[i] and results[i]["saturated"]]
    if sat:
        print("Bottomed out (gain unreliable, repeat with a lighter load):", ", ".join(sat))

    with open(args.out, "w") as f:
        json.dump(dict(
            date=datetime.datetime.now().isoformat(timespec="seconds"),
            channels=CHANNELS, baseline=base, noise=std,
            loaded=[None if r is None else r["level"] for r in results],
            drop=drops, gain=gains, reference_drop=ref,
            saturated=[None if r is None else r["saturated"] for r in results],
        ), f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
