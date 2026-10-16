#!/usr/bin/env python3
"""
Normalise recordings with calibration.json so every pad reads 1.0 at the standard (bottle) load.

  py -3.13 normalize.py recordings\\abhi_normal_20261014_190000.csv
  py -3.13 normalize.py recordings\\*.csv --cal calibration.json

Writes <name>_norm.csv next to each recording (same columns, values in "standard loads":
0 = unloaded, 1 = the calibration load, 2 = twice that, and so on) and prints a before/after table.

Method: an FSR in this divider has conductance proportional to force, and
  conductance (relative) = (4095 - reading) / reading     (the resistor cancels within one pad)
so for each pad  normalised = (G(reading) - G(unloaded)) / (G(loaded in calibration) - G(unloaded in calibration)).
This is more accurate than scaling the raw drop, because the raw reading is strongly nonlinear in force.
Limits: one calibration load, so it is exact at that load and approximate elsewhere. Readings under 30
are treated as 30 (the ESP32 ADC is unreliable near zero), so very hard presses are capped.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

CHANNELS = ["L_toe", "L_mid", "L_heel", "R_toe", "R_mid", "R_heel"]
FLOOR = 150          # the ESP32 ADC is unreliable near zero, so readings below this count as this (pad saturated)


def cond(x):
    x = np.clip(np.asarray(x, dtype=float), FLOOR, 4095)
    return (4095.0 - x) / x


def pad_units(cal):
    """Conductance of one standard load for each pad (None if the pad has no usable calibration)."""
    units = []
    for b, l, g in zip(cal["baseline"], cal["loaded"], cal.get("gain", [1] * 6)):
        if b is None or l is None or g is None:
            units.append(None)
            continue
        u = float(cond(l) - cond(b))
        units.append(u if u > 0.02 else None)
    return units


def normalise(raw, units, baseline="recording"):
    """raw: (N, 6) ADC counts. Returns (N, 6) in standard loads, plus a list of pads that could not be normalised."""
    base = np.percentile(raw, 99, axis=0)
    out = np.zeros_like(raw, dtype=float)
    bad = []
    for i in range(6):
        if units[i] is None:
            bad.append(CHANNELS[i])
            continue
        out[:, i] = np.clip(cond(raw[:, i]) - cond(base[i]), 0, None) / units[i]
    return out, bad


def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if len(r) == 8:
                try:
                    rows.append([int(v) for v in r])
                except ValueError:
                    pass
    if len(rows) < 100:
        raise ValueError("too few valid rows")
    a = np.array(rows)
    return a[:, :2], a[:, 2:].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--cal", default="calibration.json")
    args = ap.parse_args()
    try:
        cal = json.load(open(args.cal))
    except (OSError, ValueError) as e:
        sys.exit(f"Cannot read {args.cal}: {e}. Run calibrate.py first.")
    units = pad_units(cal)
    paths = [p for pat in args.files for p in (glob.glob(pat) or [pat]) if not p.endswith("_norm.csv")]
    for path in paths:
        try:
            head, raw = read_csv(path)
        except (OSError, ValueError) as e:
            print(f"skip {path}: {e}")
            continue
        norm, bad = normalise(raw, units)
        base = np.percentile(raw, 99, axis=0)
        raw_drop = (base - raw) / 4095.0
        out = os.path.splitext(path)[0] + "_norm.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["n", "t_ms"] + CHANNELS)
            for h, row in zip(head, norm):
                w.writerow([int(h[0]), int(h[1])] + [f"{v:.3f}" for v in row])
        print(f"\n{os.path.basename(path)}  ->  {os.path.basename(out)}")
        print(f"{'pad':7s} {'raw peak':>9s} {'normalised peak (loads)':>25s} {'clipped %':>10s}")
        for i, ch in enumerate(CHANNELS):
            rp = np.percentile(raw_drop[:, i], 95)
            npk = np.percentile(norm[:, i], 95)
            clip = 100.0 * float(np.mean(raw[:, i] <= FLOOR))
            val = f"{npk:.1f}" if clip <= 5 else f">= {npk:.1f}"
            note = "   no usable calibration" if ch in bad else ("   saturated: lower bound only" if clip > 5 else "")
            print(f"{ch:7s} {rp:9.2f} {val:>25s} {clip:10.1f}{note}")


if __name__ == "__main__":
    main()
