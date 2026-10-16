#!/usr/bin/env python3
"""
Record the insole stream from the ESP32 (insole_logger.ino) into a CSV and print a
quality report.

Install once:   pip install pyserial
Examples:
  python record.py --list-ports
  python record.py --port COM5 --subject abhi --condition normal --seconds 60
  python record.py --port COM5 --subject abhi --condition limp --seconds 30

Close the Arduino Serial Monitor and Serial Plotter first: only one program can
use the port at a time.
"""
import argparse
import datetime
import os
import sys
import time

import serial
from serial.tools import list_ports

CHANNELS = ["L_toe", "L_mid", "L_heel", "R_toe", "R_mid", "R_heel"]
N_COLS = 2 + len(CHANNELS)  # n, t_ms, then the channels


def parse_line(raw):
    """Return a list of ints, or None if the line is not a valid data row."""
    try:
        parts = raw.decode("ascii", errors="ignore").strip().split(",")
        if len(parts) != N_COLS:
            return None
        return [int(p) for p in parts]
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-ports", action="store_true")
    ap.add_argument("--port", help="e.g. COM5 (Windows) or /dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--subject", default="subject")
    ap.add_argument("--condition", default="normal",
                    help="normal, slow, fast, limp, shuffle, or your own label")
    ap.add_argument("--out", default="recordings")
    ap.add_argument("--countdown", type=int, default=3)
    args = ap.parse_args()

    if args.list_ports:
        for p in list_ports.comports():
            print(f"{p.device}  {p.description}")
        return
    if not args.port:
        sys.exit("Give --port (use --list-ports to see the options)")

    os.makedirs(args.out, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.out, f"{args.subject}_{args.condition}_{stamp}.csv")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        sys.exit(f"Could not open {args.port}: {e}\nClose the Serial Monitor and try again.")
    time.sleep(2.0)            # opening the port resets the ESP32; let it restart
    ser.reset_input_buffer()

    for k in range(args.countdown, 0, -1):
        print(f"Starting in {k}...", flush=True)
        time.sleep(1)
    print(f"Recording {args.seconds:.0f} s to {path}  (Ctrl+C to stop early)", flush=True)

    rows = []
    malformed = 0
    t_first = None
    wall_start = time.time()
    last_print = wall_start
    ser.readline()             # discard a possibly partial first line
    try:
        while True:
            raw = ser.readline()
            if not raw:
                if time.time() - wall_start > args.seconds + 10:
                    print("No data from the ESP32. Check the port and the sketch.")
                    break
                continue
            row = parse_line(raw)
            if row is None:
                if not raw.startswith(b"n,"):   # ignore the header line
                    malformed += 1
                continue
            if t_first is None:
                t_first = row[1]
            rows.append(row)
            if row[1] - t_first >= args.seconds * 1000:
                break
            if time.time() - last_print >= 1.0:
                last_print = time.time()
                print(f"  {(row[1] - t_first) / 1000:5.1f} s  {len(rows)} samples", flush=True)
    except KeyboardInterrupt:
        print("\nStopped early.")
    finally:
        ser.close()

    if len(rows) < 2:
        sys.exit("Not enough data recorded, nothing saved.")

    with open(path, "w", newline="") as f:
        f.write("n,t_ms," + ",".join(CHANNELS) + "\n")
        for r in rows:
            f.write(",".join(str(v) for v in r) + "\n")

    # ---- quality report ----
    duration = (rows[-1][1] - rows[0][1]) / 1000
    gaps_ms = [b[1] - a[1] for a, b in zip(rows, rows[1:])]
    dropped = sum(max(b[0] - a[0] - 1, 0) for a, b in zip(rows, rows[1:]))
    rate = (len(rows) - 1) / duration if duration > 0 else 0
    print("\n--- Recording report ---")
    print(f"File:            {path}")
    print(f"Samples:         {len(rows)} over {duration:.1f} s  ->  {rate:.1f} Hz (target 100)")
    print(f"Dropped samples: {dropped}   Malformed lines: {malformed}")
    print(f"Largest gap:     {max(gaps_ms)} ms (normal is 10 ms)")
    if dropped or max(gaps_ms) > 30:
        print("WARNING: gaps found. Try a shorter USB cable, another port, or a higher baud rate.")
    print("Channel   min   max  mean")
    for i, name in enumerate(CHANNELS):
        col = [r[2 + i] for r in rows]
        mean = sum(col) / len(col)
        note = ""
        if max(col) - min(col) < 100:
            note = "  <- barely moved: not pressed, or sensor/wiring problem"
        print(f"{name:8s} {min(col):5d} {max(col):5d} {mean:6.0f}{note}")


if __name__ == "__main__":
    main()
