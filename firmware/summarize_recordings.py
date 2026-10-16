#!/usr/bin/env python3
"""
Summarise the recordings made with record.py, for the report (Section 3).

  python summarize_recordings.py recordings

Files are named  <subject>_<condition>_<date>_<time>.csv  (record.py does this).
Prints a table per condition and the sentence to paste into the report.
"""
import csv
import glob
import os
import sys
from collections import defaultdict


def read(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r, None)
        rows = [row for row in r if len(row) == 8]
    if len(rows) < 2:
        return None
    n = [int(x[0]) for x in rows]
    t = [int(x[1]) for x in rows]
    dur = (t[-1] - t[0]) / 1000
    dropped = sum(max(b - a - 1, 0) for a, b in zip(n, n[1:]))
    return dict(samples=len(rows), seconds=dur, dropped=dropped,
                rate=(len(rows) - 1) / dur if dur > 0 else 0)


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "recordings"
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        sys.exit(f"No CSV files in {folder}")
    by_cond = defaultdict(lambda: dict(walks=0, seconds=0.0, subjects=set()))
    subjects, total_s, total_walks, problems = set(), 0.0, 0, []
    for p in files:
        parts = os.path.basename(p)[:-4].split("_")
        if len(parts) < 4:
            problems.append(f"{os.path.basename(p)}: name does not look like subject_condition_date_time.csv")
            continue
        subject, cond = parts[0], "_".join(parts[1:-2])
        info = read(p)
        if info is None:
            problems.append(f"{os.path.basename(p)}: empty or too short")
            continue
        if info["dropped"] > 0 or not 90 <= info["rate"] <= 110:
            problems.append(f"{os.path.basename(p)}: {info['rate']:.0f} Hz, {info['dropped']} dropped samples")
        c = by_cond[cond]
        c["walks"] += 1
        c["seconds"] += info["seconds"]
        c["subjects"].add(subject)
        subjects.add(subject)
        total_s += info["seconds"]
        total_walks += 1

    print(f"{'Condition':12s} {'Volunteers':>10s} {'Walks':>6s} {'Seconds':>8s}")
    for cond in sorted(by_cond):
        c = by_cond[cond]
        print(f"{cond:12s} {len(c['subjects']):10d} {c['walks']:6d} {c['seconds']:8.0f}")
    print(f"{'Total':12s} {len(subjects):10d} {total_walks:6d} {total_s:8.0f}")
    if problems:
        print("\nCheck these files:")
        for pr in problems:
            print("  ", pr)
    conds = ", ".join(sorted(by_cond))
    print("\nFor Section 3 of the report:")
    print(f"  {len(subjects)} healthy volunteer(s) walked {total_walks} times in the conditions {conds}, "
          f"giving {total_s:.0f} seconds ({total_s / 60:.1f} minutes) of recording at 100 Hz.")


if __name__ == "__main__":
    main()
