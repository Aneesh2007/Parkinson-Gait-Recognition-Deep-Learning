"""
Fill the database with SIMULATED demo patients so the dashboard has something to show.

Every demo recording is generated here (not measured) and goes through the same
metrics code as a real upload. Demo patients are marked 'Demo' and are not real people.

  python seed_demo.py            # add demo patients if none exist
  python seed_demo.py --reset    # delete everything and re-seed
  python seed_demo.py --reset --ai-preview   # also add illustrative AI scores (flagged as preview)
  python seed_demo.py --sample-csv sample.csv   # write one recording in recorder format
"""
import argparse
import datetime
import os

import numpy as np

import app as backend
import gait_metrics as gm

gm.CAL_PATH = ""   # demo data is simulated: never equalise it with a real insole's calibration


def simulate(rng, seconds=60, fs=100, stride_s=1.1, cv=0.03,
             stance_l=0.60, stance_r=0.60, amp=1.0, flat_l=0.0, flat_r=0.0,
             amp_l=None, amp_r=None, toe_l=1.0, toe_r=1.0):
    """Return rows (n, t_ms, 6 raw ADC channels) of a simulated walk.
    amp_l / amp_r: loading of each foot (default amp). toe_l / toe_r: toe push-off strength (1 = normal).
    flat_l / flat_r (0 to 1): 0 = normal heel-to-toe roll-over, 1 = flat-footed, every zone
    loads at the same time (shuffling gait)."""
    n = int(seconds * fs)
    t = np.arange(n) / fs
    raw = np.full((n, 6), 4095.0)

    def one_foot(first_hs, stance_frac, cols, flat, famp, ftoe):
        centres = {"toe": 0.78 - 0.28 * flat, "mid": 0.5, "heel": 0.22 + 0.28 * flat}
        width = 0.22 + 0.12 * flat
        hs = first_hs
        while hs < seconds:
            d = stride_s * (1 + cv * rng.standard_normal())
            d = max(d, 0.6)
            stance = stance_frac * d
            m = (t >= hs) & (t < hs + stance)
            x = (t[m] - hs) / stance
            for col, pos in zip(cols, ("toe", "mid", "heel")):
                bump = np.exp(-((x - centres[pos]) / width) ** 2)
                raw[m, col] -= 3000 * famp * (ftoe if pos == "toe" else 1.0) * bump * rng.uniform(0.9, 1.1)
            hs += d

    al = amp if amp_l is None else amp_l
    ar = amp if amp_r is None else amp_r
    one_foot(0.3, stance_l, (0, 1, 2), flat_l, al, toe_l)
    one_foot(0.3 + stride_s / 2, stance_r, (3, 4, 5), flat_r, ar, toe_r)
    raw += rng.normal(0, 12, raw.shape)
    raw = np.clip(raw, 0, 4095).astype(int)
    idx = np.arange(n)
    return np.column_stack([idx, (t * 1000).astype(int), raw])


def write_csv(rows, path):
    with open(path, "w") as f:
        f.write("n,t_ms," + ",".join(gm.CHANNELS) + "\n")
        for r in rows:
            f.write(",".join(str(int(v)) for v in r) + "\n")


# Simulated caseload. Each patient: name, condition, start and end parameters (they move
# linearly over the sessions), number of weekly sessions, days since the last session,
# and an optional note. None of this is real patient data.
def P(stride, cv, sl, sr, amp=1.0, fl=0.0, fr=0.0, al=None, ar=None, tl=1.0, tr=1.0):
    return dict(stride_s=stride, cv=cv, stance_l=sl, stance_r=sr, amp=amp, flat_l=fl, flat_r=fr,
                amp_l=amp if al is None else al, amp_r=amp if ar is None else ar, toe_l=tl, toe_r=tr)


DEMO = [
    ("Demo patient A", "Simulated Parkinson's-like gait: slow and irregular, improving",
     P(1.30, .090, .62, .61, .75, .6, .6), P(1.20, .055, .61, .60, .85, .4, .4), 8, 0,
     "Started cueing exercises. Variability falling over the weeks."),
    ("Demo patient B", "Simulated stroke-like gait: uneven stance, improving with rehab",
     P(1.25, .050, .68, .52, .90, .5, .1, .62, 1.0, .45, 1.0), P(1.12, .030, .62, .58, 1.0, .25, .05, .90, 1.0, .85, 1.0), 8, 1,
     "Left side weaker after stroke. Physiotherapy twice a week."),
    ("Demo patient C", "Simulated healthy walking, for comparison",
     P(1.05, .025, .60, .60), P(1.05, .025, .60, .60), 6, 2, ""),
    ("Demo patient D", "Simulated Parkinson's-like gait: getting worse",
     P(1.15, .040, .61, .60, .9, .3, .3), P(1.38, .105, .63, .60, .65, .8, .8, tl=.6, tr=.6), 8, 3,
     "Family reports more shuffling at home. Review medication timing."),
    ("Demo patient E", "Simulated stroke-like gait: asymmetry not improving",
     P(1.22, .045, .69, .53, .9, .5, .1, .60, 1.0, .5, 1.0), P(1.22, .050, .70, .52, .9, .5, .1, .58, 1.0, .5, 1.0), 7, 5, ""),
    ("Demo patient F", "Simulated frail older adult: slow walking, fall risk",
     P(1.55, .060, .62, .60, .8, .5, .5), P(1.58, .068, .62, .60, .8, .5, .5), 5, 9,
     "Slow cadence. Balance programme suggested."),
    ("Demo patient G", "Simulated knee surgery recovery: limp easing",
     P(1.20, .040, .72, .54, .95, .1, .0, .55, 1.0, .6, 1.0), P(1.08, .028, .62, .58, 1.0, .0, .0, .95, 1.0, .95, 1.0), 8, 0,
     "Week 8 after surgery. Weight is going through the operated leg again."),
    ("Demo patient H", "Simulated healthy walking, for comparison",
     P(1.02, .022, .59, .60), P(1.02, .022, .59, .60), 5, 4, ""),
    ("Demo patient I", "Simulated mild Parkinson's-like gait: stable",
     P(1.18, .058, .61, .60, .85, .3, .3), P(1.18, .056, .61, .60, .85, .3, .3), 6, 6, ""),
    ("Demo patient J", "Simulated stroke-like gait: slow recovery",
     P(1.28, .055, .66, .54, .9, .4, .1), P(1.18, .042, .63, .56, .95, .3, .1), 8, 2, ""),
    ("Demo patient K", "Simulated hip replacement follow-up: asymmetry rising",
     P(1.10, .030, .60, .58, 1.0, .0, .0), P(1.16, .036, .68, .55, .95, .0, .1, 1.0, .65, 1.0, .55), 6, 20,
     "No session for 3 weeks. Contact patient."),
    ("Demo patient L", "Simulated healthy walking, for comparison",
     P(1.08, .026, .60, .61), P(1.08, .026, .60, .61), 4, 1, ""),
]


# ILLUSTRATIVE AI screening scores for the demo patients (start, end over their sessions).
# Used only with --ai-preview so the dashboard can show how the AI card looks before the real
# model file is installed. Every such value is flagged preview=True and the dashboard says so.
# Only Parkinson's-like and healthy demo patients get one: the model only knows Parkinson's vs healthy.
AI_PREVIEW = {
    "Demo patient A": (0.78, 0.58), "Demo patient C": (0.12, 0.10), "Demo patient D": (0.55, 0.84),
    "Demo patient H": (0.15, 0.14), "Demo patient I": (0.62, 0.61), "Demo patient L": (0.18, 0.16),
}


def preview_ai(rng, name, f):
    if name not in AI_PREVIEW:
        return None
    a, b = AI_PREVIEW[name]
    score = min(max(a + (b - a) * f + rng.normal(0, 0.02), 0.02), 0.98)
    n = 54
    return {"ok": True, "score": round(float(score), 3), "band": "low" if score < 0.35 else "high" if score > 0.65 else "uncertain",
            "windows": n, "windows_pd": int(round(score * n)), "model": "CNN (3 sensors per foot), PhysioNet gaitpdb",
            "preview": True}


def lerp(a, b, f):
    return {k: a[k] + (b[k] - a[k]) * f for k in a}


def seed(reset=False, ai_preview=False):
    rng = np.random.default_rng(7)
    with backend.db() as c:
        if reset:
            c.execute("DELETE FROM notes")
            c.execute("DELETE FROM sessions")
            c.execute("DELETE FROM patients")
        elif c.execute("SELECT 1 FROM patients WHERE demo=1").fetchone():
            print("Demo patients already exist. Use --reset to rebuild.")
            return
        today = datetime.datetime.now().replace(microsecond=0)
        for name, cond, p0, p1, n_sessions, last_days, note in DEMO:
            pid = backend.create_patient(c, name, cond, demo=1)
            for k in range(n_sessions):
                f = k / (n_sessions - 1) if n_sessions > 1 else 0
                rows = simulate(rng, **lerp(p0, p1, f))
                m = gm.compute_metrics(rows[:, 1] / 1000.0, rows[:, 2:8].astype(float))
                if not m["ok"]:
                    raise RuntimeError(f"Demo recording failed for {name}: {m['reason']}")
                when = today - datetime.timedelta(days=last_days + 7 * (n_sessions - 1 - k), hours=k % 5)
                label = "clinic walk" if k % 2 == 0 else "home walk"
                backend.save_session(c, pid, when.isoformat(), label, m,
                                     ai=preview_ai(rng, name, f) if ai_preview else None)
            if note:
                c.execute("INSERT INTO notes(patient_id, created, author, text) VALUES (?,?,?,?)",
                          (pid, (today - datetime.timedelta(days=last_days + 1)).isoformat(),
                           "Demo clinician", note + " (simulated note)"))
            print(f"Added {name} with {n_sessions} sessions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--ai-preview", action="store_true",
                    help="add ILLUSTRATIVE AI screening scores (flagged as preview) to some demo patients")
    ap.add_argument("--sample-csv", help="write one simulated recording in recorder format")
    a = ap.parse_args()
    if a.sample_csv:
        write_csv(simulate(np.random.default_rng(3), seconds=40), a.sample_csv)
        print("Wrote", a.sample_csv)
    else:
        seed(reset=a.reset, ai_preview=a.ai_preview)
