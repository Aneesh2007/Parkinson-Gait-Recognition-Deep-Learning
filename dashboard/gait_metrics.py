"""
Gait metrics from raw insole recordings (6 FSR channels, ADC counts).

Input rows: n, t_ms, L_toe, L_mid, L_heel, R_toe, R_mid, R_heel  (see record.py)
Pipeline: per-channel normalisation -> foot pressure signal -> stance detection ->
stride time, variability, stance asymmetry, cadence.
"""
import json
import os

import numpy as np

CHANNELS = ["L_toe", "L_mid", "L_heel", "R_toe", "R_mid", "R_heel"]

# Illustrative alert thresholds for the demo. They are NOT clinical values.
THRESHOLDS = {
    "asym_pct": 10.0,      # stance-time asymmetry between feet
    "cv_pct": 6.0,         # stride-time variability (coefficient of variation)
    "cadence_min": 80.0,   # steps per minute
    "cadence_max": 140.0,
}
# Wider limits for stepping in place: the signals are less regular than in walking. Also illustrative.
THRESHOLDS_INPLACE = {"asym_pct": 15.0, "cv_pct": 30.0, "cadence_min": 20.0, "cadence_max": 140.0}
MODES = ("walk", "inplace", "auto")


def parse_csv(text):
    """Parse recorder CSV text into (time in seconds, raw array of shape N x 6)."""
    rows = []
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) != 8:
            continue
        try:
            rows.append([int(p) for p in parts])
        except ValueError:
            continue  # header line or garbage
    if len(rows) < 200:
        raise ValueError("Need at least 2 seconds of valid rows (n,t_ms and 6 channels)")
    a = np.array(rows, dtype=float)
    return a[:, 1] / 1000.0, a[:, 2:8]


def normalise(raw):
    """Per-channel pressure in 0..1. A channel that barely moves stays at 0."""
    p = np.zeros_like(raw)
    for i in range(raw.shape[1]):
        col = raw[:, i]
        hi, lo = np.percentile(col, 95), np.percentile(col, 1)
        if hi - lo >= 150:
            p[:, i] = np.clip((hi - col) / (hi - lo), 0, 1)
    return p


def foot_signal(p, cols):
    active = [c for c in cols if p[:, c].max() > 0]
    if not active:
        return np.zeros(len(p))
    s = p[:, active].mean(axis=1)
    return np.convolve(s, np.ones(5) / 5, mode="same")


def contacts(s, fs, min_stance=0.15, min_swing=0.10):
    """Return a list of (heel_strike_s, toe_off_s) for complete stance phases."""
    ref = np.percentile(s, 95)
    if ref < 0.05:
        return []
    on_thr, off_thr = 0.25 * ref, 0.15 * ref
    stance, start, ints = False, 0, []
    for i, v in enumerate(s):
        if not stance and v > on_thr:
            stance, start = True, i
        elif stance and v < off_thr:
            stance = False
            ints.append([start, i])
    # merge stances separated by a very short swing, then drop very short stances
    merged = []
    for iv in ints:
        if merged and (iv[0] - merged[-1][1]) / fs < min_swing:
            merged[-1][1] = iv[1]
        else:
            merged.append(iv)
    return [(a / fs, b / fs) for a, b in merged if (b - a) / fs >= min_stance]


HERE = os.path.dirname(os.path.abspath(__file__))
CAL_PATH = os.environ.get("INSOLE_CAL", os.path.join(HERE, "calibration.json"))
LOAD_FULL = 50.0     # calibrated signal is drawn on a log scale: 1 standard (bottle) load = 0.18, 10 = 0.61, 50 or more = 1.0
_FLOOR = 150         # ESP32 ADC is unreliable near zero, so readings below this count as this (pad saturated)
_cal = {"mtime": None, "path": None, "units": None}
GAIN_PATH = os.path.join(HERE, "pad_gain.json")


def pad_gain():
    """Fixed per-pad gain that evens out the left and right mid-foot pads (hardware difference shared by all users)."""
    try:
        with open(GAIN_PATH) as f:
            g = [float(x) for x in json.load(f)["gain"]]
        return g if len(g) == 6 else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _cond(x):
    """Relative conductance of an FSR from its divider reading (proportional to force; the resistor cancels per pad)."""
    x = np.clip(np.asarray(x, dtype=float), _FLOOR, 4095)
    return (4095.0 - x) / x


def calibration_units():
    """Conductance of one standard load for each pad (None entries = no usable calibration), or None if no file.
    Reads calibration.json from this folder (or INSOLE_CAL); reloads when the file changes."""
    path = CAL_PATH
    if not path or not os.path.exists(path):
        _cal.update(mtime=None, path=path, units=None)
        return None
    mtime = os.path.getmtime(path)
    if _cal["path"] == path and _cal["mtime"] == mtime:
        return _cal["units"]
    units = None
    try:
        with open(path) as f:
            c = json.load(f)
        units = []
        for b, l in zip(c["baseline"], c["loaded"]):
            u = None if (b is None or l is None) else float(_cond(l) - _cond(b))
            units.append(u if (u is not None and u > 0.02) else None)
        if len(units) != 6:
            units = None
    except (OSError, ValueError, KeyError, TypeError):
        units = None
    _cal.update(mtime=mtime, path=path, units=units)
    return units


def calibration_info():
    u = calibration_units()
    if u is None:
        return {"calibrated": False}
    return {"calibrated": True, "pads_calibrated": sum(x is not None for x in u),
            "pads": {n: (x is not None) for n, x in zip(CHANNELS, u)}}


def unloaded_drop(raw):
    """Signal of each sensor below its unloaded level, drawn on a 0..1 scale.
    The unloaded level is the 99th percentile of the recording.
    With calibration.json present, each pad is equalised so the standard calibration load gives the same value on every pad
    (via conductance, which is proportional to force), drawn on a log scale; without it this is the raw drop as a fraction of the 12-bit
    full scale, and sensors still differ in sensitivity."""
    base = np.percentile(raw, 99, axis=0)
    drop = np.clip((base - raw) / 4095.0, 0, 1)
    units = calibration_units()
    if units is None:
        g = pad_gain()
        if g:
            drop = np.clip(drop * np.array(g), 0, 1)
        return drop
    for i, u in enumerate(units):
        if u is not None:
            loads = np.clip(_cond(raw[:, i]) - _cond(base[i]), 0, None) / u
            drop[:, i] = np.clip(np.log1p(loads) / np.log1p(LOAD_FULL), 0, 1)
    return drop


def zone_summary(drop, clip=None):
    """Peak loading per zone (95th percentile of the drop) and mean drop, per foot."""
    out = {}
    for foot, cols in (("L", [0, 1, 2]), ("R", [3, 4, 5])):                   # toe, mid, heel
        out[foot] = {
            z: {"peak": round(float(np.percentile(drop[:, c], 95)), 3), "mean": round(float(drop[:, c].mean()), 3),
                "clip": round(float(clip[c]), 3) if clip is not None else 0.0}
            for z, c in zip(("toe", "mid", "heel"), cols)
        }
    return out


def _quality(raw, seg):
    """Per-pad quality over the part of the recording that counts, plus calibrated loads when calibration.json exists.
    A pad that sits at the ADC floor for more than 5% of the time is saturated: its load is only a lower bound."""
    r = raw[seg]
    base = np.percentile(raw, 99, axis=0)
    units = calibration_units()
    pads, loads_peak, loads_mean, lower = {}, {}, {}, []
    for i, name in enumerate(CHANNELS):
        sat = float(np.mean(r[:, i] <= _FLOOR) * 100)
        rng = float(np.percentile(base[i] - r[:, i], 95) - np.percentile(base[i] - r[:, i], 5))
        status = "saturated" if sat > 5 else "flat" if rng < 150 else "weak" if rng < 400 else "ok"
        pads[name] = {"sat": round(sat, 1), "range": int(rng), "status": status, "calibrated": bool(units and units[i] is not None)}
        if units and units[i] is not None:
            ld = np.clip(_cond(r[:, i]) - _cond(base[i]), 0, None) / units[i]
            loads_peak[name] = round(float(np.percentile(ld, 95)), 2)
            loads_mean[name] = float(np.mean(np.minimum(ld, LOAD_FULL)))
            if sat > 5:
                lower.append(name)
    out = {"pads": pads}
    if len(loads_peak) == 6:
        out["loads"] = loads_peak
        out["lower_bound"] = lower
        l, rr = sum(loads_mean[n] for n in CHANNELS[:3]), sum(loads_mean[n] for n in CHANNELS[3:])
        out["load_share_l"] = round(100 * l / (l + rr), 1) if l + rr > 0 else None
        # experimental mid-foot load index per foot: mid / (toe + mid + heel), from calibrated mean loads.
        # Only trusted when no pad of that foot bottoms out (a capped heel would inflate it).
        arch = {}
        for foot, names in (("L", CHANNELS[:3]), ("R", CHANNELS[3:])):
            tot = sum(loads_mean[n] for n in names)
            ok = not any(n in lower for n in names)
            arch[foot] = {"index": round(loads_mean[names[1]] / tot, 2) if tot > 0 else None, "reliable": bool(ok)}
        out["arch"] = arch
    return out


def _trace_points(n_rows, fs0, max_pts=3000):
    """Row indices for the stored trace: about 25 per second, thinned for very long recordings."""
    step = max(int(round(fs0 / 25)), 1)
    while n_rows / step > max_pts:
        step += 1
    return step


def compute_inplace(t, raw):
    """Stepping in place (marching on the spot). The feet take turns carrying the weight, so the signal of one
    foot minus the other (each scaled to its own range) swings positive and negative. A switch of sign with some
    hysteresis is one step; that works even when the pads of the lifted foot never fully unload.
    Same output fields as compute_metrics."""
    t = t - t[0]
    duration = float(t[-1])
    if duration < 5:
        return {"ok": False, "reason": "Recording shorter than 5 s"}
    fs0 = (len(t) - 1) / duration
    ds = max(int(round(fs0 / 25)), 1)
    n = len(raw) // ds
    D = np.clip((4095.0 - raw[: n * ds]) / 4095.0, 0, 1).reshape(n, ds, 6).mean(1)
    tt = t[: n * ds].reshape(n, ds).mean(1)
    fs = fs0 / ds
    tot = np.convolve(D.sum(1), np.ones(5) / 5, mode="same")
    w = max(int(fs), 2)                           # stepping = the total load keeps changing; standing still (even preloaded) = flat
    move = np.convolve(np.abs(np.diff(tot, prepend=tot[0])), np.ones(w), mode="same")
    act = np.where((move > 0.3) & (tot > 0.15))[0]
    if len(act) < 4 * fs:
        return {"ok": False, "reason": "No stepping found. The insoles show almost no load. Check the sensors and the recording."}
    i0, i1 = int(act[0]), int(act[-1])
    if (i1 - i0) / fs < 5:
        return {"ok": False, "reason": "Less than 5 s of stepping found in the recording."}
    seg = slice(i0, i1 + 1)

    sig, wts = {}, {}
    for name, cols in (("L", [0, 1, 2]), ("R", [3, 4, 5])):
        refs = {c: float(np.percentile(D[seg, c], 95)) for c in cols}
        top = max(refs.values())
        # drop dead pads (never pressed) and stuck pads (pressed all the time): neither follows the stepping.
        # Weighting or dropping merely weak pads was tried and made step counts less stable.
        used = [c for c in cols if refs[c] >= 0.12 and (np.percentile(D[seg, c], 95) - np.percentile(D[seg, c], 5)) >= 0.10]
        if not used:
            return {"ok": False, "reason": f"No {'left' if name == 'L' else 'right'} foot sensor responded."}
        wt = np.ones(len(used)) / len(used)
        x = np.sum([wt[k] * np.clip(D[:, c] / refs[c], 0, 1) for k, c in enumerate(used)], axis=0)
        x = np.convolve(x, np.ones(3) / 3, mode="same")
        lo, hi = np.percentile(x[seg], 5), np.percentile(x[seg], 95)
        sig[name] = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
        wts[name] = {["toe", "mid", "heel"][c % 3]: round(float(w), 2) for c, w in zip(used, wt)}
    d = sig["L"] - sig["R"]

    def intervals(idx):
        v = np.diff(np.array(idx) / fs)
        return v[(v >= 0.5) & (v <= 8.0)]

    def run(thr):
        # phases: +1 = left foot carries the weight, -1 = right foot, switching at +/- 0.25 (hysteresis)
        min_len = max(int(0.25 * fs), 2)
        raw_segs, state, start = [], 0, i0
        for i in range(i0, i1 + 1):
            v = d[i]
            ns = 1 if v > thr else -1 if v < -thr else state
            if ns != state:
                if state != 0:
                    raw_segs.append([state, start, i])
                state, start = ns, i
        if state != 0:
            raw_segs.append([state, start, i1 + 1])
        segs = []
        for st, a, b in raw_segs:                     # drop blips shorter than 0.25 s, merge neighbours of the same side
            if b - a < min_len:
                continue
            if segs and segs[-1][0] == st:
                segs[-1][2] = b
            else:
                segs.append([st, a, b])
        on = {"L": [a for st, a, b in segs if st == 1], "R": [a for st, a, b in segs if st == -1]}
        dur = {"L": [(b - a) / fs for st, a, b in segs if st == 1], "R": [(b - a) / fs for st, a, b in segs if st == -1]}
        if len(on["L"]) < 4 or len(on["R"]) < 4:
            return {"err": "Too few steps found. Step more clearly, with each foot lifting off the floor in turn."}

        iL, iR = intervals(on["L"]), intervals(on["R"])
        if len(iL) < 3 or len(iR) < 3:
            return {"err": "Step timing is too irregular to measure (steps more than 8 s apart)."}
        cv = float(np.mean([iL.std() / iL.mean(), iR.std() / iR.mean()]) * 100)
        st_l, st_r = float(np.mean(dur["L"])), float(np.mean(dur["R"]))
        asym = abs(st_l - st_r) / (0.5 * (st_l + st_r)) * 100
        nL, nR = len(on["L"]), len(on["R"])
        return {"on": on, "iL": iL, "iR": iR, "cv": cv, "st_l": st_l, "st_r": st_r, "asym": asym, "nL": nL, "nR": nR}

    # the switch threshold is a judgement call, so run it at three values and keep the middle result
    runs = {thr: run(thr) for thr in (0.25, 0.3, 0.35)}
    for r_ in runs.values():
        if "err" in r_:
            return {"ok": False, "reason": r_["err"]}
    rs = list(runs.values())
    mid = sorted(rs, key=lambda r_: r_["asym"])[1]            # the run with the median asymmetry
    spread = {"asym": round(max(r_["asym"] for r_ in rs) - min(r_["asym"] for r_ in rs), 1),
              "cv": round(max(r_["cv"] for r_ in rs) - min(r_["cv"] for r_ in rs), 1),
              "steps": max(r_["nL"] + r_["nR"] for r_ in rs) - min(r_["nL"] + r_["nR"] for r_ in rs)}
    on, iL, iR, cv, st_l, st_r, asym, nL, nR = (mid[k] for k in ("on", "iL", "iR", "cv", "st_l", "st_r", "asym", "nL", "nR"))
    steps = nL + nR
    active_s = (i1 - i0 + 1) / fs
    allon = sorted([(a, "L") for a in on["L"]] + [(a, "R") for a in on["R"]])
    step_t = np.diff([a / fs for a, _ in allon])          # time between one foot's step and the next foot's step
    step_t = step_t[(step_t >= 0.25) & (step_t <= 4.0)]
    step_cv = float(step_t.std() / step_t.mean() * 100) if len(step_t) > 2 else None

    # stored trace: the whole stepping period, both foot signals scaled 0..1, with step markers
    drop = unloaded_drop(raw)
    t_start = 0.0                                          # the trace and playback cover the whole recording, not only the stepping part
    ar0 = int(np.searchsorted(t, tt[i0]))                  # rows of the active (stepping) part, used for the zone summary
    ar1 = int(np.searchsorted(t, tt[i1])) + 1
    r0, r1 = 0, len(t)
    step = _trace_points(r1 - r0, fs0)
    idx = np.arange(r0, r1, step)
    marks = {k: [round(float(tt[a] - t_start), 2) for a in on[k]] for k in on}
    trace = {
        "t": [round(float(t[i] - t_start), 2) for i in idx],
        "L": [round(float(v), 3) for v in np.interp(t[idx], tt, sig["L"])],
        "R": [round(float(v), 3) for v in np.interp(t[idx], tt, sig["R"])],
        "ch": {name: [round(float(drop[i, k]), 3) for i in idx] for k, name in enumerate(CHANNELS)},
        "marks": marks,
        "hz": round(fs0 / step, 2),
    }
    zr = slice(ar0, ar1)
    clip = np.mean(raw[zr] <= _FLOOR, axis=0)
    extra = {"mode": "inplace", "steps_l": nL, "steps_r": nR,
             "step_imbalance": round(abs(nL - nR) / (0.5 * (nL + nR)) * 100, 1),
             "step_time_s": round(float(step_t.mean()), 3) if len(step_t) else None,
             "step_cv": round(step_cv, 1) if step_cv is not None else None,
             "recording_s": round(duration, 1), "spread": spread, "pad_weights": wts,
             "reliable": bool(spread["asym"] <= 8 and spread["cv"] <= 10 and spread["steps"] <= 4)}
    extra.update(_quality(raw, zr))
    return {
        "ok": True, "mode": "inplace",
        "duration_s": round(active_s, 1),
        "steps": int(steps),
        "cadence": round(steps / active_s * 60, 1),
        "stride_time_s": round(float(np.mean(np.concatenate([iL, iR]))), 3),
        "stride_cv": round(cv, 2),
        "stance_l": round(st_l, 3),
        "stance_r": round(st_r, 3),
        "asym": round(float(asym), 2),
        "trace": trace,
        "calibrated": calibration_units() is not None,
        "zones": zone_summary(drop[zr], clip),
        "extra": extra,
    }


def compute_metrics(t, raw, mode="walk"):
    if mode == "inplace":
        return compute_inplace(t, raw)
    if mode == "auto":
        m = compute_metrics(t, raw, "walk")
        if m["ok"]:
            return m
        m2 = compute_inplace(t, raw)
        if m2["ok"]:
            m2["extra"]["auto"] = True
        return m2 if m2["ok"] else m
    t = t - t[0]
    duration = float(t[-1])
    if duration < 2:
        return {"ok": False, "reason": "Recording shorter than 2 s"}
    fs = (len(t) - 1) / duration
    p = normalise(raw)
    drop = unloaded_drop(raw)
    sL, sR = foot_signal(p, [0, 1, 2]), foot_signal(p, [3, 4, 5])
    cL, cR = contacts(sL, fs), contacts(sR, fs)
    if len(cL) < 4 or len(cR) < 4:
        return {"ok": False,
                "reason": "Too few steps detected. Check the sensors are loaded while walking."}

    def strides(c):
        d = np.diff([a for a, _ in c])
        return d[(d > 0.4) & (d < 3.0)]

    stL, stR = strides(cL), strides(cR)
    if len(stL) < 3 or len(stR) < 3:
        return {"ok": False, "reason": "Stride times look irregular, too few valid strides."}

    cv = float(np.mean([stL.std() / stL.mean(), stR.std() / stR.mean()]) * 100)
    stance_l = float(np.mean([b - a for a, b in cL]))
    stance_r = float(np.mean([b - a for a, b in cR]))
    asym = abs(stance_l - stance_r) / (0.5 * (stance_l + stance_r)) * 100
    steps = len(cL) + len(cR)

    # the whole recording at about 25 Hz (thinned if very long), for the dashboard trace and playback
    step = _trace_points(len(t), fs)
    idx = np.arange(0, len(t), step)
    trace = {
        "t": [round(float(t[i]), 2) for i in idx],
        "L": [round(float(sL[i]), 3) for i in idx],
        "R": [round(float(sR[i]), 3) for i in idx],
        # per-sensor signal drop (fraction of full scale), used for the foot map playback
        "ch": {name: [round(float(drop[i, k]), 3) for i in idx] for k, name in enumerate(CHANNELS)},
        "marks": {"L": [round(float(a_), 2) for a_, _ in cL], "R": [round(float(a_), 2) for a_, _ in cR]},
        "hz": round(fs / step, 2),
    }
    return {
        "ok": True, "mode": "walk", "extra": dict({"mode": "walk"}, **_quality(raw, slice(0, len(raw)))),
        "duration_s": round(duration, 1),
        "steps": int(steps),
        "cadence": round(steps / duration * 60, 1),
        "stride_time_s": round(float(np.mean(np.concatenate([stL, stR]))), 3),
        "stride_cv": round(cv, 2),
        "stance_l": round(stance_l, 3),
        "stance_r": round(stance_r, 3),
        "asym": round(float(asym), 2),
        "trace": trace,
        "calibrated": calibration_units() is not None,
        "zones": zone_summary(drop, np.mean(raw <= _FLOOR, axis=0)),
    }


def alert_flags(m):
    T = THRESHOLDS_INPLACE if m.get("mode") == "inplace" else THRESHOLDS
    flags = []
    if m["asym"] > T["asym_pct"]:
        flags.append("Stance asymmetry above %.0f%%" % T["asym_pct"])
    if m["stride_cv"] > T["cv_pct"]:
        flags.append("Stride variability above %.0f%%" % T["cv_pct"])
    if m["cadence"] < T["cadence_min"]:
        flags.append("Cadence below %.0f steps/min" % T["cadence_min"])
    if m["cadence"] > T["cadence_max"]:
        flags.append("Cadence above %.0f steps/min" % T["cadence_max"])
    return flags
