"""
Smart insole cloud backend (demo): FastAPI + SQLite.

Run:   uvicorn app:app --reload
Open:  http://127.0.0.1:8000
Demo data:  python seed_demo.py

There is no login: this is a course demo. Add authentication before using real
patient data anywhere.
"""
import datetime
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

import ai_screen
import gait_metrics as gm

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("INSOLE_DB", os.path.join(HERE, "insole.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  condition TEXT NOT NULL DEFAULT '',
  demo INTEGER NOT NULL DEFAULT 0,
  created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL REFERENCES patients(id),
  recorded_at TEXT NOT NULL,
  label TEXT NOT NULL DEFAULT 'walk',
  duration_s REAL, steps INTEGER, cadence REAL, stride_time_s REAL,
  stride_cv REAL, stance_l REAL, stance_r REAL, asym REAL,
  model_score REAL,
  flags TEXT NOT NULL DEFAULT '[]',
  trace TEXT NOT NULL DEFAULT '{}',
  zones TEXT NOT NULL DEFAULT '{}',
  ai TEXT NOT NULL DEFAULT '{}',
  mode TEXT NOT NULL DEFAULT 'walk',
  extra TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL REFERENCES patients(id),
  created TEXT NOT NULL,
  author TEXT NOT NULL DEFAULT 'Clinician',
  text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_patient ON notes(patient_id, created);
CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id, recorded_at);
"""


@contextmanager
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with db() as c:
        c.executescript(SCHEMA)
        cols = {r["name"] for r in c.execute("PRAGMA table_info(sessions)")}
        if "zones" not in cols:      # database created by an earlier version
            c.execute("ALTER TABLE sessions ADD COLUMN zones TEXT NOT NULL DEFAULT '{}'")
        if "ai" not in cols:         # database created before the AI screening step existed
            c.execute("ALTER TABLE sessions ADD COLUMN ai TEXT NOT NULL DEFAULT '{}'")
        if "mode" not in cols:       # database created before stepping-in-place analysis existed
            c.execute("ALTER TABLE sessions ADD COLUMN mode TEXT NOT NULL DEFAULT 'walk'")
        if "extra" not in cols:
            c.execute("ALTER TABLE sessions ADD COLUMN extra TEXT NOT NULL DEFAULT '{}'")


def now_iso():
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def create_patient(conn, name, condition="", demo=0):
    cur = conn.execute(
        "INSERT INTO patients(name, condition, demo, created) VALUES (?,?,?,?)",
        (name, condition, demo, now_iso()),
    )
    return cur.lastrowid


def save_session(conn, patient_id, recorded_at, label, m, model_score=None, ai=None):
    flags = gm.alert_flags(m)
    if ai and ai.get("ok") and model_score is None:
        model_score = ai["score"]
    cur = conn.execute(
        """INSERT INTO sessions(patient_id, recorded_at, label, duration_s, steps, cadence,
           stride_time_s, stride_cv, stance_l, stance_r, asym, model_score, flags, trace, zones, ai, mode, extra)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (patient_id, recorded_at, label, m["duration_s"], m["steps"], m["cadence"],
         m["stride_time_s"], m["stride_cv"], m["stance_l"], m["stance_r"], m["asym"],
         model_score, json.dumps(flags), json.dumps(m["trace"]), json.dumps(m.get("zones", {})),
         json.dumps(ai or {}), m.get("mode", "walk"), json.dumps(m.get("extra", {}))),
    )
    return cur.lastrowid


def session_row(r, with_trace=False):
    d = {k: r[k] for k in r.keys() if k not in ("trace", "zones", "ai", "extra")}
    d["extra"] = json.loads(r["extra"]) if "extra" in r.keys() and r["extra"] else {}
    d["zones"] = json.loads(r["zones"]) if "zones" in r.keys() else {}
    d["ai"] = json.loads(r["ai"]) if "ai" in r.keys() and r["ai"] else {}
    d["flags"] = json.loads(d["flags"])
    d["alert"] = len(d["flags"]) > 0
    if with_trace:
        d["trace"] = json.loads(r["trace"])
    return d


init_db()
app = FastAPI(title="Smart insole cloud (demo)")


class PatientIn(BaseModel):
    name: str
    condition: str = ""


class NoteIn(BaseModel):
    text: str
    author: str = "Clinician"


def trend_of(sessions):
    """Compare the first and the latest sessions (two each when there are four or more).
    Illustrative rule, not a clinical one: a change of 0.5 points in stride variability or
    1 point in stance asymmetry counts as better or worse."""
    if len(sessions) < 2:
        return {"label": "New", "d_cv": 0.0, "d_asym": 0.0}
    k = 2 if len(sessions) >= 4 else 1
    first, last = sessions[:k], sessions[-k:]
    mean = lambda rows, key: sum(r[key] for r in rows) / len(rows)
    d_cv = mean(last, "stride_cv") - mean(first, "stride_cv")
    d_asym = mean(last, "asym") - mean(first, "asym")
    better = d_cv <= -0.5 or d_asym <= -1.0
    worse = d_cv >= 0.5 or d_asym >= 1.0
    label = "Mixed" if better and worse else "Improving" if better else "Worsening" if worse else "Stable"
    return {"label": label, "d_cv": round(d_cv, 2), "d_asym": round(d_asym, 2)}


@app.get("/")
def dashboard():
    return FileResponse(os.path.join(HERE, "static", "dashboard.html"))


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/ai/status")
def ai_status():
    return ai_screen.info()


@app.get("/api/calibration/status")
def calibration_status():
    return gm.calibration_info()


@app.get("/api/thresholds")
def thresholds():
    return gm.THRESHOLDS


@app.get("/api/patients")
def patients():
    with db() as c:
        out = []
        for p in c.execute("SELECT * FROM patients ORDER BY id"):
            last = c.execute(
                "SELECT * FROM sessions WHERE patient_id=? ORDER BY recorded_at DESC LIMIT 1",
                (p["id"],),
            ).fetchone()
            n_alert = sum(
                1 for r in c.execute("SELECT flags FROM sessions WHERE patient_id=?", (p["id"],))
                if json.loads(r["flags"])
            )
            out.append({
                "id": p["id"], "name": p["name"], "condition": p["condition"],
                "demo": bool(p["demo"]),
                "last_session": session_row(last) if last else None,
                "alert_sessions": n_alert,
            })
        return out


@app.get("/api/caseload")
def caseload():
    """One row per patient for the physician's overview: latest session, trend, sparklines."""
    now = datetime.datetime.now()
    out = []
    with db() as c:
        for p in c.execute("SELECT * FROM patients ORDER BY id"):
            rows = [session_row(r) for r in c.execute(
                "SELECT * FROM sessions WHERE patient_id=? ORDER BY recorded_at", (p["id"],))]
            last = rows[-1] if rows else None
            days = None
            if last:
                days = max((now - datetime.datetime.fromisoformat(last["recorded_at"])).days, 0)
            out.append({
                "id": p["id"], "name": p["name"], "condition": p["condition"], "demo": bool(p["demo"]),
                "n_sessions": len(rows),
                "last": {k: v for k, v in last.items() if k != "zones"} if last else None,
                "days_since": days,
                "spark_cv": [r["stride_cv"] for r in rows[-8:]],
                "spark_asym": [r["asym"] for r in rows[-8:]],
                "trend": trend_of(rows),
                "alert_sessions": sum(1 for r in rows if r["alert"]),
            })
    return out


@app.get("/api/patients/{pid}/notes")
def get_notes(pid: int):
    with db() as c:
        rows = c.execute("SELECT * FROM notes WHERE patient_id=? ORDER BY created DESC, id DESC", (pid,)).fetchall()
        return [dict(r) for r in rows]


@app.post("/api/patients/{pid}/notes")
def add_note(pid: int, n: NoteIn):
    text = n.text.strip()
    if not text:
        raise HTTPException(422, "Note is empty")
    with db() as c:
        if not c.execute("SELECT 1 FROM patients WHERE id=?", (pid,)).fetchone():
            raise HTTPException(404, "Patient not found")
        cur = c.execute("INSERT INTO notes(patient_id, created, author, text) VALUES (?,?,?,?)",
                        (pid, now_iso(), n.author.strip() or "Clinician", text[:2000]))
        return dict(c.execute("SELECT * FROM notes WHERE id=?", (cur.lastrowid,)).fetchone())


@app.post("/api/patients")
def add_patient(p: PatientIn):
    name = p.name.strip()
    if not name:
        raise HTTPException(422, "Name is required")
    with db() as c:
        pid = create_patient(c, name, p.condition.strip())
    return {"id": pid}


@app.get("/api/patients/{pid}/sessions")
def patient_sessions(pid: int):
    with db() as c:
        if not c.execute("SELECT 1 FROM patients WHERE id=?", (pid,)).fetchone():
            raise HTTPException(404, "Patient not found")
        rows = c.execute(
            "SELECT * FROM sessions WHERE patient_id=? ORDER BY recorded_at", (pid,)
        ).fetchall()
        return [session_row(r) for r in rows]


@app.get("/api/sessions/{sid}")
def one_session(sid: int):
    with db() as c:
        r = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        if not r:
            raise HTTPException(404, "Session not found")
        return session_row(r, with_trace=True)


@app.get("/api/alerts")
def alerts():
    with db() as c:
        rows = c.execute(
            """SELECT s.*, p.name AS patient_name FROM sessions s
               JOIN patients p ON p.id = s.patient_id
               WHERE s.flags != '[]' ORDER BY s.recorded_at DESC LIMIT 50"""
        ).fetchall()
        out = []
        for r in rows:
            d = session_row(r)
            d["patient_name"] = r["patient_name"]
            out.append(d)
        return out


@app.post("/api/sessions/upload")
async def upload_session(
    patient_id: int = Form(...),
    label: str = Form("walk"),
    model_score: Optional[float] = Form(None),
    recorded_at: Optional[str] = Form(None),
    mode: str = Form("inplace"),
    file: UploadFile = File(...),
):
    if mode not in gm.MODES:
        raise HTTPException(422, "mode must be walk, inplace or auto")
    text = (await file.read()).decode("utf-8", errors="ignore")
    try:
        t, raw = gm.parse_csv(text)
    except ValueError as e:
        raise HTTPException(422, str(e))
    m = gm.compute_metrics(t, raw, mode)
    if not m["ok"]:
        raise HTTPException(422, m["reason"])
    ai = ai_screen.screen(t, raw)        # None when no model file is installed
    with db() as c:
        if not c.execute("SELECT 1 FROM patients WHERE id=?", (patient_id,)).fetchone():
            raise HTTPException(404, "Patient not found")
        sid = save_session(c, patient_id, recorded_at or now_iso(), label, m, model_score, ai)
        r = c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
        return session_row(r)


def _recorded_at_from_name(stem):
    m = re.search(r"_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", stem)
    return "%s-%s-%sT%s:%s:%s" % m.groups() if m else None


@app.post("/api/import")
async def import_recordings(mode: str = Form("inplace"), files: list[UploadFile] = File(...)):
    """Import several recordings at once. The patient is taken from the start of the file name
    (abhi_steady_20261014_215508.csv -> patient Abhi, session name steady) and created when it does not exist."""
    if mode not in gm.MODES:
        raise HTTPException(422, "mode must be walk, inplace or auto")
    results = []
    for f in files:
        name = os.path.basename(f.filename or "recording.csv")
        stem = os.path.splitext(name)[0]
        parts = stem.split("_")
        subject = (parts[0].strip() or "Unknown").title()
        label = parts[1] if len(parts) > 1 and not parts[1].isdigit() else "recording"
        try:
            t, raw = gm.parse_csv((await f.read()).decode("utf-8", errors="ignore"))
            m = gm.compute_metrics(t, raw, mode)
            if not m["ok"]:
                raise ValueError(m["reason"])
        except ValueError as e:
            results.append({"file": name, "ok": False, "error": str(e)})
            continue
        ai = ai_screen.screen(t, raw)
        with db() as c:
            row = c.execute("SELECT id FROM patients WHERE lower(name)=lower(?)", (subject,)).fetchone()
            pid = row["id"] if row else create_patient(c, subject, "")
            sid = save_session(c, pid, _recorded_at_from_name(stem) or now_iso(), label, m, None, ai)
        results.append({"file": name, "ok": True, "patient_id": pid, "patient": subject, "session_id": sid,
                        "mode": m["mode"], "cadence": m["cadence"], "stride_cv": m["stride_cv"], "asym": m["asym"]})
    return results
