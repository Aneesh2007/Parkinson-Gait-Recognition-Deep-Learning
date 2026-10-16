# Smart insole cloud backend and physician dashboard (demo)

Receives gait recordings from the insole, computes gait metrics, stores them, and shows them on a physician dashboard for a whole caseload.

## What the dashboard shows

- **Caseload:** all patients in one table with status, trend (improving, stable, worsening), last session, cadence, variability, a sparkline, search, filters and sorting by priority.
- **Patient page:** KPI tiles, rehabilitation trend, pressure trace, **foot pressure map** (average loading or playback of the walk, heel-to-toe roll-over), session list, **comparison of two sessions** (maps and measures side by side), clinical notes, CSV export and a printable summary.
- **Alerts:** every session outside the illustrative limits, across all patients.
- The foot map is interpolated from 3 sensors per foot. It is not a measured pressure image.
- Trend rule (illustrative): compare the first and latest sessions; a change of 0.5 points in variability or 1 point in asymmetry counts as better or worse.

## Run

    pip install -r requirements.txt
    python seed_demo.py          # adds 3 SIMULATED patients with 8 weekly sessions each
    uvicorn app:app --reload

Open http://127.0.0.1:8000. `python seed_demo.py --reset` rebuilds the demo data.

## Add a real recording

1. Record with `record.py` (it saves a CSV with columns n, t_ms, L_toe, L_mid, L_heel, R_toe, R_mid, R_heel).
2. On the dashboard, pick a patient, choose the CSV under "Upload a recording", press Upload.
   Or: `curl -F patient_id=1 -F label="home walk" -F file=@recording.csv http://127.0.0.1:8000/api/sessions/upload`

## What is computed (gait_metrics.py)

Per-channel pressure normalisation, stance detection with hysteresis on each foot, then cadence, stride time, stride-time variability (CV), stance time per foot and stance asymmetry.
Alert limits in `THRESHOLDS` are illustrative, not clinical.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | /api/patients | patients with latest session and alert count |
| GET | /api/caseload | one row per patient: latest session, trend, sparklines, days since last session |
| GET, POST | /api/patients/{id}/notes | clinical notes for a patient |
| POST | /api/patients | add a patient (name, condition) |
| GET | /api/patients/{id}/sessions | session history |
| GET | /api/sessions/{id} | one session with pressure trace |
| POST | /api/sessions/upload | upload a recording CSV |
| GET | /api/alerts | recent sessions that crossed a limit |
| GET | /api/thresholds | current alert limits |

## Limits

- No login. Add authentication and HTTPS before any real patient data.
- Demo patients are simulated; they are not measurements of real people.
- `model_score` (AI model output) is accepted by the upload endpoint, but no trained model is wired in yet.
- Not a medical device.

## AI gait screening (optional)

`ai_screen.py` runs the Task 1 CNN (3 sensors per foot) in plain NumPy on every uploaded recording.

1. In Colab, run `colab_export_cnn.py` as one cell. It downloads `cnn_s3.npz` and `cnn_s3_meta.json`.
2. Copy both files into this `cloud/` folder and restart the app.
3. `GET /api/ai/status` should say `available: true`. New uploads then get an AI score on the dashboard.

Without the model files nothing breaks: sessions show "Not screened". To see the layout before the model
is trained, run `python seed_demo.py --ai-preview` (values are illustrative and labelled so on screen).
The score is a research prototype (trained on a different insole, mostly older adults). Not a diagnosis.

## Calibration (optional, recommended)

Copy `calibration.json` (written by `calibrate.py`) into this `cloud/` folder. The app then equalises the six pads
for the foot map and the zone peaks: one standard (bottle) load is drawn on a log scale: 0.18 on every pad, 10 loads 0.61, 50 or more 1.0.
`GET /api/calibration/status` shows whether it is active. Without the file the raw signal is used.
This only affects the foot map and zone peaks. Steps, cadence, stride variability, asymmetry and the AI score do not depend on it.
Method: conductance (4095 - reading) / reading is proportional to force, so each pad is scaled by its own conductance at the calibration load.
Demo data from `seed_demo.py` is never calibrated.

## Stepping in place

Recordings made by stepping on the spot often fail the walking detector, because the pads of one foot never fully unload. When you upload a recording, choose the recording type:

- **Stepping in place** (default): finds the loading peaks of each foot's signal after removing slow drift. Reports step rate, step-time variability, contact-time asymmetry, how often the feet alternate, and the foot map for the stepping part only.
- **Walking**: stance and swing detection, for real walking.
- **Auto**: tries walking first and falls back to stepping in place.

The limits that raise a "Needs review" flag are wider for stepping in place (illustrative, not clinical). The AI score was trained on walking, so for in-place recordings it is shown with a caution and should not be read as a finding.

## Import, full-length playback and sensor quality

- **Import recordings** (Patients page): choose several CSVs at once. The patient comes from the start of the file name (`dhruv_steady_20261014_215920.csv` becomes patient Dhruv) and is created if new. Recording time comes from the timestamp in the name.
- **Trace window**: the pressure trace shows the whole recording. Use 15 s / 30 s / Whole recording and the slider to move along it. Foot-map playback also runs over the whole recording.
- The per-pad sensor quality card is hidden in the web app; the checks still run in the background.
- **Compare page**: latest session of every person side by side.
- Stepping in place is analysed by which foot carries the load (anti-phase), so steady one-foot-at-a-time marching works best.

## AI score and saturated pads

A pad that bottoms out under body weight (at or below 150 counts for more than 5% of samples) gives a clipped, flat-topped signal. The training data (gaitpdb) never clips, and the CNN scores such signals as Parkinson-like. Before scoring, `ai_screen.py` replaces a saturated pad by the average of the other pads of the same foot, and the dashboard shows the raw score next to the adjusted one. This removes a known sensor artefact; it does not validate the score. Fixing the hardware (a lower fixed resistor on the heel pads) is the real remedy.

## Whole-recording playback, step counter, mid-foot index

- The stored trace and playback now cover the whole recording (before, the stepping-in-place trace started when stepping started and stopped when it stopped, losing a few seconds, and walking sessions kept only 12 s).
- A **Steps counted** tile and a live step counter during playback (left, right and total up to the current time).
- An experimental **mid-foot load index** per foot (mid / (toe + mid + heel), from calibrated loads). It is shown only as a reliable value when no pad of that foot bottoms out. It is a pattern indicator for discussion, not a flat-foot diagnosis.

## Mid-foot pad equalisation
`pad_gain.json` holds a fixed gain per pad. It evens out the left and right mid-foot pads (the left one is about 2.6 times weaker). It applies to the foot map and load tables only, and only when a pad has no `calibration.json` entry. The gain is the same for everyone, so a person's own left/right difference is kept. Gait metrics and the AI score already normalise each channel per recording.
