# Parkinson-Gait-Recognition-Deep-Learning

Deep learning for Parkinson's gait recognition (MLP, CNN, LSTM, Transformer, TFT) and a low-cost ESP32 smart insole with six force-sensitive resistors (FSRs) that records gait and scores it with a lightweight 1-D CNN.

Authors: Abhigya Mahawar, Aneesh Bansal, Shrihan Yekkele. Embedded Systems course project, October 2026.

## What is in this repository

| Folder | Contents |
|---|---|
| `src/`, `notebooks/` | Baseline model code (MLP, CNN, LSTM, Transformer, TFT) trained on all sensors of the PhysioNet gaitpdb dataset |
| `insole_6fsr/` | Work for the insole: models trained on 2, 3 and 8 sensors per foot, subject-wise cross-validation, leave-one-study-out test, CNN export and the NumPy scoring code |
| `firmware/` | ESP32 sketches (Bluetooth logger at 100 Hz), PC recorder, sensor and calibration scripts |
| `dashboard/` | FastAPI backend and physician dashboard (caseload, patient page, session comparison, AI card) |
| `docs/` | Project write-up (`.docx` and `.pdf`) and figures |

## Hardware

ESP32 DevKit V1 (38-pin), 6 FSRs (3 per foot: toe, mid, heel), each in a voltage divider with a 10 kOhm resistor, read on ADC1 pins. The ESP32 streams over Bluetooth Classic SPP at 100 Hz to `firmware/record.py`, which saves a CSV (`n,t_ms,L_toe,L_mid,L_heel,R_toe,R_mid,R_heel`). Wiring and photos are in `docs/`.

## Model

The 3-sensor-per-foot set (6 channels, matching the insole) is the deployed setup. A 1-D CNN with 37,090 parameters reaches macro F1 0.759 and AUC 0.866 on subject-wise 5-fold cross-validation, and macro F1 0.706 and AUC 0.805 when a whole study is held out. Majority-class baseline accuracy is 0.69. The model scores a 60 s recording in about 36 ms on a CPU.

## Run the dashboard

```
cd dashboard
pip install -r requirements.txt
python seed_demo.py        # adds simulated patients
uvicorn app:app --reload
```

Open http://127.0.0.1:8000.

## Limits

- The model was trained on gaitpdb, not on insole data, and is not validated on the insole. Scores on our own recordings are not reliable.
- No ESP32 inference, memory or power measurements. The model runs on the backend, not on the microcontroller.
- The AI card is only an indicator, not a diagnosis. Stroke evaluation is future work.
- No volunteer recordings or the PhysioNet data are included. Download gaitpdb from PhysioNet to retrain.
