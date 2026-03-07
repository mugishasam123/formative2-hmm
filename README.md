# Formative 2 — Human Activity Recognition (HMM)

**Course:** Machine Learning Techniques II (ALU)  
**Deliverable:** Formative 2 — Modeling activity states with a Hidden Markov Model from smartphone sensor streams.

---

## What This Repo Does

We take **accelerometer and gyroscope** logs from a phone (e.g. Sensor Logger), turn them into fixed-length **observation vectors**, and train a **Hidden Markov Model** to label each segment as one of four activities: **standing**, **walking**, **jumping**, or **still**. Everything runs in a single Jupyter notebook: raw ZIPs → cleaned CSVs → features → HMM fit → Viterbi decoding → metrics and plots on held-out test recordings.

**In short:**

- Ingest flat ZIP archives (`set1_` / `set2_`), resample to 100 Hz, merge acc/gyro, trim edges → cleaned CSVs.
- Sliding windows (e.g. 2 s, 75% overlap) → **41 numbers per window** (time stats, correlations, SMA, FFT-based dominant frequency and energy). Train-set Z-score applied to train and test.
- **HMM**: 4 states (one per activity), diagonal Gaussian emissions, supervised initialization then Baum–Welch until log-likelihood change is below threshold; **Viterbi** used to get the best state sequence per file.
- **Evaluation**: held-out test files only — confusion matrix, per-class sensitivity/specificity, overall accuracy, plus heatmaps (transition matrix, emission means) and decoded timelines. Section 7 in the notebook is a short **analysis and reflection** (which activities are easy/hard, what the transition matrix means, effect of noise/sampling, and ideas to improve).

---

## Why It Matters (Use Case)

The idea is to **label what someone is doing from a continuous stream of motion data** without manual tagging. That fits scenarios like:

- **Activity logging** — e.g. how much time is sedentary vs walking vs more intense movement.
- **Safety / care** — e.g. detecting unusual lack of movement or possible falls.
- **Rehab or fitness** — checking that prescribed activities are being done.
- **Context awareness** — e.g. muting notifications when the user is still or in a meeting.

So the code is aimed at “sensor stream in → activity labels out,” in a way that could sit inside a bigger app or research pipeline.

---

## Repo Layout

```
formative2-hmm/
├── notebooks/
│   └── formative2-hmm.ipynb       # End-to-end: load → clean → features → HMM → evaluate
├── src/
│   ├── __init__.py
│   ├── config.py                  # TARGET_HZ, ACTIVITY_STATES, trim/merge constants
│   ├── data_loader.py              # process_raw_archives, get_sampling_rate, get_duration_seconds, …
│   ├── feature_extractor.py       # extract_window_features, build_feature_dataset
│   ├── model.py                   # init_hmm_from_labels, viterbi_decode_sequences, baum_welch_em, metrics
│   └── visualizer.py              # show_transition_heatmap, show_emission_heatmap, show_confusion_matrix, …
├── data/
│   ├── raw/train/                 # set1_<Activity>_*.zip
│   ├── raw/test/                  # set2_<Activity>_*.zip
│   └── processed/{train,test}/   # *_cleaned.csv (created by the notebook)
├── requirements.txt
├── .gitignore
└── README.md
```

`data/` is not in the repo; you create it and drop your ZIPs in `raw/train/` and `raw/test/`.

---

## Setup and Run

**Requirements:** Python 3.9 or newer; `numpy` and `pandas` (see `requirements.txt`).

1. Clone or download the repo and go into the project folder:
   ```bash
   cd formative2-hmm
   ```

2. Optional but recommended: create and activate a virtualenv, then install deps:
   ```bash
   python3 -m venv venv
   source venv/bin/activate    # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Put your raw ZIPs in `data/raw/train/` and `data/raw/test/` with the naming below.

4. Open and run the notebook from the project root:
   ```bash
   jupyter notebook notebooks/formative2-hmm.ipynb
   ```
   Execute cells top to bottom. Section titles in the notebook match the pipeline (data cleaning → verification → analysis → feature extraction → HMM training → evaluation → analysis and reflection).

**What you’ll get:** Cleaned CSVs under `data/processed/`, plus in the notebook: transition matrix heatmap, emission heatmap, confusion matrix, per-activity sensitivity/specificity/accuracy, and example decoding timelines.

---

## Data Format and Recording Tips

**Naming:**  
- Train: `set1_<Activity>_<anything>.zip` in `data/raw/train/` (e.g. `set1_Jumping_2026-03-05_18-40-03.zip`).  
- Test:  `set2_<Activity>_<anything>.zip` in `data/raw/test/`.  

Inside each ZIP we expect CSV(s) with timestamps and accelerometer (and optionally gyroscope) axes; the loader looks for common column names (e.g. time, x/y/z or acc/gyro variants).

**Assignment-style recording:**

| Activity  | Length   | Tip |
|-----------|----------|-----|
| Standing  | 5–10 s   | Phone held steady at waist height |
| Walking   | 5–10 s   | Steady pace |
| Jumping   | 5–10 s   | Repeated jumps |
| Still     | 5–10 s   | Phone lying flat, no movement |

Use a **fixed target rate** (e.g. 100 Hz) across devices; the script resamples to the value in `config.TARGET_HZ`. For the report, note each member’s phone and sampling rate.

---

## Model and Features (Brief)

**Observations:** For every 2 s window (default 75% overlap), we compute **41 scalars**: per-axis (ax, ay, az, gx, gy, gz) and acc-magnitude stats (mean, std, variance, peak-to-peak, RMS), three acc correlations, SMA, and from the acc-magnitude FFT the dominant frequency and total spectral energy. All observation dimensions are Z-scored using **train** statistics only.

**HMM:** Four hidden states (standing, walking, jumping, still), 41-D observations, diagonal Gaussian emissions per state, 4×4 transition matrix, 4-D initial distribution. Parameters are first set from labeled training windows (supervised init), then updated with Baum–Welch EM until the increase in log-likelihood is smaller than 1e-3. Decoding is done with the **Viterbi algorithm** in log space, one recording at a time.

---

## Evaluation and Takeaways

All reported metrics (sensitivity, specificity, accuracy, confusion matrix) are on **test files only** (not used in training). Run the notebook on your data to see your numbers.

**Section 7 (Analysis and reflection)** in the notebook summarises: which activities were easier or harder to tell apart; how the learned transition matrix lines up with “one activity per clip”; how sampling rate and sensor noise matter; and what could be improved (more data, different or extra features, other sensors, etc.).

---

## Contributors

- **JD Amour Tuyishime** — Data collection, preprocessing (raw recordings, cleaning pipeline, resampling, verification).
- **Mugisha Samuel** — Feature extraction, HMM implementation (Viterbi, Baum–Welch), evaluation, visualizations, analysis and reflection.

---

## References

- Rabiner, L. R. (1989). A tutorial on hidden Markov models. _Proceedings of the IEEE_, 77(2), 257–286.
- Baum, L. E., et al. (1970). A maximization technique occurring in the statistical analysis of probabilistic functions of Markov chains. _The Annals of Mathematical Statistics_.
- Forney, G. D. (1973). The Viterbi algorithm. _Proceedings of the IEEE_, 61(3), 268–278.

---

## License

Submitted for **ALU Machine Learning Techniques II — Formative 2**. Educational use; ensure any data collection follows your institution’s and participants’ consent requirements.
