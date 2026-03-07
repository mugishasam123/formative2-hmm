# Observation vectors for the HMM: per-window time- and frequency-domain features from acc/gyro.

from pathlib import Path
import numpy as np
import pandas as pd

# --- Windowing: fixed-length segments with step (overlap) ---

def _segment_indices(df: pd.DataFrame, window_sec: float, overlap_frac: float, sample_hz: int) -> list[tuple[int, int]]:
    """Index pairs (start, end) for non-overlapping or overlapping windows along the signal."""
    if df.empty:
        return []
    n_samp = int(round(window_sec * sample_hz))
    step = max(1, int(round(n_samp * (1.0 - overlap_frac))))
    n = len(df)
    indices = []
    start = 0
    while start + n_samp <= n:
        indices.append((start, start + n_samp))
        start += step
    return indices


# --- Time-domain: statistics that separate static vs dynamic and periodic motion ---

def _window_stats(signal: np.ndarray) -> dict:
    """Mean, spread (std), variance, range (ptp), and RMS for one signal window."""
    signal = np.asarray(signal, dtype=float)
    n = len(signal)
    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal, ddof=0)) if n > 0 else 0.0,
        "var": float(np.var(signal, ddof=0)) if n > 0 else 0.0,
        "ptp": float(np.ptp(signal)) if n > 0 else 0.0,
        "rms": float(np.sqrt(np.mean(signal**2))) if n > 0 else 0.0,
    }


def _signal_magnitude_area(acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> float:
    """Sum of mean absolute values on the three axes (total activity level in the window)."""
    return float(np.mean(np.abs(acc_x)) + np.mean(np.abs(acc_y)) + np.mean(np.abs(acc_z)))


# --- Frequency-domain: periodicity and energy from FFT ---

def _spectral_features(signal: np.ndarray, sample_hz: float) -> tuple[float, float]:
    """Dominant frequency (excluding DC) and normalized power from real FFT."""
    spec = np.fft.rfft(signal)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_hz)
    if len(power) > 1:
        ac_idx = int(np.argmax(power[1:])) + 1
        dominant_hz = float(freqs[ac_idx])
    else:
        dominant_hz = 0.0
    total_energy = float(power.sum() / len(power)) if len(power) > 0 else 0.0
    return dominant_hz, total_energy


def extract_window_features(
    csv_path: Path | str,
    window_sec: float,
    overlap_frac: float,
    sample_hz: int,
) -> pd.DataFrame:
    """
    Build the observation sequence for one recording: each row is one time window.
    Columns: metadata (activity, split, recording_id, t_start, t_end) plus numeric features.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    required = ["time_s", "ax", "ay", "az", "gx", "gy", "gz", "activity", "split", "recording_id"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path.name}")
    pairs = _segment_indices(df, window_sec, overlap_frac, sample_hz)
    rows = []
    for (i0, i1) in pairs:
        seg = df.iloc[i0:i1]
        ax = seg["ax"].to_numpy()
        ay = seg["ay"].to_numpy()
        az = seg["az"].to_numpy()
        gx = seg["gx"].to_numpy()
        gy = seg["gy"].to_numpy()
        gz = seg["gz"].to_numpy()
        acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

        sa = _window_stats(ax)
        sy = _window_stats(ay)
        sz = _window_stats(az)
        sgx = _window_stats(gx)
        sgy = _window_stats(gy)
        sgz = _window_stats(gz)
        smag = _window_stats(acc_mag)

        r_xy = float(np.corrcoef(ax, ay)[0, 1]) if len(ax) > 1 else 0.0
        r_xz = float(np.corrcoef(ax, az)[0, 1]) if len(ax) > 1 else 0.0
        r_yz = float(np.corrcoef(ay, az)[0, 1]) if len(ax) > 1 else 0.0
        sma_val = _signal_magnitude_area(ax, ay, az)

        dom_hz, energy = _spectral_features(acc_mag, float(sample_hz))

        rows.append({
            "activity": seg["activity"].iloc[0],
            "split": seg["split"].iloc[0],
            "recording_id": int(seg["recording_id"].iloc[0]),
            "t_start": float(seg["time_s"].iloc[0]),
            "t_end": float(seg["time_s"].iloc[-1]),
            "ax_mean": sa["mean"], "ax_std": sa["std"], "ax_var": sa["var"], "ax_ptp": sa["ptp"], "ax_rms": sa["rms"],
            "ay_mean": sy["mean"], "ay_std": sy["std"], "ay_var": sy["var"], "ay_ptp": sy["ptp"], "ay_rms": sy["rms"],
            "az_mean": sz["mean"], "az_std": sz["std"], "az_var": sz["var"], "az_ptp": sz["ptp"], "az_rms": sz["rms"],
            "gx_mean": sgx["mean"], "gx_std": sgx["std"], "gx_var": sgx["var"], "gx_ptp": sgx["ptp"], "gx_rms": sgx["rms"],
            "gy_mean": sgy["mean"], "gy_std": sgy["std"], "gy_var": sgy["var"], "gy_ptp": sgy["ptp"], "gy_rms": sgy["rms"],
            "gz_mean": sgz["mean"], "gz_std": sgz["std"], "gz_var": sgz["var"], "gz_ptp": sgz["ptp"], "gz_rms": sgz["rms"],
            "amag_mean": smag["mean"], "amag_std": smag["std"], "amag_var": smag["var"], "amag_ptp": smag["ptp"], "amag_rms": smag["rms"],
            "acc_corr_xy": r_xy, "acc_corr_xz": r_xz, "acc_corr_yz": r_yz,
            "acc_sma": sma_val,
            "amag_domfreq": dom_hz,
            "amag_energy": energy,
        })
    return pd.DataFrame(rows)


def build_feature_dataset(
    paths: list[Path],
    window_sec: float,
    overlap_frac: float,
    sample_hz: int,
) -> pd.DataFrame:
    """Stack observation rows from multiple cleaned CSV files into one DataFrame."""
    out = []
    for path in paths:
        tbl = extract_window_features(path, window_sec, overlap_frac, sample_hz)
        if not tbl.empty:
            out.append(tbl)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
