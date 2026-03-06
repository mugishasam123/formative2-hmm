# data_loader.py — load ZIPs from activity folders, clean and write CSVs.

from pathlib import Path
import io
import re
import zipfile
import numpy as np
import pandas as pd

from src.config import (
    TARGET_HZ,
    EDGE_TRIM_SEC,
    MERGE_TOL_SEC,
    RANDOM_SEED,
    TRAIN_RATIO,
)

# Recording id from filenames like standing4.zip, walking12.zip
REC_ID_RE = re.compile(r"(\d+)\.zip$", re.IGNORECASE)


def _read_sensor(zp: zipfile.ZipFile, keywords: list[str]) -> pd.DataFrame | None:
    """Return first CSV matching keywords → DataFrame with [time_s, x, y, z]."""
    target = None
    for info in zp.infolist():
        nm = info.filename.lower()
        if nm.endswith(".csv") and any(k in nm for k in keywords):
            target = info
            break
    if target is None:
        return None

    with zp.open(target) as f:
        raw = f.read()
    df = pd.read_csv(io.BytesIO(raw))

    lower = {c.lower(): c for c in df.columns}
    for cand in ("seconds_elapsed", "timestamp", "time", "epoch(s)", "epoch(ms)"):
        if cand in lower:
            tcol = lower[cand]
            break
    else:
        tcol = df.columns[0]

    t = df[tcol].astype(float).to_numpy()
    if t.max() > 1e6:
        t = t / 1000.0
    t = t - t[0]

    def pick(*names):
        for n in names:
            if n in df:
                return df[n].astype(float).to_numpy()
            if n.upper() in df:
                return df[n.upper()].astype(float).to_numpy()
        return None

    x = pick("x", "ax", "accx")
    y = pick("y", "ay", "accy")
    z = pick("z", "az", "accz")
    if x is None or y is None or z is None:
        others = [
            c
            for c in df.columns
            if c != tcol and pd.api.types.is_numeric_dtype(df[c])
        ]
        if len(others) >= 3:
            x = df[others[0]].to_numpy(float)
            y = df[others[1]].to_numpy(float)
            z = df[others[2]].to_numpy(float)
        else:
            raise ValueError("x/y/z axes not found")

    out = pd.DataFrame({"time_s": t, "x": x, "y": y, "z": z})
    out = (
        out.drop_duplicates(subset="time_s")
        .sort_values("time_s")
        .reset_index(drop=True)
    )
    return out


def _estimate_hz(t: np.ndarray) -> float:
    if len(t) < 2:
        return np.nan
    dt = np.diff(t)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    return float(1.0 / np.median(dt)) if len(dt) else np.nan


def _resample(df: pd.DataFrame, hz: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    t0, t1 = df["time_s"].iloc[0], df["time_s"].iloc[-1]
    step = 1.0 / hz
    new_t = np.arange(t0, t1 + 1e-9, step)
    out = pd.DataFrame({"time_s": new_t})
    for c in ("x", "y", "z"):
        out[c] = np.interp(new_t, df["time_s"].to_numpy(), df[c].to_numpy())
    return out


def _merge(
    acc: pd.DataFrame | None,
    gyr: pd.DataFrame | None,
    tol: float,
) -> pd.DataFrame:
    """Nearest-time merge; returns [time_s, ax, ay, az, gx, gy, gz]."""
    if acc is None and gyr is None:
        return pd.DataFrame(columns=["time_s", "ax", "ay", "az", "gx", "gy", "gz"])
    if acc is None:
        g = gyr.rename(columns={"x": "gx", "y": "gy", "z": "gz"}).copy()
        g[["ax", "ay", "az"]] = 0.0
        return g[["time_s", "ax", "ay", "az", "gx", "gy", "gz"]]
    if gyr is None:
        a = acc.rename(columns={"x": "ax", "y": "ay", "z": "az"}).copy()
        a[["gx", "gy", "gz"]] = 0.0
        return a[["time_s", "ax", "ay", "az", "gx", "gy", "gz"]]

    A = acc.sort_values("time_s")
    G = gyr.sort_values("time_s").rename(columns={"x": "gx", "y": "gy", "z": "gz"})
    m = pd.merge_asof(A, G, on="time_s", tolerance=tol, direction="nearest")
    m = m.dropna(subset=["gx", "gy", "gz"])
    m = m.rename(columns={"x": "ax", "y": "ay", "z": "az"})
    return m[["time_s", "ax", "ay", "az", "gx", "gy", "gz"]].reset_index(drop=True)


def _trim(df: pd.DataFrame, sec: float) -> pd.DataFrame:
    if df.empty:
        return df
    t0 = df["time_s"].iloc[0] + sec
    t1 = df["time_s"].iloc[-1] - sec
    if t1 <= t0:
        return df.iloc[0:0].copy()
    return df[(df["time_s"] >= t0) & (df["time_s"] <= t1)].reset_index(drop=True)


def _parse_recording(zip_path: Path, activity_from_folder: str) -> tuple[str, int]:
    """Extract activity (normalized) and recording id from path and folder name."""
    activity = activity_from_folder.lower()
    m = REC_ID_RE.search(zip_path.name)
    if not m:
        raise ValueError(f"Cannot parse recording id from filename: {zip_path.name}")
    rec_id = int(m.group(1))
    return activity, rec_id


def _assign_split(
    activity: str, rec_id: int, train_ratio: float, seed: int
) -> str:
    """Deterministic train/test assignment from (activity, rec_id)."""
    h = hash((activity, rec_id, seed)) & 0x7FFFFFFF
    return "train" if (h / 0x7FFFFFFF) < train_ratio else "test"


def discover_recordings(
    data_dir: str | Path,
) -> list[tuple[Path, str, int]]:
    """
    Read from all subfolders of data_dir: find every *.zip under data_dir/<subfolder>/
    and return (path, activity, rec_id). Activity is taken from the subfolder name;
    rec_id from the filename (e.g. standing4.zip → 4).
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return []

    out: list[tuple[Path, str, int]] = []
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        activity_from_folder = sub.name
        for zpath in sorted(sub.glob("*.zip")):
            try:
                activity, rec_id = _parse_recording(zpath, activity_from_folder)
                out.append((zpath, activity, rec_id))
            except ValueError as e:
                raise ValueError(f"Bad recording {zpath}: {e}") from e
    return out


def unpack_and_clean_dir(
    raw_dir: str | Path,
    out_dir: str | Path,
    train_ratio: float | None = None,
) -> None:
    """
    Discover ZIPs under raw_dir (per-activity subdirs), clean each, assign split,
    and write *_cleaned.csv under out_dir/train/ and out_dir/test/.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    ratio = train_ratio if train_ratio is not None else TRAIN_RATIO
    seed = RANDOM_SEED

    (out_dir / "train").mkdir(parents=True, exist_ok=True)
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    recordings = discover_recordings(raw_dir)
    for zpath, activity, rec_id in recordings:
        split = _assign_split(activity, rec_id, ratio, seed)
        with zipfile.ZipFile(zpath, "r") as zp:
            acc = _read_sensor(zp, ["acc", "accelerometer"])
            gyr = _read_sensor(zp, ["gyro", "gyroscope"])

        if acc is not None:
            hz = _estimate_hz(acc["time_s"].to_numpy())
            if np.isfinite(hz) and abs(hz - TARGET_HZ) > 1.0:
                acc = _resample(acc, TARGET_HZ)
        if gyr is not None:
            hz = _estimate_hz(gyr["time_s"].to_numpy())
            if np.isfinite(hz) and abs(hz - TARGET_HZ) > 1.0:
                gyr = _resample(gyr, TARGET_HZ)

        m = _merge(acc, gyr, MERGE_TOL_SEC)
        m = _trim(m, EDGE_TRIM_SEC)

        m["activity"] = activity
        m["split"] = split
        m["recording_id"] = rec_id

        stem = zpath.stem
        dest = out_dir / split / f"{stem}_cleaned.csv"
        m.to_csv(dest, index=False)


def read_split_activity(csv_path: str | Path) -> tuple[str, str]:
    """Return (split, activity) from the first row of a cleaned CSV."""
    csv_path = Path(csv_path)
    s = pd.read_csv(csv_path, usecols=["split", "activity"], nrows=1)
    return str(s.loc[0, "split"]), str(s.loc[0, "activity"])


def estimate_hz_csv(csv_path: str | Path) -> float:
    """Estimate sampling rate (Hz) from a cleaned CSV (median Δt)."""
    s = pd.read_csv(csv_path, usecols=["time_s"])
    t = s["time_s"].to_numpy()
    if len(t) < 2:
        return float("nan")
    dt = t[1:] - t[:-1]
    dt = dt[dt > 0]
    return float(1.0 / pd.Series(dt).median()) if len(dt) else float("nan")


def duration_seconds_csv(csv_path: str | Path) -> float:
    """Duration (seconds) from first to last timestamp in a cleaned CSV."""
    s = pd.read_csv(csv_path, usecols=["time_s"])
    if s.empty:
        return 0.0
    return float(s["time_s"].iloc[-1] - s["time_s"].iloc[0])
