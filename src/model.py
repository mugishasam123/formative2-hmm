# HMM with diagonal Gaussian emissions: parameter init from labels, Viterbi decoding, Baum–Welch EM, evaluation helpers.

import numpy as np
import pandas as pd
try:
    from src.config import ACTIVITY_STATES
except ImportError:
    from src.config import STATES as ACTIVITY_STATES

# Columns that are not observation features (labels and metadata)
NON_FEATURE_COLS = ["activity", "split", "recording_id", "t_start", "t_end"]


def observation_columns(df: pd.DataFrame) -> list[str]:
    """List of column names used as observation (feature) dimensions."""
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def compute_zscore_params(df: pd.DataFrame, cols: list[str]) -> dict:
    """From training data, compute mean and std per column for Z-score; zero std is replaced by 1."""
    mu = df[cols].mean()
    sigma = df[cols].std(ddof=0).replace(0, 1.0)
    return {"mean": mu, "std": sigma, "cols": list(cols)}


def normalize_features(df: pd.DataFrame, zscore_params: dict) -> pd.DataFrame:
    """Apply Z-score using precomputed mean/std (e.g. from train set only). Returns a copy."""
    cols = zscore_params["cols"]
    out = df.copy()
    out[cols] = (out[cols] - zscore_params["mean"]) / zscore_params["std"]
    return out


def _clamp_variance(var_arr: np.ndarray, min_val: float = 1e-3) -> np.ndarray:
    arr = np.asarray(var_arr, dtype=float)
    arr[arr < min_val] = min_val
    return arr


def init_hmm_from_labels(
    train_df: pd.DataFrame,
    feat_cols: list[str],
    var_floor: float = 1e-3,
) -> dict:
    """
    Set HMM parameters from labeled windows: per-state mean and diagonal variance from
    windows with that label; initial distribution and transition matrix from first-window
    and consecutive-window counts per recording.
    """
    K = len(ACTIVITY_STATES)
    D = len(feat_cols)
    mu = np.zeros((K, D), dtype=float)
    sigma2 = np.ones((K, D), dtype=float)
    for k, label in enumerate(ACTIVITY_STATES):
        Xk = train_df.loc[train_df["activity"] == label, feat_cols].to_numpy()
        if len(Xk) == 0:
            mu[k] = 0.0
            sigma2[k] = 1.0
        else:
            mu[k] = Xk.mean(axis=0)
            sigma2[k] = _clamp_variance(Xk.var(axis=0, ddof=0), var_floor)

    pi_cnt = np.ones(K, dtype=float)
    trans_cnt = np.ones((K, K), dtype=float)
    label_to_k = {lbl: k for k, lbl in enumerate(ACTIVITY_STATES)}
    for _, grp in train_df.groupby("recording_id", sort=False):
        grp = grp.sort_values("t_start")
        states = grp["activity"].map(label_to_k).to_numpy()
        if len(states) == 0:
            continue
        pi_cnt[states[0]] += 1.0
        for i in range(len(states) - 1):
            trans_cnt[states[i], states[i + 1]] += 1.0
    pi = pi_cnt / pi_cnt.sum()
    A = trans_cnt / np.maximum(trans_cnt.sum(axis=1, keepdims=True), 1e-12)
    return {"means": mu, "vars": sigma2, "A": A, "pi": pi}


def _log_diag_gaussian(X: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Log density of N(x | mean, diag(var)) for each row of X; output shape (T,)."""
    d = X - mean
    inv = 1.0 / var
    q = np.sum(d * d * inv, axis=1)
    log_det = np.sum(np.log(var))
    dim = X.shape[1]
    return -0.5 * (q + log_det + dim * np.log(2.0 * np.pi))


def viterbi_decode(
    log_b: np.ndarray,
    log_A: np.ndarray,
    log_pi: np.ndarray,
) -> np.ndarray:
    """
    Most likely state sequence (length T) given log emission probs (T,K), log transition (K,K), log initial (K).
    """
    T, K = log_b.shape
    score = np.full((T, K), -np.inf)
    back = np.zeros((T, K), dtype=int)
    score[0] = log_pi + log_b[0]
    for t in range(1, T):
        combo = score[t - 1][:, None] + log_A
        back[t] = np.argmax(combo, axis=0)
        score[t] = combo[back[t], np.arange(K)] + log_b[t]
    path = np.zeros(T, dtype=int)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def viterbi_decode_sequences(df: pd.DataFrame, feat_cols: list[str], params: dict) -> pd.DataFrame:
    """Run Viterbi per recording; add columns y_pred (state index) and pred_activity (label)."""
    K = len(ACTIVITY_STATES)
    mu = params["means"]
    sigma2 = params["vars"]
    log_A = np.log(params["A"] + 1e-12)
    log_pi = np.log(params["pi"] + 1e-12)
    k_to_label = {k: lbl for k, lbl in enumerate(ACTIVITY_STATES)}
    result = []
    for _, grp in df.groupby("recording_id", sort=False):
        grp = grp.sort_values("t_start").copy()
        X = grp[feat_cols].to_numpy()
        log_b = np.zeros((len(grp), K), dtype=float)
        for k in range(K):
            log_b[:, k] = _log_diag_gaussian(X, mu[k], sigma2[k])
        path = viterbi_decode(log_b, log_A, log_pi)
        grp["y_pred"] = path
        grp["pred_activity"] = grp["y_pred"].map(k_to_label)
        result.append(grp)
    return pd.concat(result, ignore_index=True) if result else df.copy()


def build_confusion_matrix(decoded_df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Confusion matrix (rows = true, cols = predicted) and list of state labels in order."""
    lbl2k = {lbl: k for k, lbl in enumerate(ACTIVITY_STATES)}
    y_true = decoded_df["activity"].map(lbl2k).to_numpy()
    y_pred = decoded_df["pred_activity"].map(lbl2k).to_numpy()
    K = len(ACTIVITY_STATES)
    cm = np.zeros((K, K), dtype=int)
    for tt, pp in zip(y_true, y_pred):
        cm[tt, pp] += 1
    return cm, list(ACTIVITY_STATES)


def per_class_metrics(cm: np.ndarray) -> tuple[pd.DataFrame, float]:
    """Sensitivity and specificity per class; overall accuracy (fraction of correct predictions)."""
    K = cm.shape[0]
    total = cm.sum()
    rows = []
    for k in range(K):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        rows.append({"activity": ACTIVITY_STATES[k], "sensitivity": sens, "specificity": spec})
    acc = float(np.trace(cm) / total) if total > 0 else 0.0
    return pd.DataFrame(rows), acc


def _logsumexp(a: np.ndarray, axis=None) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    s = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(s, axis=axis)


def _log_emission_matrix(X: np.ndarray, mu: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """(T,D), (K,D), (K,D) -> (T,K) log emission probabilities."""
    T, D = X.shape
    K = mu.shape[0]
    out = np.empty((T, K), dtype=float)
    const = -0.5 * (D * np.log(2.0 * np.pi))
    for k in range(K):
        d = X - mu[k]
        inv = 1.0 / sigma2[k]
        q = np.sum(d * d * inv, axis=1)
        log_det = np.sum(np.log(sigma2[k]))
        out[:, k] = const - 0.5 * q - 0.5 * log_det
    return out


def _forward_backward(log_b: np.ndarray, log_A: np.ndarray, log_pi: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns total log-likelihood, posterior over states (T,K), and pairwise posteriors (T-1,K,K)."""
    T, K = log_b.shape
    alpha = np.empty((T, K))
    beta = np.empty((T, K))
    alpha[0] = log_pi + log_b[0]
    for t in range(1, T):
        alpha[t] = log_b[t] + _logsumexp(alpha[t - 1][:, None] + log_A, axis=0)
    loglik = float(_logsumexp(alpha[-1], axis=0))
    beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        beta[t] = _logsumexp(log_A + (log_b[t + 1] + beta[t + 1])[None, :], axis=1)
    gamma = np.exp(alpha + beta - loglik)
    xi = np.empty((T - 1, K, K))
    for t in range(T - 1):
        m = alpha[t][:, None] + log_A + (log_b[t + 1] + beta[t + 1])[None, :]
        m -= _logsumexp(m, axis=None)
        xi[t] = np.exp(m)
    return loglik, gamma, xi


def baum_welch_em(
    train_df: pd.DataFrame,
    feat_cols: list[str],
    init_params: dict,
    max_iter: int = 25,
    tol: float = 1e-3,
    var_floor: float = 1e-3,
) -> tuple[dict, list[float]]:
    """
    EM algorithm to refine HMM parameters. Stops when the gain in log-likelihood is below tol
    or after max_iter iterations. Returns updated params and list of log-likelihoods per iteration.
    """
    K = len(ACTIVITY_STATES)
    mu = init_params["means"].copy()
    sigma2 = init_params["vars"].copy()
    A = init_params["A"].copy()
    pi = init_params["pi"].copy()
    history = []
    for it in range(max_iter):
        gamma_sum = np.zeros(K)
        xi_sum = np.zeros((K, K))
        mu_num = np.zeros((K, len(feat_cols)))
        var_num = np.zeros((K, len(feat_cols)))
        pi_sum = np.zeros(K)
        total_ll = 0.0
        log_A = np.log(A + 1e-12)
        log_pi = np.log(pi + 1e-12)
        for _, grp in train_df.groupby("recording_id", sort=False):
            grp = grp.sort_values("t_start")
            X = grp[feat_cols].to_numpy()
            log_b = _log_emission_matrix(X, mu, sigma2)
            ll, gamma, xi = _forward_backward(log_b, log_A, log_pi)
            total_ll += ll
            gamma_sum += gamma.sum(axis=0)
            xi_sum += xi.sum(axis=0)
            pi_sum += gamma[0]
            mu_num += gamma.T @ X
            for k in range(K):
                d = X - mu[k]
                var_num[k] += (gamma[:, k][:, None] * (d * d)).sum(axis=0)
        A = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-12)
        pi = pi_sum / np.maximum(pi_sum.sum(), 1e-12)
        mu = mu_num / np.maximum(gamma_sum[:, None], 1e-12)
        sigma2 = var_num / np.maximum(gamma_sum[:, None], 1e-12)
        sigma2[sigma2 < var_floor] = var_floor
        history.append(total_ll)
        if it > 0 and (total_ll - history[-2]) < tol:
            break
    return {"means": mu, "vars": sigma2, "A": A, "pi": pi}, history
