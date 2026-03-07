# Plotting utilities for HMM: transition matrix, confusion matrix, decoding timeline, emission heatmap.

import numpy as np
import matplotlib.pyplot as plt
try:
    from src.config import ACTIVITY_STATES
except ImportError:
    from src.config import STATES as ACTIVITY_STATES


def show_transition_heatmap(A: np.ndarray, title: str = "Transition matrix A (from → to)") -> None:
    """Display the HMM transition probability matrix as a heatmap with values annotated."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    im = ax.imshow(A, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(ACTIVITY_STATES)))
    ax.set_yticks(range(len(ACTIVITY_STATES)))
    ax.set_xticklabels(ACTIVITY_STATES, rotation=0)
    ax.set_yticklabels(ACTIVITY_STATES)
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title(title, pad=12, fontsize=16)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j, i, f"{A[i, j]:.3f}", va="center", ha="center", color="black", fontsize=10)
    fig.tight_layout()
    plt.show()


def show_confusion_matrix(cm: np.ndarray, labels: list[str] | None = None, title: str = "Confusion matrix (test)") -> None:
    """Plot confusion matrix with counts and row-wise percentage in each cell."""
    labels = labels or list(ACTIVITY_STATES)
    row_totals = cm.sum(axis=1, keepdims=True).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(row_totals > 0, cm / row_totals, 0.0)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    im = ax.imshow(cm, cmap=plt.cm.Blues, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, pad=12, fontsize=18)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:d}\n({pct[i, j] * 100:.1f}%)",
                ha="center", va="center", color="black", fontsize=10, linespacing=1.2,
            )
    fig.tight_layout()
    plt.show()


def show_decoding_timeline(
    decoded: "pd.DataFrame",
    title: str = "Decoded activity vs ground truth over time",
) -> None:
    """Plot true and predicted state indices over window start time for one recording."""
    import pandas as pd
    lbl2i = {lbl: i for i, lbl in enumerate(ACTIVITY_STATES)}
    t = decoded["t_start"].to_numpy()
    y_true = np.array([lbl2i[lbl] for lbl in decoded["activity"]])
    y_pred = np.array([lbl2i[lbl] for lbl in decoded["pred_activity"]])
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(t, y_true, marker="o", linestyle="-", label="Ground truth")
    ax.plot(t, y_pred, marker="x", linestyle="--", label="Viterbi decode")
    ax.set_yticks(range(len(ACTIVITY_STATES)))
    ax.set_yticklabels(ACTIVITY_STATES)
    ax.set_xlabel("Time (window start, s)")
    ax.set_ylabel("Activity")
    ax.set_title(title, pad=10, fontsize=16)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    plt.show()


def show_emission_heatmap(
    means: np.ndarray,
    feature_names: list[str],
    title: str = "Emission mean vectors by state (Z-scored)",
) -> None:
    """Heatmap: rows = states, columns = features; shows how each state’s mean differs across features."""
    K, D = means.shape
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(means, cmap=plt.cm.RdBu_r, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Z-score")
    ax.set_yticks(range(K))
    ax.set_yticklabels(ACTIVITY_STATES)
    ax.set_ylabel("State")
    if D > 20:
        step = max(1, D // 20)
        ax.set_xticks(range(0, D, step))
        ax.set_xticklabels([feature_names[i] for i in range(0, D, step)], rotation=90)
    else:
        ax.set_xticks(range(D))
        ax.set_xticklabels(feature_names, rotation=90)
    ax.set_xlabel("Feature")
    ax.set_title(title, pad=12, fontsize=16)
    fig.tight_layout()
    plt.show()
