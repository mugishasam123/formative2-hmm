# config.py — constants for data loading and processing

TARGET_HZ = 100
EDGE_TRIM_SEC = 2.0
MERGE_TOL_SEC = 0.01
RANDOM_SEED = 42

# Hidden states of the HMM (activity labels), in fixed order for matrix indexing
ACTIVITY_STATES = ("standing", "walking", "jumping", "still")
STATES = ACTIVITY_STATES  # alias for compatibility

# Fraction of recordings to assign to train (rest → test) when no split is in the data.
TRAIN_RATIO = 0.8
