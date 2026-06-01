"""
Version-safe preprocessing loader — NO sklearn required.
Provides:
    transform_genomic(df)
    transform_clinical(df)

Loads artifacts created by create_preprocessors_safe.py.
"""

import os
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)

# ------------------------------
# Load genomic preprocessing info
# ------------------------------
GEN_MEAN = np.load(os.path.join(HERE, "genomic_mean.npy"))
GEN_STD = np.load(os.path.join(HERE, "genomic_std.npy"))

with open(os.path.join(HERE, "genomic_cols.json"), "r", encoding="utf-8") as f:
    GEN_COLS = json.load(f)

# ------------------------------
# Load clinical preprocessing info
# ------------------------------
CLIN_MEAN = np.load(os.path.join(HERE, "clin_num_mean.npy"))
CLIN_STD = np.load(os.path.join(HERE, "clin_num_std.npy"))

with open(os.path.join(HERE, "clin_numeric_cols.json"), "r", encoding="utf-8") as f:
    CLIN_NUM_COLS = json.load(f)

with open(os.path.join(HERE, "clin_cat_vocabs.json"), "r", encoding="utf-8") as f:
    CLIN_CAT_VOCABS = json.load(f)


# ===========================================================
#                GENOMIC TRANSFORMER
# ===========================================================
def transform_genomic(df):
    """
    Input:
        df: DataFrame containing gene_### columns
    Output:
        numpy array standardized (n_samples, n_genes)
    """

    # Ensure required columns exist
    missing = [c for c in GEN_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing genomic columns: {missing}")

    # Convert to array
    X = df[GEN_COLS].to_numpy(dtype=np.float32)

    # Standardize
    X = (X - GEN_MEAN) / (GEN_STD + 1e-8)
    return X


# ===========================================================
#                CLINICAL TRANSFORMER
# ===========================================================
def transform_clinical(df):
    """
    Input:
        df: DataFrame with clinical info
    Output:
        numpy array (numeric scaled + one-hot categorical)
    """

    # -------- Numeric --------
    for col in CLIN_NUM_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing clinical numeric column: {col}")

    X_num = df[CLIN_NUM_COLS].to_numpy(dtype=np.float32)
    X_num = (X_num - CLIN_MEAN) / (CLIN_STD + 1e-8)

    # -------- Categorical (one-hot) --------
    cat_arrays = []

    for col, vocab in CLIN_CAT_VOCABS.items():
        if col not in df.columns:
            raise ValueError(f"Missing clinical categorical column: {col}")

        vals = df[col].astype(str).fillna("").to_numpy()
        onehot = np.zeros((len(vals), len(vocab)), dtype=np.float32)

        for i, v in enumerate(vals):
            if v in vocab:
                onehot[i, vocab.index(v)] = 1.0

        cat_arrays.append(onehot)

    # Concatenate numeric + categorical
    if cat_arrays:
        X_cat = np.concatenate(cat_arrays, axis=1)
        X = np.concatenate([X_num, X_cat], axis=1)
    else:
        X = X_num

    return X
