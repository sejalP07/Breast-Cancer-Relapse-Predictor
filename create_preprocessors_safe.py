# ================================================================
#  create_preprocessors_safe.py
#  RUN THIS ONCE to generate safe preprocessing artifacts that
#  DO NOT depend on sklearn and WILL NOT break in Streamlit.
#
#  After running, use:  from preprocessors_safe.loader import transform_genomic, transform_clinical
# ================================================================

import os
import json
import numpy as np
import pandas as pd

# ----------------------------------------
# Paths
# ----------------------------------------
PROJECT_DIR = os.path.abspath(".")
GENOMIC_CSV = os.path.join(PROJECT_DIR, "genomic.csv")
CLINICAL_CSV = os.path.join(PROJECT_DIR, "clinical.csv")

OUT_DIR = os.path.join(PROJECT_DIR, "preprocessors_safe")
os.makedirs(OUT_DIR, exist_ok=True)

print("Project Folder:", PROJECT_DIR)
print("Saving preprocessors to:", OUT_DIR)

# ----------------------------------------
# Load data
# ----------------------------------------
if not os.path.exists(GENOMIC_CSV):
    raise FileNotFoundError("ERROR: genomic.csv not found in project folder!")

if not os.path.exists(CLINICAL_CSV):
    raise FileNotFoundError("ERROR: clinical.csv not found in project folder!")

gen_df = pd.read_csv(GENOMIC_CSV)
clin_df = pd.read_csv(CLINICAL_CSV)

# ----------------------------------------
# GENOMIC PREPROCESSING INFO
# ----------------------------------------
genomic_cols = [c for c in gen_df.columns if c.startswith("gene_")]
if not genomic_cols:
    raise ValueError("No gene_### columns found in genomic.csv")

gen_mean = gen_df[genomic_cols].mean().to_numpy(dtype=np.float32)
gen_std = gen_df[genomic_cols].std().replace(0, 1).to_numpy(dtype=np.float32)

np.save(os.path.join(OUT_DIR, "genomic_mean.npy"), gen_mean)
np.save(os.path.join(OUT_DIR, "genomic_std.npy"), gen_std)

with open(os.path.join(OUT_DIR, "genomic_cols.json"), "w", encoding="utf-8") as f:
    json.dump(genomic_cols, f, indent=2)

# ----------------------------------------
# CLINICAL PREPROCESSING INFO
# ----------------------------------------
clin_numeric_cols = ["age", "lymph_nodes", "tumor_size_cm"]
clin_categorical_cols = ["treatment_type", "tumor_stage"]

# Validate presence
missing = [c for c in clin_numeric_cols + clin_categorical_cols if c not in clin_df.columns]
if missing:
    raise ValueError("Clinical CSV missing required columns: " + str(missing))

# Save numeric stats
clin_mean = clin_df[clin_numeric_cols].mean().to_numpy(dtype=np.float32)
clin_std = clin_df[clin_numeric_cols].std().replace(0, 1).to_numpy(dtype=np.float32)

np.save(os.path.join(OUT_DIR, "clin_num_mean.npy"), clin_mean)
np.save(os.path.join(OUT_DIR, "clin_num_std.npy"), clin_std)

with open(os.path.join(OUT_DIR, "clin_numeric_cols.json"), "w", encoding="utf-8") as f:
    json.dump(clin_numeric_cols, f, indent=2)

# Save categorical vocabularies
cat_vocabs = {}
for col in clin_categorical_cols:
    cat_vocabs[col] = clin_df[col].astype(str).dropna().unique().tolist()

with open(os.path.join(OUT_DIR, "clin_cat_vocabs.json"), "w", encoding="utf-8") as f:
    json.dump(cat_vocabs, f, indent=2)

# ----------------------------------------
# WRITE loader.py (runtime preprocessing module)
# ----------------------------------------
loader_code = '''
"""
Version-safe preprocessing loader — NO sklearn required.
Provides:
    transform_genomic(df)
    transform_clinical(df)
"""

import os, json, numpy as np, pandas as pd

HERE = os.path.dirname(__file__)

# Load genomic settings
GEN_MEAN = np.load(os.path.join(HERE, "genomic_mean.npy"))
GEN_STD = np.load(os.path.join(HERE, "genomic_std.npy"))
with open(os.path.join(HERE, "genomic_cols.json"), "r", encoding="utf-8") as f:
    GEN_COLS = json.load(f)

# Load clinical settings
CLIN_MEAN = np.load(os.path.join(HERE, "clin_num_mean.npy"))
CLIN_STD = np.load(os.path.join(HERE, "clin_num_std.npy"))
with open(os.path.join(HERE, "clin_numeric_cols.json"), "r", encoding="utf-8") as f:
    CLIN_NUM_COLS = json.load(f)

with open(os.path.join(HERE, "clin_cat_vocabs.json"), "r", encoding="utf-8") as f:
    CLIN_CAT_VOCABS = json.load(f)


def transform_genomic(df):
    missing = [c for c in GEN_COLS if c not in df.columns]
    if missing:
        raise ValueError("Missing genomic columns: " + str(missing))

    x = df[GEN_COLS].to_numpy(dtype=np.float32)
    x = (x - GEN_MEAN) / (GEN_STD + 1e-8)
    return x


def transform_clinical(df):
    # numeric
    for col in CLIN_NUM_COLS:
        if col not in df.columns:
            raise ValueError("Missing clinical numeric column: " + col)

    num = df[CLIN_NUM_COLS].to_numpy(dtype=np.float32)
    num = (num - CLIN_MEAN) / (CLIN_STD + 1e-8)

    # categorical → one-hot
    cat_arrays = []
    for col, vocab in CLIN_CAT_VOCABS.items():
        if col not in df.columns:
            raise ValueError("Missing clinical categorical column: " + col)

        vals = df[col].astype(str).fillna("").to_numpy()
        oh = np.zeros((len(vals), len(vocab)), dtype=np.float32)

        for i, v in enumerate(vals):
            if v in vocab:
                oh[i, vocab.index(v)] = 1.0

        cat_arrays.append(oh)

    if cat_arrays:
        cat = np.concatenate(cat_arrays, axis=1)
        return np.concatenate([num, cat], axis=1)
    else:
        return num
'''

with open(os.path.join(OUT_DIR, "loader.py"), "w", encoding="utf-8") as f:
    f.write(loader_code)

print("\n🎉 SAFE PREPROCESSORS CREATED SUCCESSFULLY!")
print("Files generated in:", OUT_DIR)
print("\nUse in app.py:\n")
print("from preprocessors_safe.loader import transform_genomic, transform_clinical")
