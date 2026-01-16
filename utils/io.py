import os
import numpy as np
import pandas as pd
import h5py

def read_ids(path: str):
    """Read one protein ID per line from a txt file."""
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]

def load_embeddings_h5(h5_path: str, ids: list[str]) -> np.ndarray:
    """
    Load ProtT5 embeddings from HDF5 in the order given by `ids`.
    Assumes each id is a dataset in the H5 file.
    """
    with h5py.File(h5_path, "r") as h5:
        X = np.stack([np.asarray(h5[pid], dtype=np.float32) for pid in ids])
    return X

def load_train_set(train_set_path: str) -> pd.DataFrame:
    """
    Load train_set.tsv and normalize column names.
    Expects columns: Protein_ID, GO_term, aspect.
    """
    df = pd.read_csv(train_set_path, sep="\t", dtype=str).dropna()
    df["aspect"] = df["aspect"].str.strip().str.lower()
    df["Protein_ID"] = df["Protein_ID"].str.strip()
    df["GO_term"] = df["GO_term"].str.strip()
    return df

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
