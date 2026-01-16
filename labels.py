import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

ASPECTS = {
    "mf": "molecular_function",
    "bp": "biological_process",
    "cc": "cellular_component",
}

def build_Y(train_ids, gaf: pd.DataFrame, aspect_value: str):
    """
    Build multi-label binary matrix Y for a given aspect.
    Rows follow `train_ids` order, columns are GO terms.
    """
    df = gaf[gaf["aspect"] == aspect_value]

    labels_per_protein = (
        df.groupby("Protein_ID")["GO_term"]
          .apply(list)
          .reindex(train_ids)
    )
    labels_per_protein = labels_per_protein.apply(
        lambda x: x if isinstance(x, list) else []
    )

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels_per_protein)
    return Y, mlb

def train2go_map(gaf: pd.DataFrame, aspect_value: str):
    """
    Build mapping: train protein -> set of GO terms for a given aspect.
    Used by BLAST post-hoc to transfer GO labels from train hits.
    """
    df = gaf[gaf["aspect"] == aspect_value]
    return df.groupby("Protein_ID")["GO_term"].apply(set).to_dict()
