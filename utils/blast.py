import numpy as np
import pandas as pd

def infer_blast_columns(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    # common variants
    query = cols.get("query") or cols.get("qseqid") or cols.get("qacc") or cols.get("qaccver")
    target = cols.get("target") or cols.get("sseqid") or cols.get("sacc") or cols.get("saccver")
    evalue = cols.get("evalue")
    bitscore = cols.get("bitscore")
    if query is None or target is None or evalue is None:
        raise ValueError(f"Cannot infer BLAST columns from: {list(df.columns)}")
    return query, target, evalue, bitscore

def blast_scores(
    blast_df: pd.DataFrame,
    test_ids: list[str],
    term_to_col: dict[str,int],
    train2go: dict[str,set[str]],
    topk: int = 50,
    evalue_thresh: float = 1e-3,
):
    """
    P_blast: (n_test, n_terms) float32 in [0,1] via noisy-OR over weighted hits
    """
    query_col, target_col, evalue_col, bitscore_col = infer_blast_columns(blast_df)

    df = blast_df.copy()
    df[evalue_col] = df[evalue_col].astype(float)
    if bitscore_col is not None:
        df[bitscore_col] = df[bitscore_col].astype(float)

    df = df[df[evalue_col] <= evalue_thresh]

    n_terms = len(term_to_col)
    P = np.zeros((len(test_ids), n_terms), dtype=np.float32)
    test_index = {pid: i for i, pid in enumerate(test_ids)}

    grouped = df.groupby(query_col, sort=False)

    for q, g in grouped:
        qi = test_index.get(str(q))
        if qi is None:
            continue

        if bitscore_col is not None:
            g = g.sort_values(bitscore_col, ascending=False).head(topk)
            w_raw = g[bitscore_col].to_numpy()
        else:
            g = g.sort_values(evalue_col, ascending=True).head(topk)
            w_raw = -np.log10(g[evalue_col].to_numpy() + 1e-300)

        if w_raw.size == 0:
            continue
        w = w_raw / (w_raw.sum() + 1e-12)

        targets = g[target_col].astype(str).to_list()
        for t, wi in zip(targets, w):
            gos = train2go.get(t)
            if not gos:
                continue
            for go in gos:
                col = term_to_col.get(go)
                if col is not None:
                    # noisy-OR
                    P[qi, col] = 1.0 - (1.0 - P[qi, col]) * (1.0 - float(wi))

    return P
