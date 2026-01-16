import numpy as np
import pandas as pd

def topk_rows(test_ids, P, terms, topk=500, min_score=1e-6):
    rows = []
    for i, pid in enumerate(test_ids):
        p = P[i]
        if p.size == 0:
            continue
        k = min(topk, p.size)
        idx = np.argpartition(-p, kth=k-1)[:k]
        idx = idx[np.argsort(-p[idx])]
        for j in idx:
            s = float(p[j])
            if s <= 0:
                continue
            s = max(s, min_score)
            rows.append((pid, terms[j], s))
    df = pd.DataFrame(rows, columns=["Protein_ID", "GO_term", "score"])
    df["score"] = df["score"].map(lambda x: float(f"{x:.3g}"))
    return df

def cap_global_1500(df_all):
    # df_all columns: Protein_ID, GO_term, score, aspect
    out = []
    for pid, g in df_all.groupby("Protein_ID", sort=False):
        g = g.sort_values("score", ascending=False).head(1500)
        out.append(g)
    return pd.concat(out, ignore_index=True)

def write_submission(df_all, path):
    # final format: Protein  GO  score (no header)
    df_all[["Protein_ID", "GO_term", "score"]].to_csv(path, sep="\t", index=False, header=False)
