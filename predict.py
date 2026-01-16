import os
import joblib
import yaml
import numpy as np
import pandas as pd

from utils.io import read_ids, load_embeddings_h5, load_train_set, ensure_dir
from utils.labels import train2go_map, ASPECTS
from utils.blast import blast_scores
from utils.submit import topk_rows, cap_global_1500, write_submission

def main(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train_dir = cfg["data"]["train_dir"]
    test_dir  = cfg["data"]["test_dir"]
    out_dir   = cfg["data"]["out_dir"]

    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "preds"))

    # load data
    gaf = load_train_set(os.path.join(train_dir, "train_set.tsv"))

    test_ids = read_ids(os.path.join(test_dir, "test_ids.txt"))
    X_test = load_embeddings_h5(os.path.join(test_dir, "test_embeddings.h5"), test_ids)

    blast_enabled = bool(cfg["blast"]["enabled"])
    if blast_enabled:
        blast_path = os.path.join(test_dir, "blast_test_results.tsv")
        blast_df = pd.read_csv(blast_path, sep="\t")
        print("Loaded BLAST:", blast_df.shape, "columns:", list(blast_df.columns))

    all_dfs = []

    for short, aspect_value in ASPECTS.items():
        bundle = joblib.load(os.path.join(out_dir, "models", f"model_{short}.joblib"))
        clf = bundle["clf"]
        mlb = bundle["mlb"]

        P_model = clf.predict_proba(X_test).astype(np.float32)

        if blast_enabled:
            train2go = train2go_map(gaf, aspect_value)
            term_to_col = {t:i for i,t in enumerate(mlb.classes_)}
            P_blast = blast_scores(
                blast_df=blast_df,
                test_ids=test_ids,
                term_to_col=term_to_col,
                train2go=train2go,
                topk=int(cfg["blast"]["topk"]),
                evalue_thresh=float(cfg["blast"]["evalue_thresh"]),
            )
            alpha = float(cfg["blast"]["alpha"])
            P = alpha * P_model + (1.0 - alpha) * P_blast
        else:
            P = P_model

        df_aspect = topk_rows(
            test_ids=test_ids,
            P=P,
            terms=mlb.classes_,
            topk=int(cfg["predict"]["topk_per_aspect"]),
            min_score=float(cfg["predict"]["min_score"]),
        )
        df_aspect["aspect"] = short
        all_dfs.append(df_aspect)

        df_aspect.to_csv(os.path.join(out_dir, "preds", f"preds_{short}.tsv"),
                         sep="\t", index=False)
        print(f"[{short}] wrote preds_{short}.tsv rows={len(df_aspect)}")

    df_all = pd.concat(all_dfs, ignore_index=True)

    # global cap of 1500 across aspects
    df_all = cap_global_1500(df_all)

    out_path = os.path.join(out_dir, "submission.tsv")
    write_submission(df_all, out_path)
    print("Wrote final submission:", out_path, "rows:", len(df_all))

if __name__ == "__main__":
    main()
