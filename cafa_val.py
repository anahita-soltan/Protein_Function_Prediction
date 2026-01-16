import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from utils.io import read_ids, load_embeddings_h5, load_train_set, ensure_dir
from utils.labels import build_Y, ASPECTS
from utils.submit import topk_rows, cap_global_1500

def main(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train_dir = cfg["data"]["train_dir"]
    out_dir   = cfg["data"]["out_dir"]
    ensure_dir(out_dir)

    # where to put CAFA validation things
    val_out_dir = os.path.join(out_dir, "cafa_val")
    preds_dir   = os.path.join(val_out_dir, "preds")
    ensure_dir(val_out_dir)
    ensure_dir(preds_dir)

    # ------------ load full train ------------
    train_ids = read_ids(os.path.join(train_dir, "train_ids.txt"))
    gaf = load_train_set(os.path.join(train_dir, "train_set.tsv"))
    X = load_embeddings_h5(os.path.join(train_dir, "train_embeddings.h5"), train_ids)

    # ------------ choose subset + split into train/val ------------
    subset_n   = cfg["model"]["subset_n"]
    val_frac   = cfg["model"]["val_frac"]
    random_sta = cfg["model"]["random_state"]

    rng = np.random.default_rng(random_sta)
    idx_all = np.arange(len(train_ids))

    if subset_n is not None:
        subset_n = min(int(subset_n), len(train_ids))
        idx_all = np.sort(rng.choice(idx_all, size=subset_n, replace=False))

    X_sub = X[idx_all]
    ids_sub = [train_ids[i] for i in idx_all]

    idx = np.arange(X_sub.shape[0])
    idx_tr, idx_val = train_test_split(
        idx, test_size=val_frac, random_state=random_sta
    )

    X_tr, X_val = X_sub[idx_tr], X_sub[idx_val]
    ids_tr = [ids_sub[i] for i in idx_tr]
    ids_val = [ids_sub[i] for i in idx_val]

    # ground truth for validation proteins only
    gaf_val = gaf[gaf["Protein_ID"].isin(ids_val)]

    # CAFA ground truth must be: target_id \t term_id (first two cols)
    gt = gaf_val[["Protein_ID", "GO_term"]]

    gt_path = os.path.join(val_out_dir, "ground_truth_val.tsv")
    gt.to_csv(gt_path, sep="\t", index=False, header=False)
    print("Wrote validation ground truth:", gt_path, "rows:", len(gt))


    all_preds = []

    # ------------ train & predict per aspect ------------
    for short, aspect_value in ASPECTS.items():
        print(f"[{short}] aspect={aspect_value}")

        # Y for ALL subset proteins, then index by train/val
        Y_sub, mlb = build_Y(ids_sub, gaf, aspect_value)
        Y_tr = Y_sub[idx_tr]
        Y_val = Y_sub[idx_val]

        clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=cfg["model"]["max_iter"],
                solver="lbfgs",
            ),
            n_jobs=-1,
        )

        print(f"[{short}] train proteins={X_tr.shape[0]} val proteins={X_val.shape[0]} terms={Y_tr.shape[1]}")
        clf.fit(X_tr, Y_tr)

        P_val = clf.predict_proba(X_val).astype(np.float32)

        df_aspect = topk_rows(
            test_ids=ids_val,
            P=P_val,
            terms=mlb.classes_,
            topk=int(cfg["predict"]["topk_per_aspect"]),
            min_score=float(cfg["predict"]["min_score"]),
        )
        # no aspect column needed for CAFA, but we can keep it in an intermediate
        df_aspect["aspect"] = short

        # save per-aspect preds (optional, nice to inspect)
        aspect_path = os.path.join(preds_dir, f"preds_val_{short}.tsv")
        df_aspect.to_csv(aspect_path, sep="\t", index=False)
        print(f"[{short}] wrote {aspect_path} rows={len(df_aspect)}")

        all_preds.append(df_aspect[["Protein_ID", "GO_term", "score"]])  # strip aspect for CAFA

    # ------------ combine aspects + global cap ------------
    df_all = pd.concat(all_preds, ignore_index=True)
    df_all = cap_global_1500(df_all)  # respect 1500 terms / protein total

    # this is the CAFA-style prediction file for validation
    preds_path = os.path.join(val_out_dir, "ours_val.tsv")
    df_all.to_csv(preds_path, sep="\t", index=False, header=False)
    print("Wrote CAFA validation predictions:", preds_path, "rows:", len(df_all))

if __name__ == "__main__":
    main()

