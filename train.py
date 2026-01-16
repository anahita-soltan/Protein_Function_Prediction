import os
import joblib
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

from utils.io import read_ids, load_embeddings_h5, load_train_set, ensure_dir
from utils.labels import build_Y, ASPECTS

def main(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train_dir = cfg["data"]["train_dir"]
    out_dir   = cfg["data"]["out_dir"]
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "models"))

    train_ids = read_ids(os.path.join(train_dir, "train_ids.txt"))
    gaf = load_train_set(os.path.join(train_dir, "train_set.tsv"))
    X = load_embeddings_h5(os.path.join(train_dir, "train_embeddings.h5"), train_ids)

    subset_n = cfg["model"]["subset_n"]
    rng = np.random.default_rng(cfg["model"]["random_state"])
    idx_all = np.arange(len(train_ids))

    if subset_n is not None:
        subset_n = min(int(subset_n), len(train_ids))
        idx_all = np.sort(rng.choice(idx_all, size=subset_n, replace=False))

    X_sub = X[idx_all]
    train_ids_sub = [train_ids[i] for i in idx_all]

    for short, aspect_value in ASPECTS.items():
        Y, mlb = build_Y(train_ids_sub, gaf, aspect_value)

        idx = np.arange(X_sub.shape[0])
        idx_tr, idx_val = train_test_split(
            idx, test_size=cfg["model"]["val_frac"], random_state=cfg["model"]["random_state"]
        )

        X_tr, X_val = X_sub[idx_tr], X_sub[idx_val]
        Y_tr, Y_val = Y[idx_tr], Y[idx_val]

        clf = OneVsRestClassifier(
            LogisticRegression(
                max_iter=cfg["model"]["max_iter"],
                solver="lbfgs",
            ),
            n_jobs=-1
        )

        print(f"[{short}] Training on {X_tr.shape[0]} proteins, {Y.shape[1]} terms...")
        clf.fit(X_tr, Y_tr)

        # quick sanity metric: avg predicted positives at 0.5
        P_val = clf.predict_proba(X_val)
        avg_pos = (P_val >= 0.5).sum(axis=1).mean()
        print(f"[{short}] sanity avg #terms@0.5 on val: {avg_pos:.2f}")

        joblib.dump(
            {"clf": clf, "mlb": mlb, "aspect_value": aspect_value},
            os.path.join(out_dir, "models", f"model_{short}.joblib"),
            compress=3
        )

    print("Done. Models saved in:", os.path.join(out_dir, "models"))

if __name__ == "__main__":
    main()
