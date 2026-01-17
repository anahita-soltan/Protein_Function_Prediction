# crossval.py
import os
import yaml
import numpy as np

from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from utils.io import read_ids, load_embeddings_h5, load_train_set, ensure_dir
from utils.labels import build_Y, ASPECTS


def main(cfg_path="config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train_dir = cfg["data"]["train_dir"]
    out_dir   = cfg["data"]["out_dir"]
    ensure_dir(out_dir)

    # load full training stuff
    train_ids = read_ids(os.path.join(train_dir, "train_ids.txt"))
    gaf       = load_train_set(os.path.join(train_dir, "train_set.tsv"))
    X_full    = load_embeddings_h5(os.path.join(train_dir, "train_embeddings.h5"),
                                   train_ids)

    # optional subsampling (same as train.py)
    subset_n  = cfg["model"]["subset_n"]
    rng       = np.random.default_rng(cfg["model"]["random_state"])
    idx_all   = np.arange(len(train_ids))

    if subset_n is not None:
        subset_n = min(int(subset_n), len(train_ids))
        idx_all  = np.sort(rng.choice(idx_all, size=subset_n, replace=False))

    X = X_full[idx_all]
    train_ids_sub = [train_ids[i] for i in idx_all]

    print(f"Using {len(train_ids_sub)} proteins for CV, X shape = {X.shape}")

    kf = KFold(
        n_splits=10,
        shuffle=True,
        random_state=cfg["model"]["random_state"],
    )

    for short, aspect_value in ASPECTS.items():
        Y, mlb = build_Y(train_ids_sub, gaf, aspect_value)
        print(f"\n[{short}] aspect={aspect_value}, labels={Y.shape[1]}")

        fold_scores = []

        for fold, (tr_idx, val_idx) in enumerate(kf.split(X), start=1):
            clf = OneVsRestClassifier(
                LogisticRegression(
                    max_iter=cfg["model"]["max_iter"],
                    solver="lbfgs",
                ),
                n_jobs=-1,
            )
            clf.fit(X[tr_idx], Y[tr_idx])

            P_val   = clf.predict_proba(X[val_idx])
            Y_pred  = (P_val >= 0.5).astype(int)

            f_micro = f1_score(Y[val_idx], Y_pred,
                               average="micro", zero_division=0)
            fold_scores.append(f_micro)
            print(f"  fold {fold:02d}: F1_micro = {f_micro:.3f}")

        mean_f = float(np.mean(fold_scores))
        std_f  = float(np.std(fold_scores))
        print(f"[{short}] 10-fold F1_micro = {mean_f:.3f} ± {std_f:.3f}")


if __name__ == "__main__":
    main()
