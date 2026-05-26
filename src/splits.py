"""Prompt 4 / 4.1 — Temporal splits: walk-forward with embargo (canonical)."""

import pandas as pd
import numpy as np
from pathlib import Path

from src.data_loader import PROCESSED_DIR, TARGET_COL
from src.features import FEATURES_CLEAN_PATH, SPREADS_CLEAN_PATH, ROUTING_PATH

WF_DIR = PROCESSED_DIR / "walkforward"

DEV_END = "2018-12-31"

WALKFORWARD_FOLDS = [
    (1, "2006-12-31", "2009-12-31", "GFC 2008"),
    (2, "2009-12-31", "2012-12-31", "Euro 2011"),
    (3, "2012-12-31", "2014-12-31", "Taper 2013"),
    (4, "2014-12-31", "2016-12-31", "China-Oil 2015-16"),
    (5, "2016-12-31", "2018-12-31", "Q4 2018 selloff"),
]


def _load_model_features():
    feats = pd.read_parquet(FEATURES_CLEAN_PATH)
    spreads = pd.read_parquet(SPREADS_CLEAN_PATH)
    merged = feats.join(spreads, how="inner")
    return merged


def walkforward_split(embargo_weeks=4, save=True, verbose=True):
    df_model = _load_model_features()
    df_routing = pd.read_parquet(ROUTING_PATH)

    def _split_one(df, is_model):
        dev = df.loc[:DEV_END]
        test = df.loc[DEV_END:].iloc[1:]  # strictly after DEV_END

        cv_folds = []
        train_normals_list = []
        for fold_id, train_end, val_end, crisis in WALKFORWARD_FOLDS:
            tr = df.loc[:train_end]
            embargo_start = tr.index[-1]
            # val: after train_end + embargo, up to val_end
            after_train = df.loc[train_end:].iloc[1:]
            if len(after_train) > embargo_weeks:
                val_start_idx = embargo_weeks
            else:
                val_start_idx = 0
            val_candidates = after_train.iloc[val_start_idx:]
            val = val_candidates.loc[:val_end]

            # Remove embargo rows from train end
            tr_clean = tr.iloc[:-embargo_weeks] if embargo_weeks < len(tr) else tr

            assert tr_clean.index.max() < val.index.min(), \
                f"Fold {fold_id}: train/val overlap!"

            fold = {"fold_id": fold_id, "train": tr, "val": val,
                    "crisis_captured": crisis}
            cv_folds.append(fold)

            if is_model and TARGET_COL in df.columns:
                normals = tr[tr[TARGET_COL] == 0]
                train_normals_list.append({"fold_id": fold_id, "train_normals": normals})

                n_pos = (val[TARGET_COL] == 1).sum()
                pct = 100 * n_pos / len(val) if len(val) else 0
                if verbose:
                    flag = ""
                    if pct < 10:
                        flag = " ⚠️ WARN low prevalence"
                    elif pct < 12:
                        flag = " ℹ️ NOTE"
                    if n_pos <= 20:
                        flag += " ⚠️ few positives"
                    print(f"  Fold {fold_id} ({crisis}): train={len(tr)}, "
                          f"val={len(val)}, Y=1 {n_pos} ({pct:.1f}%){flag}")

        result = {
            "development": dev,
            "test_holdout": test,
            "cv_folds": cv_folds,
        }
        if is_model:
            result["train_normals_per_fold"] = train_normals_list

        # COVID check
        if is_model:
            covid_rows = test.loc["2020-02":"2020-04"]
            assert len(covid_rows) > 0, "COVID must be in test holdout!"
            if verbose:
                print(f"  ✓ COVID in test holdout ({len(covid_rows)} rows)")

        return result

    if verbose:
        print("Model splits:")
    wf_models = _split_one(df_model, is_model=True)
    if verbose:
        print(f"\nRouting splits:")
    wf_routing = _split_one(df_routing, is_model=False)

    if save:
        WF_DIR.mkdir(parents=True, exist_ok=True)
        for key in ["development", "test_holdout"]:
            wf_models[key].to_parquet(WF_DIR / f"{key}.parquet")
            wf_routing[key].to_parquet(WF_DIR / f"routing_{key}.parquet")
        for fold_data in wf_models["cv_folds"]:
            i = fold_data["fold_id"]
            fold_data["train"].to_parquet(WF_DIR / f"fold_{i}_train.parquet")
            fold_data["val"].to_parquet(WF_DIR / f"fold_{i}_val.parquet")
        for norm_data in wf_models["train_normals_per_fold"]:
            i = norm_data["fold_id"]
            norm_data["train_normals"].to_parquet(WF_DIR / f"fold_{i}_train_normals.parquet")
        for fold_data in wf_routing["cv_folds"]:
            i = fold_data["fold_id"]
            fold_data["train"].to_parquet(WF_DIR / f"routing_fold_{i}_train.parquet")
            fold_data["val"].to_parquet(WF_DIR / f"routing_fold_{i}_val.parquet")
        if verbose:
            print(f"\nSaved walk-forward splits to {WF_DIR}")

    return {"models": wf_models, "routing": wf_routing}


if __name__ == "__main__":
    walkforward_split()
