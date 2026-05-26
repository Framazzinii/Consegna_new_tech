"""Prompt 5 — FoldScaler: per-fold StandardScaler with no leakage."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.data_loader import TARGET_COL
from src.features import ROUTING_PATH


def _add_equity_bond_corr(df):
    """Add equity_bond_corr_13w from routing triggers to the model feature set (→ 57 cols)."""
    if "equity_bond_corr_13w" in df.columns:
        return df
    routing = pd.read_parquet(ROUTING_PATH)
    if "equity_bond_corr_13w" in routing.columns:
        corr_col = routing[["equity_bond_corr_13w"]]
        df = df.join(corr_col, how="left")
    return df


def prepare_X_y(df):
    """Split df into X (numeric features) and y (target). Adds equity_bond_corr_13w."""
    df = _add_equity_bond_corr(df)
    df = df.dropna()
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    return X, y


class FoldScaler:
    """Per-fold StandardScaler. Fits on fold train, transforms train+val.
    Final scaler fits on full development set for test holdout."""

    def __init__(self):
        self.scaler_ = None
        self.feature_names_ = None

    def fit(self, X_train):
        self.scaler_ = StandardScaler()
        self.scaler_.fit(X_train)
        self.feature_names_ = list(X_train.columns) if hasattr(X_train, "columns") else None
        return self

    def transform(self, X):
        arr = self.scaler_.transform(X)
        if hasattr(X, "columns"):
            return pd.DataFrame(arr, index=X.index, columns=X.columns)
        return arr

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
