"""Prompt 1 — Load and clean raw Bloomberg xlsx, coverage report."""

from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_FILE = RAW_DIR / "04_May_Zenti_exercises.xlsx"

TARGET_COL = "Y"
DATE_COL = "Date"


def load_metadata(path=None):
    path = path or DEFAULT_FILE
    meta = pd.read_excel(path, sheet_name="Metadata")
    meta.columns = ["variable", "description", "type"]
    return meta


def load_raw(path=None, sheet="Markets"):
    path = path or DEFAULT_FILE
    df = pd.read_excel(path, sheet_name=sheet)
    df.rename(columns={"Data": DATE_COL}, inplace=True)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df.sort_values(DATE_COL, inplace=True)
    df.set_index(DATE_COL, inplace=True)
    return df


def trim_to_common_coverage(df):
    first_valid = df.apply(lambda s: s.first_valid_index())
    last_valid = df.apply(lambda s: s.last_valid_index())
    start = first_valid.max()
    end = last_valid.min()
    df = df.loc[start:end]
    df = df.dropna()
    return df


def coverage_report(df):
    records = []
    for col in df.columns:
        s = df[col]
        records.append({
            "feature": col,
            "first_valid": s.first_valid_index(),
            "last_valid": s.last_valid_index(),
            "n_obs": s.count(),
            "n_missing": s.isna().sum(),
            "pct_missing": 100.0 * s.isna().sum() / len(s),
        })
    return pd.DataFrame(records).set_index("feature")


def load_dataset(path=None, sheet="Markets", verbose=True):
    df = load_raw(path, sheet)
    df = trim_to_common_coverage(df)
    if verbose:
        report = coverage_report(df)
        print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} cols "
              f"({df.index.min().date()} → {df.index.max().date()})")
        print(f"Target prevalence: Y=1 {(df[TARGET_COL]==1).sum()}/{len(df)} "
              f"({100*(df[TARGET_COL]==1).mean():.1f}%)")
        print(report.to_string())
    return df


if __name__ == "__main__":
    df = load_dataset()
