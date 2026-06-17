"""
Layer 4.5 — chronological train / holdout split (leak-free backtest).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

TRAIN_END = "2024-02-29 23:59:59"
VAL_START = "2024-03-01 00:00:00"


@dataclass
class TimeSplit:
    train_mask: pd.Series
    val_mask: pd.Series
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    split_column: str = "start_local"


def build_time_split(
    df: pd.DataFrame,
    time_col: str = "start_local",
    train_end: str = TRAIN_END,
    val_start: str = VAL_START,
) -> TimeSplit:
    """Single chronological split: train Nov 2023–Feb 2024, holdout Mar–Apr 2024."""
    t = pd.to_datetime(df[time_col], errors="coerce")
    train_end_ts = pd.Timestamp(train_end, tz=t.dt.tz)
    val_start_ts = pd.Timestamp(val_start, tz=t.dt.tz)
    train_mask = t <= train_end_ts
    val_mask = t >= val_start_ts
    return TimeSplit(
        train_mask=train_mask,
        val_mask=val_mask,
        train_end=train_end_ts,
        val_start=val_start_ts,
        split_column=time_col,
    )


def split_summary(df: pd.DataFrame, split: TimeSplit) -> dict:
    t = pd.to_datetime(df[split.split_column], errors="coerce")
    return {
        "n_total": len(df),
        "n_train": int(split.train_mask.sum()),
        "n_val": int(split.val_mask.sum()),
        "train_start": str(t[split.train_mask].min()),
        "train_end": str(split.train_end),
        "val_start": str(split.val_start),
        "val_end": str(t[split.val_mask].max()),
    }
