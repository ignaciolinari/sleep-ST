from __future__ import annotations

import pytest

pytest.importorskip("yasa")

import pandas as pd  # noqa: E402
from src.models.data_preparation import prepare_train_test_split  # noqa: E402


def test_prepare_train_test_split_preserves_class_coverage():
    stages = ["W", "N1", "N2", "N3", "REM"]
    df = pd.DataFrame(
        {
            "stage": stages * 2,
            "subject_core": ["A"] * len(stages) + ["B"] * len(stages),
            "session_idx": [0] * len(stages) + [1] * len(stages),
        }
    )

    train_df, test_df, _ = prepare_train_test_split(
        df,
        test_size=0.5,
        val_size=None,
        random_state=0,
        ensure_class_coverage=True,
        required_classes=stages,
    )

    assert set(train_df["stage"].unique()) == set(stages)
    assert set(test_df["stage"].unique()) == set(stages)


def test_prepare_train_test_split_detects_missing_class():
    df = pd.DataFrame(
        {
            "stage": ["W", "W", "W", "N1", "N1", "N1"],
            "subject_core": ["A", "A", "A", "B", "B", "B"],
            "session_idx": [0, 0, 0, 0, 0, 0],
        }
    )

    with pytest.raises(ValueError):
        prepare_train_test_split(
            df,
            test_size=0.5,
            val_size=None,
            random_state=0,
            ensure_class_coverage=True,
            required_classes=["W", "N1"],
            max_attempts=5,
        )


def test_temporal_split_uses_latest_sessions_for_test():
    df = pd.DataFrame(
        {
            "stage": ["W", "N1", "W", "N1", "W", "N1", "W", "N1"],
            "subject_core": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "session_idx": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )

    train_df, test_df, _ = prepare_train_test_split(
        df,
        test_size=0.5,
        val_size=None,
        random_state=0,
        ensure_class_coverage=True,
        required_classes=["W", "N1"],
        temporal_split=True,
    )

    assert set(test_df["session_idx"].unique()) == {1}
    assert set(train_df["session_idx"].unique()) == {0}
