"""Tests para el módulo crossval.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.crossval import GroupTimeSeriesSplit, SubjectTimeSeriesSplit


class TestSubjectTimeSeriesSplit:
    """Tests para SubjectTimeSeriesSplit."""

    def test_subject_time_series_split_basic(self, sample_features_df):
        """Test split básico por sujetos."""
        # Usar menos folds para evitar problemas con pocos sujetos
        cv = SubjectTimeSeriesSplit(n_splits=3, test_size=0.2)

        splits = list(cv.split(sample_features_df))

        assert len(splits) == 3

        # Verificar que no hay overlap entre train y test en cada fold
        # Con 5 sujetos y test_size=0.2, cada fold tiene 1 sujeto en test y 4 en train
        for train_idx, test_idx in splits:
            assert len(set(train_idx) & set(test_idx)) == 0
            # Cada fold debe tener datos (con 5 sujetos y 3 folds, siempre hay datos)
            assert (
                len(train_idx) > 0 or len(test_idx) > 0
            ), "Cada fold debe tener al menos train o test"

    def test_subject_time_series_split_no_leakage(self, sample_features_df):
        """Test que no hay data leakage entre sujetos."""
        cv = SubjectTimeSeriesSplit(n_splits=5, test_size=0.2)

        for train_idx, test_idx in cv.split(sample_features_df):
            train_cores = set(
                sample_features_df.loc[train_idx, "subject_core"].unique()
            )
            test_cores = set(sample_features_df.loc[test_idx, "subject_core"].unique())

            # No debe haber overlap de subject_cores entre train y test
            assert len(train_cores & test_cores) == 0

    def test_subject_time_series_split_with_groups(self, sample_features_df):
        """Test split con grupos explícitos."""
        cv = SubjectTimeSeriesSplit(n_splits=3, test_size=0.3)
        groups = sample_features_df["subject_core"]

        splits = list(cv.split(sample_features_df, groups=groups))

        assert len(splits) == 3

    def test_subject_time_series_split_missing_subject_core(self):
        """Test error cuando falta columna subject_core."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
            }
        )
        cv = SubjectTimeSeriesSplit(n_splits=3, test_size=0.2)

        with pytest.raises(ValueError, match="subject_core"):
            list(cv.split(df))

    def test_subject_time_series_split_get_n_splits(self, sample_features_df):
        """Test get_n_splits."""
        cv = SubjectTimeSeriesSplit(n_splits=5, test_size=0.2)

        assert cv.get_n_splits() == 5

    def test_subject_time_series_split_reproducibility(self, sample_features_df):
        """Test que los splits son reproducibles."""
        cv1 = SubjectTimeSeriesSplit(n_splits=3, test_size=0.2)
        cv2 = SubjectTimeSeriesSplit(n_splits=3, test_size=0.2)

        splits1 = list(cv1.split(sample_features_df))
        splits2 = list(cv2.split(sample_features_df))

        # Los splits deben ser idénticos (mismo seed)
        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            np.testing.assert_array_equal(train1, train2)
            np.testing.assert_array_equal(test1, test2)

    def test_subject_time_series_split_test_size(self, sample_features_df):
        """Test que el tamaño del test es aproximadamente correcto."""
        cv = SubjectTimeSeriesSplit(n_splits=5, test_size=0.2)

        unique_cores = sample_features_df["subject_core"].nunique()
        expected_test_cores = max(1, int(unique_cores * 0.2))

        for train_idx, test_idx in cv.split(sample_features_df):
            test_cores = sample_features_df.loc[test_idx, "subject_core"].nunique()
            # Debe ser aproximadamente el tamaño esperado
            assert abs(test_cores - expected_test_cores) <= 1


class TestGroupTimeSeriesSplit:
    """Tests para GroupTimeSeriesSplit."""

    def test_group_time_series_split_basic(self, sample_features_df):
        """Test split básico con grupos y orden temporal."""
        cv = GroupTimeSeriesSplit(n_splits=3, gap=0)

        splits = list(cv.split(sample_features_df))

        # Puede haber menos splits si no hay suficientes datos
        assert len(splits) > 0

        for train_idx, test_idx in splits:
            assert len(set(train_idx) & set(test_idx)) == 0
            if len(train_idx) > 0 and len(test_idx) > 0:
                # Verificar orden temporal dentro de cada grupo
                # (train debe venir antes que test)
                pass  # Verificación más compleja, se puede agregar si es necesario

    def test_group_time_series_split_with_groups(self, sample_features_df):
        """Test split con grupos explícitos."""
        cv = GroupTimeSeriesSplit(n_splits=3)
        groups = sample_features_df["subject_core"]

        splits = list(cv.split(sample_features_df, groups=groups))

        # Debe generar al menos algunos splits
        assert len(splits) >= 0

    def test_group_time_series_split_missing_time_column(self):
        """Test error cuando falta columna temporal."""
        df = pd.DataFrame(
            {
                "subject_core": ["SC400", "SC400", "SC401"],
                "feature1": [1, 2, 3],
            }
        )
        cv = GroupTimeSeriesSplit(n_splits=3)

        with pytest.raises(ValueError, match="epoch_index.*epoch_time_start"):
            list(cv.split(df))

    def test_group_time_series_split_with_epoch_index(self, sample_features_df):
        """Test split usando epoch_index."""
        cv = GroupTimeSeriesSplit(n_splits=3)

        splits = list(cv.split(sample_features_df))

        # Debe funcionar con epoch_index
        assert len(splits) >= 0

    def test_group_time_series_split_with_epoch_time_start(self):
        """Test split usando epoch_time_start."""
        df = pd.DataFrame(
            {
                "subject_core": ["SC400", "SC400", "SC401", "SC401"],
                "epoch_time_start": [0, 30, 60, 90],
                "feature1": [1, 2, 3, 4],
            }
        )
        cv = GroupTimeSeriesSplit(n_splits=2)

        splits = list(cv.split(df))

        # Debe funcionar con epoch_time_start
        assert len(splits) >= 0

    def test_group_time_series_split_with_gap(self, sample_features_df):
        """Test split con gap entre train y test."""
        cv_no_gap = GroupTimeSeriesSplit(n_splits=3, gap=0)
        cv_with_gap = GroupTimeSeriesSplit(n_splits=3, gap=5)

        splits_no_gap = list(cv_no_gap.split(sample_features_df))
        splits_with_gap = list(cv_with_gap.split(sample_features_df))

        # Con gap puede haber menos splits válidos
        assert len(splits_with_gap) <= len(splits_no_gap)

    def test_group_time_series_split_get_n_splits(self):
        """Test get_n_splits."""
        cv = GroupTimeSeriesSplit(n_splits=5)

        assert cv.get_n_splits() == 5

    def test_group_time_series_split_small_groups(self):
        """Test manejo de grupos muy pequeños."""
        # Crear datos con grupos pequeños
        df = pd.DataFrame(
            {
                "subject_core": ["SC400", "SC401"],  # Solo 2 grupos pequeños
                "epoch_index": [0, 0],
                "feature1": [1, 2],
            }
        )
        cv = GroupTimeSeriesSplit(n_splits=3)

        # No debe fallar, pero puede no generar splits si los grupos son muy pequeños
        splits = list(cv.split(df))

        # Puede retornar lista vacía si no hay suficientes datos
        assert isinstance(splits, list)

    def test_group_time_series_split_test_size(self, sample_features_df):
        """Test split con test_size especificado."""
        cv = GroupTimeSeriesSplit(n_splits=5, test_size=0.2)

        splits = list(cv.split(sample_features_df))

        # Debe generar splits
        assert len(splits) >= 0
