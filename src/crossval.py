"""Cross-validation para series temporales de sueño.

Este módulo implementa estrategias de cross-validation que respetan:
1. Grupos por sujeto (evita data leakage entre sujetos)
   - Todos los epochs de un mismo subject_core van al mismo conjunto
   - Esto es suficiente para sleep staging donde cada sujeto tiene una o múltiples noches completas

Nota: No se divide temporalmente dentro de cada sujeto porque todos los epochs
de un sujeto van al mismo conjunto (train/test/val). Si en el futuro se necesita
dividir temporalmente dentro de una misma noche, se puede usar GroupTimeSeriesSplit.
"""

from __future__ import annotations

from typing import Iterator, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator, TimeSeriesSplit


class GroupTimeSeriesSplit(BaseCrossValidator):
    """Cross-validator que combina TimeSeriesSplit con grupos.

    Usa TimeSeriesSplit de sklearn dentro de cada grupo para respetar:
    - Grupos (ej: subject_core) - todos los epochs de un grupo van al mismo fold
    - Orden temporal dentro de cada grupo - train siempre viene antes de test

    NOTA: Esta clase está disponible para casos avanzados donde se necesite
    dividir temporalmente dentro de una misma noche/sesión. Para el caso
    estándar de sleep staging (donde todos los epochs de un sujeto van al
    mismo conjunto), usar SubjectTimeSeriesSplit es más simple y apropiado.

    Basado en TimeSeriesSplit de sklearn pero aplicado por grupo.

    Parameters
    ----------
    n_splits : int
        Número de folds
    test_size : float, optional
        Proporción del test set (si se especifica, calcula n_splits automáticamente)
    max_train_size : int, optional
        Tamaño máximo del conjunto de entrenamiento (como en TimeSeriesSplit)
    gap : int
        Número de epochs de gap entre train y test (default: 0)
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[float] = None,
        max_train_size: Optional[int] = None,
        gap: int = 0,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.max_train_size = max_train_size
        self.gap = gap

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Genera índices para train/test en cada fold usando TimeSeriesSplit por grupo.

        Parameters
        ----------
        X : pd.DataFrame
            Features (debe tener columnas 'subject_core' y 'epoch_index' o 'epoch_time_start')
        y : pd.Series, optional
            Etiquetas
        groups : pd.Series, optional
            Grupos (si no se proporciona, se usa 'subject_core' de X)

        Yields
        ------
        train_indices : np.ndarray
            Índices para entrenamiento
        test_indices : np.ndarray
            Índices para test
        """
        if groups is None:
            if "subject_core" not in X.columns:
                raise ValueError(
                    "X debe tener columna 'subject_core' o proporcionar 'groups'"
                )
            groups = X["subject_core"]

        # Determinar columna de orden temporal
        if "epoch_index" in X.columns:
            time_col = "epoch_index"
        elif "epoch_time_start" in X.columns:
            time_col = "epoch_time_start"
        else:
            raise ValueError(
                "X debe tener columna 'epoch_index' o 'epoch_time_start' para orden temporal"
            )

        # Obtener grupos únicos y ordenarlos
        unique_groups = np.sort(groups.unique())

        # Precalcular índices ordenados por grupo
        group_sorted_indices: dict[str, np.ndarray] = {}
        for group in unique_groups:
            group_mask = groups == group
            group_indices = np.where(group_mask)[0]
            if len(group_indices) < 2:
                continue
            group_df = X.loc[group_indices].copy()
            group_df = group_df.sort_values(time_col)
            group_sorted_indices[group] = group_df.index.values

        # Ruta A: test_size explícito → construir ventanas temporales con tamaño fijo
        if self.test_size is not None:
            # Determinar cuántos folds son posibles respetando el tamaño de test y la cantidad de datos
            per_group_folds = []
            for sorted_indices in group_sorted_indices.values():
                n_group = len(sorted_indices)
                test_len = max(1, int(np.ceil(n_group * self.test_size)))
                valid_folds = 0
                for fold_idx in range(self.n_splits):
                    start = n_group - (fold_idx + 1) * test_len
                    if start <= 0:
                        continue
                    train_end = max(0, start - self.gap)
                    if train_end <= 0:
                        continue
                    valid_folds += 1
                per_group_folds.append(valid_folds)

            max_global_folds = min(
                self.n_splits, max(per_group_folds) if per_group_folds else 0
            )

            for global_fold in range(max_global_folds):
                train_indices = []
                test_indices = []

                for sorted_indices in group_sorted_indices.values():
                    n_group = len(sorted_indices)
                    test_len = max(1, int(np.ceil(n_group * self.test_size)))
                    start = n_group - (global_fold + 1) * test_len
                    if start <= 0:
                        continue
                    train_end = max(0, start - self.gap)
                    if train_end <= 0:
                        continue

                    train_indices.extend(sorted_indices[:train_end])
                    test_indices.extend(sorted_indices[start : start + test_len])

                if len(train_indices) > 0 and len(test_indices) > 0:
                    yield (
                        np.array(train_indices, dtype=int),
                        np.array(test_indices, dtype=int),
                    )
        else:
            # Ruta B: comportamiento anterior (TimeSeriesSplit) cuando no se especifica test_size
            tscv = TimeSeriesSplit(
                n_splits=self.n_splits,
                max_train_size=self.max_train_size,
                test_size=None,
            )

            for global_fold in range(self.n_splits):
                train_indices = []
                test_indices = []

                for sorted_indices in group_sorted_indices.values():
                    n_group = len(sorted_indices)
                    group_array = np.arange(n_group)
                    try:
                        group_splits = list(tscv.split(group_array))
                        group_fold = global_fold % len(group_splits)
                        train_group_idx, test_group_idx = group_splits[group_fold]

                        if self.gap > 0 and len(train_group_idx) > 0:
                            max_train_idx = train_group_idx[-1]
                            test_group_idx = test_group_idx[
                                test_group_idx > max_train_idx + self.gap
                            ]

                        train_indices.extend(sorted_indices[train_group_idx])
                        if len(test_group_idx) > 0:
                            test_indices.extend(sorted_indices[test_group_idx])
                    except ValueError:
                        continue

                if len(train_indices) > 0 and len(test_indices) > 0:
                    yield (
                        np.array(train_indices, dtype=int),
                        np.array(test_indices, dtype=int),
                    )

    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> int:
        """Retorna el número de folds."""
        if X is None or groups is None or self.test_size is None:
            return self.n_splits

        # Calcular folds efectivos según tamaño de test y longitud de cada grupo
        if "epoch_index" in X.columns:
            time_col = "epoch_index"
        elif "epoch_time_start" in X.columns:
            time_col = "epoch_time_start"
        else:
            return self.n_splits

        groups_series = groups
        unique_groups = np.sort(groups_series.unique())

        max_folds = 0
        for group in unique_groups:
            group_mask = groups_series == group
            group_indices = np.where(group_mask)[0]
            if len(group_indices) < 2:
                continue
            group_df = X.loc[group_indices].copy()
            group_df = group_df.sort_values(time_col)
            n_group = len(group_df)
            test_len = max(1, int(np.ceil(n_group * self.test_size)))
            valid_folds = 0
            for fold_idx in range(self.n_splits):
                start = n_group - (fold_idx + 1) * test_len
                if start <= 0:
                    continue
                train_end = max(0, start - self.gap)
                if train_end <= 0:
                    continue
                valid_folds += 1
            max_folds = max(max_folds, valid_folds)

        return min(self.n_splits, max_folds if max_folds > 0 else self.n_splits)


class SubjectTimeSeriesSplit(BaseCrossValidator):
    """Cross-validator que divide por sujetos (subject-level k-fold).

    Divide subject_cores en folds respetando que todos los epochs de un mismo
    subject_core van al mismo conjunto. Esto evita data leakage entre sujetos,
    que es el enfoque estándar en sleep staging.

    No divide temporalmente dentro de cada sujeto porque todos los epochs de
    un subject_core van al mismo conjunto (train o test).

    Parameters
    ----------
    n_splits : int
        Número de folds
    test_size : float
        Proporción del test set (sobre subject_cores)
    max_train_size : int, optional
        No usado actualmente (mantenido para compatibilidad)
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        max_train_size: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.max_train_size = max_train_size

    def split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Genera índices para train/test en cada fold.

        Parameters
        ----------
        X : pd.DataFrame
            Features (debe tener columna 'subject_core')
        y : pd.Series, optional
            Etiquetas
        groups : pd.Series, optional
            Grupos (si no se proporciona, se usa 'subject_core' de X)

        Yields
        ------
        train_indices : np.ndarray
            Índices para entrenamiento
        test_indices : np.ndarray
            Índices para test
        """
        if groups is None:
            if "subject_core" not in X.columns:
                raise ValueError(
                    "X debe tener columna 'subject_core' o proporcionar 'groups'"
                )
            groups = X["subject_core"]

        # Obtener subject_cores únicos
        unique_cores = groups.unique()
        n_cores = len(unique_cores)

        # Calcular tamaño del test
        n_test_cores = max(1, int(n_cores * self.test_size))

        # Generar folds usando Generator para evitar afectar estado global
        rng = np.random.default_rng(42)
        shuffled_cores = rng.permutation(unique_cores)

        for fold in range(self.n_splits):
            # Rotar qué sujetos van a test en cada fold
            test_start = (fold * n_test_cores) % n_cores
            test_end = test_start + n_test_cores

            if test_end > n_cores:
                # Wrap around
                test_cores = set(shuffled_cores[test_start:])
                test_cores.update(shuffled_cores[: test_end - n_cores])
            else:
                test_cores = set(shuffled_cores[test_start:test_end])

            train_cores = set(unique_cores) - test_cores

            # Obtener índices directamente - todos los epochs de un subject_core
            # van al mismo conjunto, así que no necesitamos ordenamiento temporal
            train_mask = groups.isin(train_cores)
            test_mask = groups.isin(test_cores)

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            yield np.array(train_indices, dtype=int), np.array(test_indices, dtype=int)

    def get_n_splits(
        self,
        X: Optional[pd.DataFrame] = None,
        y: Optional[pd.Series] = None,
        groups: Optional[pd.Series] = None,
    ) -> int:
        """Retorna el número de folds."""
        return self.n_splits
