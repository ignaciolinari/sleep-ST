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

        # Usar TimeSeriesSplit dentro de cada grupo
        # Calcular n_splits por grupo basado en test_size si se especifica
        if self.test_size is not None:
            # Estimar n_splits basado en test_size
            # Si test_size=0.2, necesitamos ~5 folds para cubrir toda la serie
            estimated_n_splits = max(2, int(1.0 / self.test_size))
            n_splits_per_group = min(self.n_splits, estimated_n_splits)
        else:
            n_splits_per_group = self.n_splits

        # Crear TimeSeriesSplit
        tscv = TimeSeriesSplit(
            n_splits=n_splits_per_group,
            max_train_size=self.max_train_size,
            test_size=None,  # sklearn no tiene test_size, lo calculamos nosotros
        )

        # Para cada fold global, combinar splits de todos los grupos
        for global_fold in range(self.n_splits):
            train_indices = []
            test_indices = []

            for group in unique_groups:
                group_mask = groups == group
                group_indices = np.where(group_mask)[0]

                if len(group_indices) < 2:
                    continue

                # Ordenar por tiempo dentro del grupo
                group_df = X.loc[group_indices].copy()
                group_df = group_df.sort_values(time_col)
                sorted_indices = group_df.index.values
                n_group = len(sorted_indices)

                # Aplicar TimeSeriesSplit dentro del grupo
                # Usar el fold correspondiente (wrap around si es necesario)
                group_fold = global_fold % n_splits_per_group

                # Crear array temporal para TimeSeriesSplit
                group_array = np.arange(n_group)

                try:
                    # Obtener splits para este grupo
                    group_splits = list(tscv.split(group_array))
                    if group_fold < len(group_splits):
                        train_group_idx, test_group_idx = group_splits[group_fold]

                        # Aplicar gap si está especificado
                        if self.gap > 0 and len(train_group_idx) > 0:
                            max_train_idx = train_group_idx[-1]
                            test_group_idx = test_group_idx[
                                test_group_idx > max_train_idx + self.gap
                            ]

                        # Mapear de índices del grupo a índices globales
                        train_indices.extend(sorted_indices[train_group_idx])
                        if len(test_group_idx) > 0:
                            test_indices.extend(sorted_indices[test_group_idx])
                except ValueError:
                    # Si el grupo es muy pequeño, saltarlo
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
        return self.n_splits


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

        # Generar folds
        np.random.seed(42)  # Para reproducibilidad
        shuffled_cores = np.random.permutation(unique_cores)

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
