"""Modelo Random Forest para clasificación de estadios de sueño."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    max_features: Optional[str | int | float] = "sqrt",
    class_weight: Optional[str] = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    """Entrena un modelo Random Forest.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento
    y_train : pd.Series
        Etiquetas de entrenamiento
    n_estimators : int
        Número de árboles
    max_depth : int, optional
        Profundidad máxima de los árboles
    min_samples_split : int
        Mínimo de muestras para dividir un nodo
    min_samples_leaf : int
        Mínimo de muestras en una hoja
    class_weight : str, optional
        Estrategia de pesos de clases ('balanced' para manejar desbalance)
    random_state : int
        Semilla aleatoria
    n_jobs : int
        Número de jobs paralelos (-1 = todos los cores)

    Returns
    -------
    RandomForestClassifier
        Modelo entrenado
    """
    logging.info("=" * 60)
    logging.info("ETAPA: ENTRENAMIENTO - Random Forest")
    logging.info("=" * 60)
    logging.info("Configuración del modelo:")
    logging.info(f"  - Número de árboles: {n_estimators}")
    logging.info(f"  - Profundidad máxima: {max_depth if max_depth else 'Sin límite'}")
    logging.info(f"  - Mínimo muestras para split: {min_samples_split}")
    logging.info(f"  - Mínimo muestras en hoja: {min_samples_leaf}")
    logging.info(f"  - Max features: {max_features}")
    logging.info(f"  - Pesos de clases: {class_weight}")
    logging.info(f"  - Jobs paralelos: {n_jobs}")
    logging.info("Datos de entrenamiento:")
    logging.info(f"  - Muestras: {len(X_train)}")
    logging.info(f"  - Features: {X_train.shape[1]}")
    logging.info(f"  - Distribución de clases:\n{y_train.value_counts().sort_index()}")
    logging.info("Iniciando entrenamiento...")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model.fit(X_train, y_train)
    logging.info(
        f"✓ Entrenamiento completado: Random Forest con {len(X_train)} muestras"
    )
    logging.info("=" * 60)

    return model
