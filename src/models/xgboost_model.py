"""Modelo XGBoost para clasificación de estadios de sueño."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.01,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    n_jobs: int = -1,
    scale_pos_weight: Optional[float] = None,
) -> xgb.XGBClassifier:
    """Entrena un modelo XGBoost.

    Parameters
    ----------
    X_train : pd.DataFrame
        Features de entrenamiento
    y_train : pd.Series
        Etiquetas de entrenamiento
    n_estimators : int
        Número de árboles
    max_depth : int
        Profundidad máxima de los árboles
    learning_rate : float
        Tasa de aprendizaje
    subsample : float
        Fracción de muestras para entrenar cada árbol (default: 0.8)
    colsample_bytree : float
        Fracción de features para cada árbol (default: 0.8)
    reg_alpha : float
        Regularización L1 (default: 0.01)
    reg_lambda : float
        Regularización L2 (default: 1.0)
    random_state : int
        Semilla aleatoria
    n_jobs : int
        Número de jobs paralelos
    scale_pos_weight : float, optional
        Controla el balance de clases. Si None, se calcula automáticamente.

    Returns
    -------
    xgb.XGBClassifier
        Modelo entrenado
    """
    logging.info("=" * 60)
    logging.info("ETAPA: ENTRENAMIENTO - XGBoost")
    logging.info("=" * 60)
    logging.info("Configuración del modelo:")
    logging.info(f"  - Número de árboles: {n_estimators}")
    logging.info(f"  - Profundidad máxima: {max_depth}")
    logging.info(f"  - Tasa de aprendizaje: {learning_rate}")
    logging.info(f"  - Subsample: {subsample}")
    logging.info(f"  - Colsample bytree: {colsample_bytree}")
    logging.info(f"  - Regularización L1 (alpha): {reg_alpha}")
    logging.info(f"  - Regularización L2 (lambda): {reg_lambda}")
    logging.info(f"  - Jobs paralelos: {n_jobs}")
    logging.info("Datos de entrenamiento:")
    logging.info(f"  - Muestras: {len(X_train)}")
    logging.info(f"  - Features: {X_train.shape[1]}")
    class_counts = y_train.value_counts().sort_index()
    logging.info(f"  - Distribución de clases:\n{class_counts}")
    logging.info("Codificando etiquetas...")

    # Codificar etiquetas
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    logging.info(f"Etiquetas codificadas. Clases: {le.classes_}")

    # Calcular sample_weight para balancear clases
    if scale_pos_weight is None:
        class_weights = {}
        total_samples = len(y_train_encoded)
        n_classes = len(le.classes_)

        for class_idx, class_name in enumerate(le.classes_):
            class_count = class_counts.get(class_name, 0)
            if class_count > 0:
                class_weights[class_idx] = total_samples / (n_classes * class_count)
            else:
                class_weights[class_idx] = 1.0

        sample_weight = np.array([class_weights[y] for y in y_train_encoded])
        logging.info("  - Usando sample_weight para balancear clases automáticamente")
        logging.info(
            f"  - Pesos por clase: "
            f"{dict(zip(le.classes_, [class_weights[i] for i in range(len(le.classes_))]))}"
        )
    else:
        sample_weight = None
        logging.info(f"  - scale_pos_weight: {scale_pos_weight}")

    logging.info("Iniciando entrenamiento...")
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="mlogloss",
    )

    if sample_weight is not None:
        model.fit(X_train, y_train_encoded, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train_encoded)

    # Guardar LabelEncoder y clases originales
    # Nota: XGBoost ya tiene classes_ como propiedad de solo lectura después de fit()
    # Usamos label_encoder_ para decodificación en evaluate_model y pipeline
    model.label_encoder_ = le
    model.original_classes_ = le.classes_  # Alias para compatibilidad

    logging.info(f"✓ Entrenamiento completado: XGBoost con {len(X_train)} muestras")
    logging.info("=" * 60)

    return model
