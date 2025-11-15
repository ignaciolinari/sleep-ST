"""Entrenamiento y evaluación de modelos para clasificación de estadios de sueño.

Este módulo implementa pipelines de ML para clasificar estadios de sueño
usando Random Forest y otras técnicas, optimizadas para setups de pocos canales.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from src.features import extract_features_from_session
from src.crossval import SubjectTimeSeriesSplit, GroupTimeSeriesSplit

# Estadios en orden estándar
STAGE_ORDER = ["W", "N1", "N2", "N3", "REM"]


def prepare_features_dataset(
    manifest_path: Path | str,
    limit: Optional[int] = None,
    epoch_length: float = 30.0,
    sfreq: Optional[float] = None,
) -> pd.DataFrame:
    """Prepara dataset de features desde múltiples sesiones.

    Parameters
    ----------
    manifest_path : Path | str
        Ruta al manifest CSV con sesiones procesadas
    limit : int, optional
        Limitar número de sesiones a procesar (para pruebas)
    epoch_length : float
        Duración de cada epoch en segundos
    sfreq : float, optional
        Frecuencia de muestreo objetivo

    Returns
    -------
    pd.DataFrame
        DataFrame con features de todos los epochs de todas las sesiones
    """
    manifest = pd.read_csv(manifest_path)

    # Filtrar solo sesiones OK
    manifest_ok = manifest[manifest["status"] == "ok"].copy()

    if limit:
        manifest_ok = manifest_ok.head(limit)
        logging.info(f"Procesando solo {len(manifest_ok)} sesiones (modo limit)")

    all_features = []

    for idx, row in manifest_ok.iterrows():
        # Intentar usar las rutas del manifest primero (si existen)
        psg_path_str = row.get("psg_trimmed_path", "")
        hyp_path_str = row.get("hypnogram_trimmed_path", "")

        # Construir Paths solo si las rutas no están vacías
        psg_path = Path(psg_path_str) if psg_path_str else None
        hyp_path = Path(hyp_path_str) if hyp_path_str else None

        # Si las rutas del manifest no existen o están vacías, construir rutas relativas
        if (
            not psg_path
            or not hyp_path
            or not psg_path.exists()
            or not hyp_path.exists()
        ):
            # Construir rutas relativas basadas en subject_id, subset, version
            manifest_dir = Path(manifest_path).parent
            subject_id = row["subject_id"]
            subset = row.get("subset", "sleep-cassette")
            version = row.get("version", "1.0.0")

            psg_path = (
                manifest_dir
                / "sleep_trimmed"
                / "psg"
                / f"{subject_id}_{subset}_{version}_trimmed_raw.fif"
            )
            hyp_path = (
                manifest_dir
                / "sleep_trimmed"
                / "hypnograms"
                / f"{subject_id}_{subset}_{version}_trimmed_annotations.csv"
            )

        if not psg_path.exists() or not hyp_path.exists():
            logging.warning(
                f"Archivos faltantes para {row['subject_id']} (buscado en {psg_path}), saltando"
            )
            continue

        try:
            features_df = extract_features_from_session(
                psg_path,
                hyp_path,
                epoch_length=epoch_length,
                sfreq=sfreq,
            )

            if not features_df.empty:
                features_df["subject_id"] = row["subject_id"]
                # Extraer subject_core (primeros 5 caracteres) para agrupar noches del mismo sujeto
                features_df["subject_core"] = row["subject_id"][:5]
                features_df["session_idx"] = idx
                # Asegurar que los epochs mantengan orden temporal dentro de la sesión
                # (ya están ordenados por epoch_time_start en extract_features_from_session)
                all_features.append(features_df)
                logging.info(
                    f"Extraídas {len(features_df)} epochs de {row['subject_id']}"
                )
        except Exception as e:
            logging.exception(f"Error procesando {row['subject_id']}: {e}")
            continue

    if not all_features:
        raise ValueError("No se pudieron extraer features de ninguna sesión")

    combined = pd.concat(all_features, ignore_index=True)
    # Ordenar por sujeto y tiempo para mantener orden temporal
    combined = combined.sort_values(
        ["subject_core", "subject_id", "epoch_time_start"]
    ).reset_index(drop=True)
    logging.info(
        f"Dataset completo: {len(combined)} epochs de {combined['subject_id'].nunique()} sujetos"
    )

    return combined


def prepare_train_test_split(
    features_df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: int = 42,
    stratify_by: Optional[str] = "subject_core",
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """Divide dataset en train/test/val respetando sujetos.

    IMPORTANTE: Divide por subject_core (no subject_id) para evitar data leakage.
    Todas las noches de un mismo sujeto van al mismo conjunto.

    NOTA SOBRE TAMAÑO DEL DATASET:
    - Si tienes pocos subject_cores (<10), considera usar cross-validation
      en lugar de un split simple para obtener estimaciones más confiables.
    - Con muy pocos sujetos, el conjunto de test puede ser muy pequeño.
    - Alternativa: usar leave-one-subject-out cross-validation.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame con features
    test_size : float
        Proporción del test set (sobre subject_cores)
    val_size : float, optional
        Proporción del validation set (sobre subject_cores). Si None, no se crea val.
    random_state : int
        Semilla aleatoria
    stratify_by : str, optional
        Columna para estratificar (default: 'subject_core' para evitar data leakage)

    Returns
    -------
    train_df : pd.DataFrame
        DataFrame de entrenamiento
    test_df : pd.DataFrame
        DataFrame de test
    val_df : pd.DataFrame, optional
        DataFrame de validación (si val_size está especificado)
    """
    if stratify_by and stratify_by in features_df.columns:
        # Dividir por subject_core para evitar data leakage
        # Todas las noches de un mismo sujeto van al mismo conjunto
        subject_cores = features_df[stratify_by].unique()
        n_cores = len(subject_cores)

        np.random.seed(random_state)
        shuffled_cores = np.random.permutation(subject_cores)

        # Calcular tamaños
        n_test_cores = max(1, int(n_cores * test_size))
        n_val_cores = max(1, int(n_cores * val_size)) if val_size is not None else 0
        n_train_cores = n_cores - n_test_cores - n_val_cores

        # Dividir subject_cores
        test_cores = set(shuffled_cores[:n_test_cores])
        val_cores = (
            set(shuffled_cores[n_test_cores : n_test_cores + n_val_cores])
            if val_size
            else set()
        )
        train_cores = set(shuffled_cores[n_test_cores + n_val_cores :])

        # Asignar epochs según subject_core
        train_df = features_df[features_df[stratify_by].isin(train_cores)]
        test_df = features_df[features_df[stratify_by].isin(test_cores)]
        val_df = (
            features_df[features_df[stratify_by].isin(val_cores)] if val_size else None
        )

        # Calcular porcentajes de epochs
        total_epochs = len(features_df)
        train_pct = (len(train_df) / total_epochs * 100) if total_epochs > 0 else 0
        test_pct = (len(test_df) / total_epochs * 100) if total_epochs > 0 else 0
        val_pct = (
            (len(val_df) / total_epochs * 100)
            if val_df is not None and total_epochs > 0
            else 0
        )

        # Calcular epochs promedio por subject_core
        epochs_per_core = total_epochs / n_cores if n_cores > 0 else 0

        logging.info("División del dataset:")
        logging.info(f"  Total de epochs: {total_epochs}")
        logging.info(f"  Total de subject_cores: {n_cores}")
        logging.info(f"  Epochs promedio por subject_core: {epochs_per_core:.1f}")
        logging.info("")
        logging.info(
            f"Train: {len(train_df)} epochs ({train_pct:.1f}%) de {train_df[stratify_by].nunique()} subject_cores "
            f"({train_df['subject_id'].nunique()} sesiones)"
        )
        logging.info(
            f"Test:  {len(test_df)} epochs ({test_pct:.1f}%) de {test_df[stratify_by].nunique()} subject_cores "
            f"({test_df['subject_id'].nunique()} sesiones)"
        )
        if val_df is not None:
            logging.info(
                f"Val:   {len(val_df)} epochs ({val_pct:.1f}%) de {val_df[stratify_by].nunique()} subject_cores "
                f"({val_df['subject_id'].nunique()} sesiones)"
            )
        logging.info("")
        logging.info("Configuración de división:")
        logging.info(
            f"  test_size: {test_size} ({test_size*100:.0f}% de subject_cores)"
        )
        if val_size is not None:
            logging.info(
                f"  val_size: {val_size} ({val_size*100:.0f}% de subject_cores)"
            )
            logging.info(
                f"  train_size: {1 - test_size - val_size} ({(1 - test_size - val_size)*100:.0f}% de subject_cores)"
            )
        else:
            logging.info("  val_size: None (no se crea conjunto de validación)")
            logging.info(
                f"  train_size: {1 - test_size} ({(1 - test_size)*100:.0f}% de subject_cores)"
            )

        # Advertencias si hay pocos sujetos
        if n_cores < 10:
            logging.warning(
                f"⚠️  ADVERTENCIA: Solo hay {n_cores} subject_cores. "
                f"Con test_size={test_size}, el conjunto de test tendrá solo {n_test_cores} sujeto(s). "
                f"Considera usar cross-validation (--cross-validate) o leave-one-subject-out."
            )
        elif n_cores < 20:
            logging.info(
                f"ℹ️  INFO: Tienes {n_cores} subject_cores. "
                f"El conjunto de test tendrá {n_test_cores} sujeto(s). "
                f"Considera usar cross-validation para estimaciones más robustas."
            )
        else:
            logging.info(
                f"✓ Tienes {n_cores} subject_cores, suficiente para un split confiable. "
                f"Test: {n_test_cores} sujetos, Train: {n_train_cores} sujetos."
            )

        if n_test_cores < 2:
            logging.warning(
                f"⚠️  ADVERTENCIA: El conjunto de test tiene solo {n_test_cores} subject_core(s). "
                f"Esto puede dar estimaciones de performance poco confiables. "
                f"Considera usar cross-validation con más folds."
            )
        if val_size is not None and n_val_cores < 2:
            logging.warning(
                f"⚠️  ADVERTENCIA: El conjunto de validación tiene solo {n_val_cores} subject_core(s). "
                f"Considera reducir val_size o usar cross-validation."
            )

        # Verificar que no hay overlap
        train_set = set(train_df[stratify_by].unique())
        test_set = set(test_df[stratify_by].unique())
        val_set = set(val_df[stratify_by].unique()) if val_df is not None else set()

        if train_set & test_set:
            logging.warning("⚠️  OVERLAP detectado entre train y test!")
        if train_set & val_set:
            logging.warning("⚠️  OVERLAP detectado entre train y val!")
        if test_set & val_set:
            logging.warning("⚠️  OVERLAP detectado entre test y val!")
    else:
        # División simple (no recomendado para sleep staging)
        logging.warning(
            "División sin estratificación por sujeto - riesgo de data leakage"
        )
        if val_size:
            train_df, temp_df = train_test_split(
                features_df,
                test_size=test_size + val_size,
                random_state=random_state,
                stratify=features_df["stage"]
                if "stage" in features_df.columns
                else None,
            )
            val_size_adj = val_size / (test_size + val_size)
            test_df, val_df = train_test_split(
                temp_df,
                test_size=val_size_adj,
                random_state=random_state,
                stratify=temp_df["stage"] if "stage" in temp_df.columns else None,
            )
        else:
            train_df, test_df = train_test_split(
                features_df,
                test_size=test_size,
                random_state=random_state,
                stratify=features_df["stage"]
                if "stage" in features_df.columns
                else None,
            )
            val_df = None

    return train_df, test_df, val_df


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: Optional[int] = None,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
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
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model.fit(X_train, y_train)
    logging.info(
        f"✓ Entrenamiento completado: Random Forest entrenado con {len(X_train)} muestras"
    )
    logging.info("=" * 60)

    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    n_jobs: int = -1,
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
    random_state : int
        Semilla aleatoria
    n_jobs : int
        Número de jobs paralelos

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
    logging.info(f"  - Jobs paralelos: {n_jobs}")
    logging.info("Datos de entrenamiento:")
    logging.info(f"  - Muestras: {len(X_train)}")
    logging.info(f"  - Features: {X_train.shape[1]}")
    logging.info(f"  - Distribución de clases:\n{y_train.value_counts().sort_index()}")
    logging.info("Codificando etiquetas...")

    # Codificar etiquetas
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    logging.info(f"Etiquetas codificadas. Clases: {le.classes_}")

    logging.info("Iniciando entrenamiento...")
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="mlogloss",
    )

    model.fit(X_train, y_train_encoded)
    model.classes_ = le.classes_  # Guardar clases originales

    logging.info(
        f"✓ Entrenamiento completado: XGBoost entrenado con {len(X_train)} muestras"
    )
    logging.info("=" * 60)

    return model


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv: SubjectTimeSeriesSplit | GroupTimeSeriesSplit,
    scoring: str = "accuracy",
    X_full: Optional[pd.DataFrame] = None,
) -> dict:
    """Realiza cross-validation respetando grupos (subject-level).

    Parameters
    ----------
    model
        Modelo a evaluar (debe tener métodos fit y predict)
    X : pd.DataFrame
        Features (sin metadata temporal)
    y : pd.Series
        Etiquetas
    groups : pd.Series
        Grupos (subject_core)
    cv : SubjectTimeSeriesSplit | GroupTimeSeriesSplit
        Cross-validator. SubjectTimeSeriesSplit solo necesita X, mientras que
        GroupTimeSeriesSplit requiere X_full con columnas temporales.
    scoring : str
        Métrica a usar ('accuracy', 'f1_macro', 'f1_weighted', 'kappa')
    X_full : pd.DataFrame, optional
        DataFrame completo con columnas temporales (epoch_time_start, epoch_index).
        Solo necesario si se usa GroupTimeSeriesSplit. Para SubjectTimeSeriesSplit
        se puede usar None y se usará X directamente.

    Returns
    -------
    dict
        Diccionario con métricas de CV
    """
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        f1_score,
    )

    cv_scores = []
    fold_metrics = []

    # Para SubjectTimeSeriesSplit no necesitamos columnas temporales
    # Para GroupTimeSeriesSplit sí las necesitamos
    # Usar X_full si está disponible y es necesario, sino X
    if isinstance(cv, GroupTimeSeriesSplit) and X_full is not None:
        X_for_split = X_full
    else:
        X_for_split = X

    n_splits = cv.get_n_splits(X_for_split, y, groups)
    logging.info("=" * 60)
    logging.info(f"ETAPA: CROSS-VALIDATION ({n_splits} folds)")
    logging.info("=" * 60)
    logging.info(f"Métrica de evaluación: {scoring}")
    logging.info(f"Total de muestras: {len(X)}")
    logging.info(f"Total de grupos (subject_cores): {groups.nunique()}")

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_for_split, y, groups)):
        logging.info(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
        logging.info(f"  Muestras de entrenamiento: {len(train_idx)}")
        logging.info(f"  Muestras de validación: {len(test_idx)}")
        logging.info(f"  Grupos en train: {groups.iloc[train_idx].nunique()}")
        logging.info(f"  Grupos en val: {groups.iloc[test_idx].nunique()}")

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]

        logging.info(f"  Entrenando modelo para fold {fold_idx + 1}...")
        # Entrenar modelo
        model_fold = type(model)(**model.get_params())

        # Para XGBoost, necesitamos codificar las etiquetas
        is_xgboost = isinstance(model_fold, xgb.XGBClassifier)
        if is_xgboost:
            le_fold = LabelEncoder()
            y_train_fold_encoded = le_fold.fit_transform(y_train_fold)
            model_fold.fit(X_train_fold, y_train_fold_encoded)
            model_fold.classes_ = le_fold.classes_  # Guardar clases para decodificación
        else:
            model_fold.fit(X_train_fold, y_train_fold)

        logging.info("  Prediciendo en conjunto de validación...")
        # Predecir
        y_pred_fold = model_fold.predict(X_test_fold)

        # Decodificar predicciones de XGBoost si es necesario
        if (
            is_xgboost
            and hasattr(model_fold, "classes_")
            and len(y_pred_fold) > 0
            and isinstance(y_pred_fold[0], (int, np.integer))
        ):
            # Validar que los índices estén en rango válido
            if np.any(y_pred_fold < 0) or np.any(
                y_pred_fold >= len(model_fold.classes_)
            ):
                invalid_indices = np.where(
                    (y_pred_fold < 0) | (y_pred_fold >= len(model_fold.classes_))
                )[0]
                logging.warning(
                    f"Fold {fold_idx + 1}: {len(invalid_indices)} predicciones con índices fuera de rango. "
                    f"Rango válido: [0, {len(model_fold.classes_)-1}], encontrados: {np.unique(y_pred_fold[invalid_indices])}"
                )
                # Clampear índices al rango válido
                y_pred_fold = np.clip(y_pred_fold, 0, len(model_fold.classes_) - 1)
            y_pred_fold = model_fold.classes_[y_pred_fold]

        logging.info("  Calculando métricas...")
        # Calcular métricas
        if scoring == "accuracy":
            score = accuracy_score(y_test_fold, y_pred_fold)
        elif scoring == "f1_macro":
            score = f1_score(y_test_fold, y_pred_fold, average="macro")
        elif scoring == "f1_weighted":
            score = f1_score(y_test_fold, y_pred_fold, average="weighted")
        elif scoring == "kappa":
            score = cohen_kappa_score(y_test_fold, y_pred_fold)
        else:
            raise ValueError(f"Métrica desconocida: {scoring}")

        logging.info(f"  ✓ Fold {fold_idx + 1} completado - {scoring}: {score:.4f}")
        cv_scores.append(score)
        fold_metrics.append(
            {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "score": score,
            }
        )

    logging.info("\n" + "=" * 60)
    logging.info("RESUMEN DE CROSS-VALIDATION")
    logging.info("=" * 60)
    logging.info(
        f"Score promedio ({scoring}): {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})"
    )
    logging.info(f"Scores por fold: {[f'{s:.4f}' for s in cv_scores]}")
    logging.info("=" * 60)

    return {
        "mean_score": np.mean(cv_scores),
        "std_score": np.std(cv_scores),
        "scores": cv_scores,
        "fold_metrics": fold_metrics,
    }


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    stage_order: Optional[list[str]] = None,
    dataset_name: str = "TEST",
) -> dict:
    """Evalúa un modelo y retorna métricas.

    Parameters
    ----------
    model
        Modelo entrenado (sklearn o XGBoost)
    X_test : pd.DataFrame
        Features de test
    y_test : pd.Series
        Etiquetas de test
    stage_order : list[str], optional
        Orden de los estadios para el reporte
    dataset_name : str
        Nombre del conjunto de datos ("TEST", "VALIDATION", etc.)

    Returns
    -------
    dict
        Diccionario con métricas de evaluación
    """
    logging.info("=" * 60)
    logging.info(f"ETAPA: EVALUACIÓN EN {dataset_name}")
    logging.info("=" * 60)
    logging.info(f"Muestras de {dataset_name.lower()}: {len(X_test)}")
    logging.info(
        f"Distribución de clases en {dataset_name.lower()}:\n{y_test.value_counts().sort_index()}"
    )
    logging.info("Generando predicciones...")

    y_pred = model.predict(X_test)

    # Si es XGBoost, puede necesitar decodificación
    if (
        hasattr(model, "classes_")
        and len(y_pred) > 0
        and isinstance(y_pred[0], (int, np.integer))
    ):
        logging.info("Decodificando predicciones (XGBoost)...")
        # Validar que los índices estén en rango válido
        if np.any(y_pred < 0) or np.any(y_pred >= len(model.classes_)):
            invalid_indices = np.where((y_pred < 0) | (y_pred >= len(model.classes_)))[
                0
            ]
            logging.warning(
                f"Advertencia: {len(invalid_indices)} predicciones con índices fuera de rango. "
                f"Rango válido: [0, {len(model.classes_)-1}], encontrados: {np.unique(y_pred[invalid_indices])}"
            )
            # Clampear índices al rango válido
            y_pred = np.clip(y_pred, 0, len(model.classes_) - 1)
        y_pred = model.classes_[y_pred]

    logging.info("Calculando métricas de evaluación...")
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    # F1 por clase
    f1_per_class = f1_score(y_test, y_pred, average=None, labels=stage_order)
    f1_dict = {stage: float(score) for stage, score in zip(stage_order, f1_per_class)}

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=stage_order)

    # Reporte de clasificación
    report = classification_report(y_test, y_pred, labels=stage_order, output_dict=True)

    logging.info("✓ Evaluación completada")
    logging.info(f"  Accuracy: {accuracy:.4f}")
    logging.info(f"  Cohen's Kappa: {kappa:.4f}")
    logging.info(f"  F1-score (macro): {f1_macro:.4f}")
    logging.info(f"  F1-score (weighted): {f1_weighted:.4f}")
    logging.info("=" * 60)

    metrics = {
        "accuracy": float(accuracy),
        "kappa": float(kappa),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_per_class": f1_dict,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    return metrics


def print_evaluation_report(metrics: dict, stage_order: list[str]) -> None:
    """Imprime reporte de evaluación formateado.

    Parameters
    ----------
    metrics : dict
        Diccionario con métricas de evaluación
    stage_order : list[str]
        Orden de los estadios
    """
    print("\n" + "=" * 60)
    print("REPORTE DE EVALUACIÓN")
    print("=" * 60)

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Cohen's Kappa: {metrics['kappa']:.4f}")
    print(f"F1-score (macro): {metrics['f1_macro']:.4f}")
    print(f"F1-score (weighted): {metrics['f1_weighted']:.4f}")

    print("\nF1-score por estadio:")
    for stage in stage_order:
        f1 = metrics["f1_per_class"].get(stage, 0.0)
        print(f"  {stage}: {f1:.4f}")

    print("\nMatriz de confusión:")
    cm = np.array(metrics["confusion_matrix"])
    print("      ", " ".join(f"{s:>5}" for s in stage_order))
    for i, stage in enumerate(stage_order):
        row_str = f"{stage:>5} "
        for j in range(len(stage_order)):
            row_str += f"{cm[i, j]:>5} "
        print(row_str)

    print("\nReporte detallado por clase:")
    report = metrics["classification_report"]
    for stage in stage_order:
        if stage in report:
            stage_metrics = report[stage]
            print(f"\n{stage}:")
            print(f"  Precision: {stage_metrics['precision']:.4f}")
            print(f"  Recall: {stage_metrics['recall']:.4f}")
            print(f"  F1-score: {stage_metrics['f1-score']:.4f}")
            print(f"  Support: {stage_metrics['support']}")


def save_metrics(
    metrics: dict,
    path: Path | str,
    model_type: str = "unknown",
    **model_params,
) -> None:
    """Guarda métricas de evaluación en formato JSON.

    Parameters
    ----------
    metrics : dict
        Diccionario con métricas de evaluación
    path : Path | str
        Ruta donde guardar las métricas
    model_type : str
        Tipo de modelo
    **model_params
        Parámetros del modelo para guardar también
    """
    output = {
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "model_params": model_params,
        "metrics": {
            "accuracy": metrics.get("accuracy"),
            "kappa": metrics.get("kappa"),
            "f1_macro": metrics.get("f1_macro"),
            "f1_weighted": metrics.get("f1_weighted"),
            "f1_per_class": metrics.get("f1_per_class", {}),
        },
        "confusion_matrix": metrics.get("confusion_matrix", []),
        "classification_report": metrics.get("classification_report", {}),
    }

    # Agregar resultados de CV si existen
    if "cv_results" in metrics:
        output["cv_results"] = {
            "mean_score": metrics["cv_results"].get("mean_score"),
            "std_score": metrics["cv_results"].get("std_score"),
            "scores": metrics["cv_results"].get("scores", []),
            "fold_metrics": metrics["cv_results"].get("fold_metrics", []),
        }

    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    logging.info(f"Métricas guardadas en {path}")


def load_metrics(path: Path | str) -> dict:
    """Carga métricas guardadas desde JSON.

    Parameters
    ----------
    path : Path | str
        Ruta al archivo JSON con métricas

    Returns
    -------
    dict
        Diccionario con métricas cargadas
    """
    with open(path, "r") as f:
        metrics = json.load(f)
    logging.info(f"Métricas cargadas desde {path}")
    return metrics


def save_model(model, path: Path | str) -> None:
    """Guarda un modelo entrenado.

    Parameters
    ----------
    model
        Modelo entrenado
    path : Path | str
        Ruta donde guardar el modelo
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logging.info(f"Modelo guardado en {path}")


def load_model(path: Path | str):
    """Carga un modelo guardado.

    Parameters
    ----------
    path : Path | str
        Ruta al archivo del modelo

    Returns
    -------
    Modelo cargado
    """
    with open(path, "rb") as f:
        model = pickle.load(f)  # nosec B301
    logging.info(f"Modelo cargado desde {path}")
    return model


def _is_first_run(output_dir: Path | str, model_type: str) -> bool:
    """Detecta si es la primera corrida (no existe modelo guardado).

    Parameters
    ----------
    output_dir : Path | str
        Directorio donde se guardaría el modelo
    model_type : str
        Tipo de modelo ('random_forest' o 'xgboost')

    Returns
    -------
    bool
        True si no existe modelo guardado (primera corrida), False si existe
    """
    output_dir = Path(output_dir)
    model_path = output_dir / f"{model_type}_model.pkl"
    return not model_path.exists()


def _has_explicit_params(model_kwargs: dict) -> bool:
    """Detecta si se pasaron parámetros explícitos del modelo.

    Parameters
    ----------
    model_kwargs : dict
        Diccionario con parámetros del modelo

    Returns
    -------
    bool
        True si se pasaron parámetros explícitos (n_estimators, max_depth, etc.)
    """
    # Parámetros que indican configuración explícita
    explicit_params = {
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "learning_rate",
        "class_weight",
        "random_state",
        "n_jobs",
    }
    # Verificar si alguno de estos parámetros está en model_kwargs
    # (excluyendo flags como cross_validate, cv_folds, save_metrics)
    return any(
        param in model_kwargs
        for param in explicit_params
        if param not in {"cross_validate", "cv_folds", "save_metrics", "optimize"}
    )


def optimize_hyperparameters(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    groups_train: pd.Series,
    groups_val: pd.Series,
    n_iter: int = 20,
    cv_folds: int = 3,
) -> dict:
    """Optimiza hiperparámetros usando RandomizedSearchCV con subject-level splitting.

    Parameters
    ----------
    model_type : str
        Tipo de modelo ('random_forest' o 'xgboost')
    X_train : pd.DataFrame
        Features de entrenamiento
    y_train : pd.Series
        Etiquetas de entrenamiento
    X_val : pd.DataFrame
        Features de validación (no se usa directamente, solo para referencia)
    y_val : pd.Series
        Etiquetas de validación (no se usa directamente, solo para referencia)
    groups_train : pd.Series
        Grupos (subject_core) para train
    groups_val : pd.Series
        Grupos (subject_core) para val
    n_iter : int
        Número de iteraciones para RandomizedSearchCV
    cv_folds : int
        Número de folds para cross-validation interna

    Returns
    -------
    dict
        Diccionario con mejores parámetros encontrados y resultados
    """
    logging.info("=" * 60)
    logging.info("ETAPA: OPTIMIZACIÓN DE HIPERPARÁMETROS")
    logging.info("=" * 60)
    logging.info(f"Buscando mejores hiperparámetros para {model_type}...")
    logging.info(f"Evaluando {n_iter} combinaciones de parámetros")

    # Definir espacios de búsqueda según el tipo de modelo
    if model_type == "random_forest":
        param_distributions = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [None, 10, 20, 30, 40],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "class_weight": [None, "balanced"],
        }
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    elif model_type == "xgboost":
        param_distributions = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [3, 4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
        }
        base_model = xgb.XGBClassifier(
            random_state=42, n_jobs=-1, eval_metric="mlogloss"
        )
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")

    # Crear cross-validator con subject-level splitting
    cv = SubjectTimeSeriesSplit(n_splits=cv_folds, test_size=0.2)

    # Combinar train y val para cross-validation interna
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)
    groups_combined = pd.concat([groups_train, groups_val], ignore_index=True)

    # Para XGBoost, necesitamos codificar las etiquetas antes de RandomizedSearchCV
    # Crear wrapper que maneje la codificación automáticamente
    if model_type == "xgboost":

        class XGBoostWrapper(xgb.XGBClassifier):
            """Wrapper para XGBoost que maneja codificación de etiquetas automáticamente."""

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.label_encoder_ = None

            def fit(self, X, y, **kwargs):
                # Codificar etiquetas si son strings
                if len(y) > 0 and isinstance(
                    y.iloc[0] if hasattr(y, "iloc") else y[0], str
                ):
                    if self.label_encoder_ is None:
                        self.label_encoder_ = LabelEncoder()
                        y_encoded = self.label_encoder_.fit_transform(y)
                    else:
                        y_encoded = self.label_encoder_.transform(y)
                    # Guardar clases originales para decodificación
                    self.classes_ = self.label_encoder_.classes_
                else:
                    y_encoded = y
                return super().fit(X, y_encoded, **kwargs)

            def predict(self, X):
                y_pred_encoded = super().predict(X)
                # Decodificar si tenemos label_encoder_
                if self.label_encoder_ is not None:
                    return self.label_encoder_.inverse_transform(y_pred_encoded)
                return y_pred_encoded

        base_model = XGBoostWrapper(random_state=42, n_jobs=-1, eval_metric="mlogloss")
    else:
        # Para Random Forest, usar el modelo base directamente
        pass

    # RandomizedSearchCV con subject-level splitting
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    logging.info("Ejecutando búsqueda de hiperparámetros...")
    random_search.fit(X_combined, y_combined, groups=groups_combined)

    logging.info("\n" + "=" * 60)
    logging.info("RESULTADOS DE OPTIMIZACIÓN")
    logging.info("=" * 60)
    logging.info(f"Mejor score (F1-macro): {random_search.best_score_:.4f}")
    logging.info("Mejores parámetros encontrados:")
    for param, value in random_search.best_params_.items():
        logging.info(f"  {param}: {value}")
    logging.info("=" * 60)

    return {
        "best_params": random_search.best_params_,
        "best_score": random_search.best_score_,
        "cv_results": random_search.cv_results_,
    }


def run_training_pipeline(
    manifest_path: Path | str,
    model_type: str = "random_forest",
    output_dir: Path | str = "models",
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    limit: Optional[int] = None,
    epoch_length: float = 30.0,
    sfreq: Optional[float] = None,
    features_file: Optional[Path | str] = None,
    **model_kwargs,
) -> dict:
    """Ejecuta pipeline completo de entrenamiento y evaluación.

    Parameters
    ----------
    manifest_path : Path | str
        Ruta al manifest CSV (solo necesario si features_file no se proporciona)
    model_type : str
        Tipo de modelo ('random_forest' o 'xgboost')
    output_dir : Path | str
        Directorio donde guardar el modelo
    test_size : float
        Proporción del test set
    limit : int, optional
        Limitar número de sesiones (solo si se extraen features)
    epoch_length : float
        Duración de cada epoch (solo si se extraen features)
    sfreq : float, optional
        Frecuencia de muestreo objetivo (solo si se extraen features)
    features_file : Path | str, optional
        Ruta a archivo con features pre-extraídas (Parquet o CSV).
        Si se proporciona, se omite la extracción de features.
    **model_kwargs
        Argumentos adicionales para el modelo

    Returns
    -------
    dict
        Diccionario con métricas de evaluación
    """
    logging.info("Iniciando pipeline de entrenamiento")

    # Detectar contexto: primera corrida vs corrida avanzada
    output_dir_path = Path(output_dir)
    is_first_run = _is_first_run(output_dir, model_type)
    has_explicit = _has_explicit_params(model_kwargs)
    force_optimize = model_kwargs.pop("optimize", False)

    # Decidir si usar validation y optimización
    if force_optimize:
        # Flag explícito: siempre optimizar
        use_validation = True
        should_optimize = True
        logging.info("\n" + "=" * 60)
        logging.info("MODO: OPTIMIZACIÓN DE HIPERPARÁMETROS (--optimize activado)")
        logging.info("=" * 60)
    elif is_first_run and not has_explicit:
        # Primera corrida sin parámetros explícitos: optimizar automáticamente
        use_validation = True
        should_optimize = True
        logging.info("\n" + "=" * 60)
        logging.info("MODO: PRIMERA CORRIDA - Optimización automática activada")
        logging.info("=" * 60)
        logging.info("No se encontró modelo previo y no se especificaron parámetros.")
        logging.info("Se activará validation set y optimización de hiperparámetros.")
    elif not is_first_run and not force_optimize:
        # Corrida avanzada: solo train/test (a menos que se especifique val_size)
        use_validation = val_size is not None
        should_optimize = False
        logging.info("\n" + "=" * 60)
        logging.info("MODO: CORRIDA AVANZADA")
        logging.info("=" * 60)
        logging.info(
            f"Modelo previo encontrado en {output_dir_path / f'{model_type}_model.pkl'}"
        )
        if use_validation:
            logging.info("Validation set activado (val_size especificado).")
        else:
            logging.info("Usando solo train/test. Usa --optimize para re-optimizar.")
    else:
        # Parámetros explícitos pero primera corrida: usar validation para comparar pero no optimizar
        use_validation = True  # Siempre usar validation cuando hay parámetros explícitos para comparar
        should_optimize = False
        logging.info("\n" + "=" * 60)
        logging.info("MODO: CORRIDA CON PARÁMETROS ESPECÍFICOS")
        logging.info("=" * 60)
        logging.info("Parámetros explícitos detectados. Usando esos parámetros.")
        logging.info("Validation set activado para comparar configuraciones.")

    # Ajustar val_size según la decisión
    if use_validation and val_size is None:
        val_size = 0.2
        logging.info(f"Validation size configurado automáticamente: {val_size}")

    # 1. Preparar dataset (cargar features pre-extraídas o extraerlas)
    logging.info("\n" + "=" * 60)
    logging.info("ETAPA 1: PREPARACIÓN DE DATOS")
    logging.info("=" * 60)
    if features_file:
        logging.info(f"Cargando features desde {features_file}...")
        features_path = Path(features_file)
        if features_path.suffix == ".parquet":
            features_df = pd.read_parquet(features_path, engine="pyarrow")
        elif features_path.suffix == ".csv":
            features_df = pd.read_csv(features_path)
        else:
            raise ValueError(
                f"Formato de archivo no soportado: {features_path.suffix}. "
                "Use .parquet o .csv"
            )
        logging.info(f"✓ Features cargadas: {len(features_df)} epochs")
    else:
        logging.info("Extrayendo features desde archivos raw...")
        features_df = prepare_features_dataset(
            manifest_path, limit=limit, epoch_length=epoch_length, sfreq=sfreq
        )
        logging.info("✓ Extracción de features completada")

    # 2. Separar features y etiquetas
    logging.info("\n" + "=" * 60)
    logging.info("ETAPA 2: PREPARACIÓN DE FEATURES Y ETIQUETAS")
    logging.info("=" * 60)
    feature_cols = [
        col
        for col in features_df.columns
        if col
        not in [
            "stage",
            "subject_id",
            "subject_core",
            "session_idx",
            "epoch_time_start",
            "epoch_index",
        ]
    ]
    X = features_df[feature_cols]
    y = features_df["stage"]

    # Filtrar estadios válidos
    logging.info("Filtrando estadios válidos...")
    valid_stages = set(STAGE_ORDER)
    mask = y.isin(valid_stages)
    X = X[mask]
    y = y[mask]
    features_df = features_df[mask].reset_index(drop=True)

    logging.info(f"✓ Features seleccionadas: {len(feature_cols)}")
    logging.info(f"✓ Epochs totales después de filtrar: {len(X)}")
    logging.info(f"✓ Distribución de estadios:\n{y.value_counts().sort_index()}")

    # 3. Preparar datos para división (preservar metadata de sujetos)
    logging.info("\n" + "=" * 60)
    logging.info("ETAPA 3: DIVISIÓN DE DATOS")
    logging.info("=" * 60)
    logging.info("Preparando metadata de sujetos...")
    combined_df = pd.concat(
        [X.reset_index(drop=True), y.reset_index(drop=True)], axis=1
    )
    combined_df["subject_id"] = features_df["subject_id"].values
    combined_df["subject_core"] = features_df["subject_core"].values
    # Nota: Las columnas temporales (epoch_time_start, epoch_index) no son necesarias
    # para SubjectTimeSeriesSplit, pero las mantenemos por si se necesita GroupTimeSeriesSplit
    if "epoch_time_start" in features_df.columns:
        combined_df["epoch_time_start"] = features_df["epoch_time_start"].values
    if "epoch_index" in features_df.columns:
        combined_df["epoch_index"] = features_df["epoch_index"].values

    # 4. Cross-validation o train/test simple
    use_cv = model_kwargs.pop("cross_validate", False)
    cv_folds = model_kwargs.pop("cv_folds", 5)

    if use_cv:
        # Cross-validation respetando grupos y tiempo
        logging.info(f"Modo: Cross-validation ({cv_folds} folds)")
        logging.info(f"Test size: {test_size}")
        cv = SubjectTimeSeriesSplit(n_splits=cv_folds, test_size=test_size)
        groups = combined_df["subject_core"]

        # Crear modelo base
        if model_type == "random_forest":
            base_model = RandomForestClassifier(**model_kwargs)
        elif model_type == "xgboost":
            base_model = xgb.XGBClassifier(**model_kwargs)
        else:
            raise ValueError(f"Tipo de modelo desconocido: {model_type}")

        # Cross-validation (SubjectTimeSeriesSplit no necesita columnas temporales)
        cv_results = cross_validate_model(
            base_model,
            X,  # Features sin metadata
            y,
            groups,
            cv,
            scoring="f1_macro",
            X_full=None,  # No necesario para SubjectTimeSeriesSplit
        )

        # Entrenar modelo final con todos los datos de train
        logging.info("\n" + "=" * 60)
        logging.info("ETAPA 4: ENTRENAMIENTO DEL MODELO FINAL")
        logging.info("=" * 60)
        logging.info("Dividiendo datos para entrenamiento final...")
        train_df, test_df, val_df = prepare_train_test_split(
            combined_df, test_size=test_size, val_size=val_size
        )
        X_train = train_df[feature_cols]
        y_train = train_df["stage"]

        if model_type == "random_forest":
            model = train_random_forest(X_train, y_train, **model_kwargs)
        elif model_type == "xgboost":
            model = train_xgboost(X_train, y_train, **model_kwargs)

        # Evaluar en test
        logging.info("\n" + "=" * 60)
        logging.info("ETAPA 5: EVALUACIÓN EN TEST")
        logging.info("=" * 60)
        X_test = test_df[feature_cols]
        y_test = test_df["stage"]
        metrics = evaluate_model(
            model, X_test, y_test, stage_order=STAGE_ORDER, dataset_name="TEST"
        )
        metrics["cv_results"] = cv_results
        print_evaluation_report(metrics, STAGE_ORDER)
    else:
        # Train/test simple (ya respeta grupos y tiempo en prepare_train_test_split)
        logging.info("Modo: Train/Test split simple")
        logging.info(f"Test size: {test_size}")
        if val_size:
            logging.info(f"Validation size: {val_size}")
        train_df, test_df, val_df = prepare_train_test_split(
            combined_df, test_size=test_size, val_size=val_size
        )
        X_train = train_df[feature_cols]
        y_train = train_df["stage"]
        X_test = test_df[feature_cols]
        y_test = test_df["stage"]

        if val_df is not None:
            X_val = val_df[feature_cols]
            y_val = val_df["stage"]
        elif should_optimize:
            # Si necesitamos optimizar pero no hay val_df, error
            raise ValueError(
                "Se requiere validation set para optimización de hiperparámetros. "
                "Asegúrate de que val_size esté configurado o que use_validation sea True."
            )

        # 4. Optimizar hiperparámetros si es necesario
        if should_optimize:
            opt_results = optimize_hyperparameters(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                groups_train=train_df["subject_core"],
                groups_val=val_df["subject_core"],
                n_iter=model_kwargs.pop("n_iter_optimize", 20),
                cv_folds=model_kwargs.pop("cv_folds_optimize", 3),
            )
            # Actualizar model_kwargs con los mejores parámetros encontrados
            model_kwargs.update(opt_results["best_params"])
            logging.info("\n" + "=" * 60)
            logging.info(
                "Usando mejores parámetros encontrados para entrenamiento final"
            )
            logging.info("=" * 60)

        # 5. Entrenar modelo
        logging.info("\n" + "=" * 60)
        logging.info("ETAPA 4: ENTRENAMIENTO DEL MODELO")
        logging.info("=" * 60)
        if model_type == "random_forest":
            model = train_random_forest(X_train, y_train, **model_kwargs)
        elif model_type == "xgboost":
            model = train_xgboost(X_train, y_train, **model_kwargs)
        else:
            raise ValueError(f"Tipo de modelo desconocido: {model_type}")

        # 6. Evaluar en validación si existe
        if val_df is not None:
            logging.info("\n" + "=" * 60)
            logging.info("ETAPA 5: EVALUACIÓN EN VALIDACIÓN")
            logging.info("=" * 60)
            val_metrics = evaluate_model(
                model, X_val, y_val, stage_order=STAGE_ORDER, dataset_name="VALIDATION"
            )
            logging.info("Métricas de validación:")
            logging.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            logging.info(f"  Cohen's Kappa: {val_metrics['kappa']:.4f}")
            logging.info(f"  F1-score (macro): {val_metrics['f1_macro']:.4f}")
            logging.info(f"  F1-score (weighted): {val_metrics['f1_weighted']:.4f}")

        # 7. Evaluar en test
        logging.info("\n" + "=" * 60)
        logging.info("ETAPA 6: EVALUACIÓN EN TEST")
        logging.info("=" * 60)
        metrics = evaluate_model(
            model, X_test, y_test, stage_order=STAGE_ORDER, dataset_name="TEST"
        )
        print_evaluation_report(metrics, STAGE_ORDER)

    # 8. Guardar modelo
    logging.info("\n" + "=" * 60)
    logging.info("ETAPA FINAL: GUARDADO DE RESULTADOS")
    logging.info("=" * 60)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_type}_model.pkl"
    logging.info(f"Guardando modelo en {model_path}...")
    save_model(model, model_path)

    # Guardar feature names también
    feature_names_path = output_dir / f"{model_type}_feature_names.pkl"
    logging.info(f"Guardando nombres de features en {feature_names_path}...")
    with open(feature_names_path, "wb") as f:
        pickle.dump(feature_cols, f)

    # Guardar métricas también (si no se desactivó explícitamente)
    save_metrics_flag = model_kwargs.pop("save_metrics", True)
    if save_metrics_flag:
        metrics_path = output_dir / f"{model_type}_metrics.json"
        logging.info(f"Guardando métricas en {metrics_path}...")
        save_metrics(metrics, metrics_path, model_type=model_type, **model_kwargs)

    logging.info("\n" + "=" * 60)
    logging.info("✓ PIPELINE COMPLETADO EXITOSAMENTE")
    logging.info("=" * 60)
    logging.info(f"Modelo guardado en: {model_path}")
    if save_metrics_flag:
        logging.info(f"Métricas guardadas en: {metrics_path}")
    logging.info("=" * 60 + "\n")

    return metrics


def build_parser() -> argparse.ArgumentParser:
    """Construye parser de argumentos para CLI."""
    parser = argparse.ArgumentParser(
        description="Entrenar modelo para clasificación de estadios de sueño"
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/manifest_trimmed.csv",
        help="Ruta al manifest CSV con sesiones procesadas",
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "xgboost"],
        default="random_forest",
        help="Tipo de modelo a entrenar",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directorio donde guardar el modelo",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción del test set (sobre subject_cores)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=None,
        help="Proporción del validation set (sobre subject_cores). Si no se especifica, no se crea val.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limitar número de sesiones a procesar (para pruebas)",
    )
    parser.add_argument(
        "--epoch-length",
        type=float,
        default=30.0,
        help="Duración de cada epoch en segundos",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=None,
        help="Frecuencia de muestreo objetivo (None = mantener original)",
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default=None,
        help="Ruta a archivo con features pre-extraídas (Parquet o CSV). "
        "Si se proporciona, se omite la extracción de features.",
    )
    # Parámetros específicos de Random Forest
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Número de árboles (Random Forest)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Profundidad máxima de los árboles",
    )
    parser.add_argument(
        "--cross-validate",
        action="store_true",
        help="Realizar cross-validation en lugar de train/test simple",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Número de folds para cross-validation",
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        default=True,
        help="Guardar métricas en JSON (default: True)",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        default=False,
        help="Forzar optimización de hiperparámetros (usa RandomizedSearchCV). "
        "Si no se especifica, se detecta automáticamente: primera corrida sin parámetros → optimiza automáticamente.",
    )
    parser.add_argument(
        "--n-iter-optimize",
        type=int,
        default=20,
        help="Número de iteraciones para optimización de hiperparámetros (default: 20)",
    )
    parser.add_argument(
        "--cv-folds-optimize",
        type=int,
        default=3,
        help="Número de folds para cross-validation en optimización (default: 3)",
    )
    return parser


def main() -> int:
    """Función principal para ejecutar desde CLI."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = build_parser()
    args = parser.parse_args()

    model_kwargs = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "cross_validate": args.cross_validate,
        "cv_folds": args.cv_folds,
        "save_metrics": args.save_metrics,
        "optimize": args.optimize,
        "n_iter_optimize": args.n_iter_optimize,
        "cv_folds_optimize": args.cv_folds_optimize,
    }

    try:
        run_training_pipeline(
            manifest_path=args.manifest,
            model_type=args.model_type,
            output_dir=args.output_dir,
            test_size=args.test_size,
            val_size=args.val_size,
            limit=args.limit,
            epoch_length=args.epoch_length,
            sfreq=args.sfreq,
            features_file=args.features_file,
            **model_kwargs,
        )
        return 0
    except Exception as e:
        logging.exception(f"Error en pipeline: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
