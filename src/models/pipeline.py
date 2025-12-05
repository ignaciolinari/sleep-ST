"""Pipeline de entrenamiento y funciones auxiliares.

Este módulo contiene el pipeline principal de entrenamiento, cross-validation,
funciones de evaluación, y utilidades para detección de primera corrida.
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from ..crossval import SubjectTimeSeriesSplit, GroupTimeSeriesSplit

from .base import (
    STAGE_ORDER,
    TF_AVAILABLE,
    evaluate_model,
    print_evaluation_report,
    save_metrics,
    save_model,
)
from .data_preparation import (
    prepare_features_dataset,
    prepare_raw_epochs_dataset,
    prepare_sequence_dataset,
    prepare_train_test_split,
)
from ..features import DEFAULT_FILTER_BAND, DEFAULT_NOTCH_FREQS
from .random_forest import train_random_forest
from .xgboost_model import train_xgboost
from .optimization import optimize_hyperparameters

if TF_AVAILABLE:
    from .cnn1d import train_cnn1d
    from .lstm import train_lstm


def cross_validate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv: "SubjectTimeSeriesSplit | GroupTimeSeriesSplit | LeaveOneGroupOut",
    scoring: str = "accuracy",
    X_full: Optional[pd.DataFrame] = None,
    required_classes: Optional[Sequence[str]] = None,
    strict_class_coverage: bool = False,
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
    required_classes : Sequence[str], optional
        Clases que se espera cubrir en cada split. Si se proporciona, se revisa
        cobertura en train y test por fold.
    strict_class_coverage : bool
        Si True, lanza ValueError cuando falten clases requeridas en algún split.
        Si False (default), solo emite warning.

    Returns
    -------
    dict
        Diccionario con métricas de CV
    """
    cv_scores = []
    fold_metrics = []

    required_classes = list(required_classes) if required_classes else []

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

        def _check_missing_classes(y_split: pd.Series, split_label: str) -> list[str]:
            if not required_classes:
                return []
            present = set(pd.Series(y_split).dropna().unique())
            missing = [cls for cls in required_classes if cls not in present]
            if missing:
                msg = (
                    f"Fold {fold_idx + 1}: {split_label} sin clases {missing}. "
                    "Resultados pueden ser inestables."
                )
                if strict_class_coverage:
                    raise ValueError(
                        msg + " Habilita más sujetos o ajusta cv_strategy/test_size."
                    )
                logging.warning(msg)
            return missing

        missing_train = _check_missing_classes(y_train_fold, "train")
        missing_test = _check_missing_classes(y_test_fold, "test")

        if missing_train or missing_test:
            logging.info(
                f"  Cobertura de clases - train faltan: {missing_train or 'ninguna'}; "
                f"test faltan: {missing_test or 'ninguna'}"
            )

        logging.info(f"  Entrenando modelo para fold {fold_idx + 1}...")
        # Entrenar modelo
        model_fold = type(model)(**model.get_params())

        # Para XGBoost, necesitamos codificar las etiquetas
        is_xgboost = XGB_AVAILABLE and isinstance(model_fold, xgb.XGBClassifier)
        if is_xgboost:
            le_fold = LabelEncoder()
            y_train_fold_encoded = le_fold.fit_transform(y_train_fold)

            # Calcular sample_weight para balancear clases
            class_counts_fold = y_train_fold.value_counts()
            total_samples_fold = len(y_train_fold_encoded)
            n_classes_fold = len(le_fold.classes_)
            class_weights_fold = {}
            for class_idx, class_name in enumerate(le_fold.classes_):
                class_count = class_counts_fold.get(class_name, 0)
                if class_count > 0:
                    class_weights_fold[class_idx] = total_samples_fold / (
                        n_classes_fold * class_count
                    )
                else:
                    class_weights_fold[class_idx] = 1.0
            sample_weight_fold = np.array(
                [class_weights_fold[y] for y in y_train_fold_encoded]
            )

            model_fold.fit(
                X_train_fold, y_train_fold_encoded, sample_weight=sample_weight_fold
            )
            model_fold.label_encoder_ = le_fold
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
            # Usar label_encoder_ si está disponible, sino usar classes_
            if hasattr(model_fold, "label_encoder_"):
                y_pred_fold = model_fold.label_encoder_.inverse_transform(y_pred_fold)
            else:
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
            score = f1_score(y_test_fold, y_pred_fold, average="macro", zero_division=0)
        elif scoring == "f1_weighted":
            score = f1_score(
                y_test_fold, y_pred_fold, average="weighted", zero_division=0
            )
        elif scoring == "kappa":
            score = cohen_kappa_score(y_test_fold, y_pred_fold)
        else:
            raise ValueError(f"Métrica desconocida: {scoring}")

        logging.info(f"  ✓ Fold {fold_idx + 1} completado - {scoring}: {score:.4f}")
        cv_scores.append(score)
        fold_metric = {
            "fold": fold_idx,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "score": score,
        }
        if missing_train or missing_test:
            fold_metric["missing_classes"] = {
                "train": missing_train,
                "test": missing_test,
            }
        fold_metrics.append(fold_metric)

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
    sequence_length: int = 5,  # Para LSTM
    temporal_split: bool = False,
    cv_strategy: str = "subject-kfold",
    strict_class_coverage: bool = False,
    overlap: float = 0.0,
    prefilter: bool = True,
    bandpass_low: Optional[float] = None,
    bandpass_high: Optional[float] = None,
    notch_freqs: Optional[Sequence[float]] = None,
    psd_method: str = "welch",
    stage_stratify: bool = True,
    skip_cross_if_single_eeg: bool = True,
    **model_kwargs,
) -> dict:
    """Ejecuta pipeline completo de entrenamiento y evaluación.

    Parameters
    ----------
    manifest_path : Path | str
        Ruta al manifest CSV (solo necesario si features_file no se proporciona)
    model_type : str
        Tipo de modelo ('random_forest', 'xgboost', 'cnn1d', o 'lstm')
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
        Requerido para LSTM, opcional para otros modelos.
    sequence_length : int
        Longitud de secuencias para LSTM (default: 5 epochs)
    temporal_split : bool
        Si True, usa splits temporales por sesión dentro de cada sujeto para
        train/val/test y CV (holdout de sesiones recientes).
    cv_strategy : str
        Estrategia de CV cuando cross_validate está activado.
        - "subject-kfold": k-fold por sujeto (default)
        - "loso": leave-one-subject-out (un sujeto en test por fold)
        - "group-temporal": CV temporal por grupo/sujeto
    strict_class_coverage : bool
        Si True, falla cuando algún split de CV carece de clases requeridas.
        En train/test simple ya se valida cobertura vía prepare_train_test_split.
    **model_kwargs
        Argumentos adicionales para el modelo

    Returns
    -------
    dict
        Diccionario con métricas de evaluación
    """
    logging.info("Iniciando pipeline de entrenamiento")

    # Validar que TensorFlow esté disponible para modelos de deep learning
    if model_type in ["cnn1d", "lstm"] and not TF_AVAILABLE:
        raise ImportError(
            f"TensorFlow no está disponible. Instala TensorFlow para usar el modelo {model_type}."
        )

    # Permitir que temporal_split llegue vía model_kwargs pero no se propague a los entrenadores
    temporal_split = bool(temporal_split or model_kwargs.pop("temporal_split", False))

    if epoch_length <= 0:
        raise ValueError("epoch_length debe ser > 0")
    if overlap < 0:
        raise ValueError("overlap debe ser >= 0")
    if overlap >= epoch_length:
        raise ValueError("overlap debe ser menor que epoch_length")

    # Band-pass / notch para features
    bandpass_low = (
        DEFAULT_FILTER_BAND[0] if bandpass_low is None else float(bandpass_low)
    )
    bandpass_high = (
        DEFAULT_FILTER_BAND[1] if bandpass_high is None else float(bandpass_high)
    )
    if bandpass_low >= bandpass_high:
        raise ValueError("bandpass_low debe ser menor que bandpass_high")
    bandpass = (bandpass_low, bandpass_high)
    notch = tuple(notch_freqs) if notch_freqs else DEFAULT_NOTCH_FREQS

    psd_method = (psd_method or "welch").lower()
    if psd_method not in {"welch", "multitaper"}:
        psd_method = "welch"

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

    # 1. Preparar dataset según el tipo de modelo
    logging.info("\n" + "=" * 60)
    logging.info("ETAPA 1: PREPARACIÓN DE DATOS")
    logging.info("=" * 60)

    # Manejo especial para modelos de deep learning
    if model_type == "cnn1d":
        # CNN1D necesita señales raw
        logging.info("Preparando datos raw para CNN1D...")
        X_raw, y_raw, metadata_df = prepare_raw_epochs_dataset(
            manifest_path,
            limit=limit,
            epoch_length=epoch_length,
            sfreq=sfreq,
            overlap=overlap,
        )
        logging.info("✓ Datos raw preparados para CNN1D")
        # Saltar al entrenamiento directo (manejo especial más abajo)
        use_dl_pipeline = True
    elif model_type == "lstm":
        # LSTM necesita features en secuencias
        if not features_file:
            raise ValueError(
                "LSTM requiere features pre-extraídas. "
                "Proporciona --features-file con un archivo de features."
            )
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
        logging.info("Creando secuencias para LSTM...")
        X_seq, y_seq, metadata_df = prepare_sequence_dataset(
            features_df, sequence_length=sequence_length
        )
        logging.info("✓ Secuencias preparadas para LSTM")
        use_dl_pipeline = True
    else:
        # Modelos tradicionales (random_forest, xgboost)
        use_dl_pipeline = False
        # Cargar features para modelos tradicionales
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
            # Extraer features desde manifest
            features_df = prepare_features_dataset(
                manifest_path,
                limit=limit,
                epoch_length=epoch_length,
                sfreq=sfreq,
                overlap=overlap,
                apply_prefilter=prefilter,
                bandpass=bandpass,
                notch_freqs=notch,
                psd_method=psd_method,
                skip_cross_if_single_eeg=skip_cross_if_single_eeg,
            )

    # Pipeline especial para modelos de deep learning
    if use_dl_pipeline:
        # Dividir datos respetando subject_core
        logging.info("\n" + "=" * 60)
        logging.info("ETAPA 2: DIVISIÓN DE DATOS (Deep Learning)")
        logging.info("=" * 60)

        if model_type == "cnn1d":
            X_data, y_data = X_raw, y_raw
        else:  # lstm
            X_data, y_data = X_seq, y_seq

        # Usar lógica común de splitting con verificación de cobertura de clases
        metadata_with_labels = metadata_df.copy()
        metadata_with_labels["stage"] = y_data

        train_meta, test_meta, val_meta = prepare_train_test_split(
            metadata_with_labels,
            test_size=test_size,
            val_size=val_size,
            random_state=42,
            stratify_by="subject_core",
            ensure_class_coverage=True,
            temporal_split=temporal_split,
            session_col="session_idx",
            stage_stratify=stage_stratify,
        )

        train_idx = train_meta.index.to_numpy()
        test_idx = test_meta.index.to_numpy()
        val_idx = val_meta.index.to_numpy() if val_meta is not None else None

        X_train = X_data[train_idx]
        y_train = y_data[train_idx]
        X_test = X_data[test_idx]
        y_test = y_data[test_idx]
        X_val = X_data[val_idx] if val_meta is not None else None
        y_val = y_data[val_idx] if val_meta is not None else None

        logging.info(f"Train: {len(X_train)} muestras")
        logging.info(f"Test: {len(X_test)} muestras")
        if val_size:
            logging.info(f"Val: {len(X_val)} muestras")

        # Entrenar modelo
        logging.info("\n" + "=" * 60)
        logging.info("ETAPA 3: ENTRENAMIENTO DEL MODELO")
        logging.info("=" * 60)

        if model_type == "cnn1d":
            model = train_cnn1d(
                X_train,
                y_train,
                X_val,
                y_val,
                **{
                    k: v
                    for k, v in model_kwargs.items()
                    if k
                    not in [
                        "cross_validate",
                        "cv_folds",
                        "save_metrics",
                        "optimize",
                        "n_iter_optimize",
                        "cv_folds_optimize",
                    ]
                },
            )
        else:  # lstm
            model = train_lstm(
                X_train,
                y_train,
                X_val,
                y_val,
                **{
                    k: v
                    for k, v in model_kwargs.items()
                    if k
                    not in [
                        "cross_validate",
                        "cv_folds",
                        "save_metrics",
                        "optimize",
                        "n_iter_optimize",
                        "cv_folds_optimize",
                    ]
                },
            )

        # Evaluar en validación si existe
        if val_size and X_val is not None:
            logging.info("\n" + "=" * 60)
            logging.info("ETAPA 4: EVALUACIÓN EN VALIDACIÓN")
            logging.info("=" * 60)
            val_metrics = evaluate_model(
                model, X_val, y_val, stage_order=STAGE_ORDER, dataset_name="VALIDATION"
            )
            logging.info("Métricas de validación:")
            logging.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
            logging.info(f"  Cohen's Kappa: {val_metrics['kappa']:.4f}")
            logging.info(f"  F1-score (macro): {val_metrics['f1_macro']:.4f}")

        # Evaluar en test
        logging.info("\n" + "=" * 60)
        logging.info("ETAPA 5: EVALUACIÓN EN TEST")
        logging.info("=" * 60)
        metrics = evaluate_model(
            model, X_test, y_test, stage_order=STAGE_ORDER, dataset_name="TEST"
        )
        print_evaluation_report(metrics, STAGE_ORDER)

        # Guardar modelo
        logging.info("\n" + "=" * 60)
        logging.info("ETAPA FINAL: GUARDADO DE RESULTADOS")
        logging.info("=" * 60)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Para modelos de Keras, guardar como directorio
        model_path = output_dir / f"{model_type}_model"
        save_model(model, model_path)

        # Guardar métricas
        save_metrics_flag = model_kwargs.pop("save_metrics", True)
        if save_metrics_flag:
            metrics_path = output_dir / f"{model_type}_metrics.json"
            save_metrics(metrics, metrics_path, model_type=model_type, **model_kwargs)

        logging.info("\n" + "=" * 60)
        logging.info("✓ PIPELINE COMPLETADO EXITOSAMENTE")
        logging.info("=" * 60)
        logging.info(f"Modelo guardado en: {model_path}")
        if save_metrics_flag:
            logging.info(f"Métricas guardadas en: {metrics_path}")
        logging.info("=" * 60 + "\n")

        return metrics

    # Continuar con pipeline tradicional para modelos ML clásicos
    # (el código de preparación de datos ya se ejecutó arriba para estos modelos)

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
    cv_strategy_normalized = (cv_strategy or "subject-kfold").lower()

    if use_cv:
        groups = combined_df["subject_core"]
        if cv_strategy_normalized not in {
            "subject-kfold",
            "loso",
            "group-temporal",
        }:
            raise ValueError(
                "cv_strategy debe ser 'subject-kfold', 'loso' o 'group-temporal'"
            )

        if temporal_split and cv_strategy_normalized == "loso":
            raise ValueError(
                "cv_strategy='loso' no es compatible con temporal_split=True. "
                "Usa 'group-temporal' o desactiva temporal_split."
            )

        # Si se solicita temporal_split pero cv_strategy no es explícitamente temporal,
        # priorizar la división temporal para consistencia
        cv_strategy_effective = (
            "group-temporal" if temporal_split else cv_strategy_normalized
        )

        if cv_strategy_effective == "loso":
            cv = LeaveOneGroupOut()
            X_full_for_split = None
            effective_folds = groups.nunique()
            logging.info("Modo: Cross-validation [loso] (1 sujeto en test por fold)")
            logging.info("Test size se ignora en LOSO.")
        elif cv_strategy_effective == "group-temporal":
            if not {"epoch_index", "epoch_time_start"} & set(combined_df.columns):
                raise ValueError(
                    "Se solicitó temporal_split pero no hay columnas temporales "
                    "('epoch_index' o 'epoch_time_start') en los datos."
                )
            cv = GroupTimeSeriesSplit(n_splits=cv_folds, test_size=test_size)
            X_full_for_split = combined_df
            effective_folds = cv.get_n_splits(combined_df, y, groups)
            logging.info(
                f"Modo: Cross-validation [group-temporal] ({effective_folds} folds)"
            )
            logging.info(f"Test size (por grupo): {test_size}")
        else:
            cv = SubjectTimeSeriesSplit(n_splits=cv_folds, test_size=test_size)
            X_full_for_split = None
            effective_folds = cv.get_n_splits(X, y, groups)
            logging.info(
                f"Modo: Cross-validation [subject-kfold] ({effective_folds} folds)"
            )
            logging.info(f"Test size (por sujeto): {test_size}")

        # Crear modelo base
        if model_type == "random_forest":
            base_model = RandomForestClassifier(**model_kwargs)
        elif model_type == "xgboost":
            if not XGB_AVAILABLE:
                raise ImportError("XGBoost no está disponible.")
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
            X_full=X_full_for_split,
            required_classes=STAGE_ORDER,
            strict_class_coverage=strict_class_coverage,
        )

        # Entrenar modelo final con todos los datos de train
        logging.info("\n" + "=" * 60)
        logging.info("ETAPA 4: ENTRENAMIENTO DEL MODELO FINAL")
        logging.info("=" * 60)
        logging.info("Dividiendo datos para entrenamiento final...")
        train_df, test_df, val_df = prepare_train_test_split(
            combined_df,
            test_size=test_size,
            val_size=val_size,
            ensure_class_coverage=True,
            temporal_split=temporal_split,
            stage_stratify=stage_stratify,
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
            combined_df,
            test_size=test_size,
            val_size=val_size,
            ensure_class_coverage=True,
            temporal_split=temporal_split,
            stage_stratify=stage_stratify,
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

    # Guardar feature names también (solo para modelos tradicionales)
    if model_type not in ["cnn1d", "lstm"]:
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
        choices=["random_forest", "xgboost", "cnn1d", "lstm"],
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
        "--overlap",
        type=float,
        default=0.0,
        help="Solapamiento entre epochs en segundos (debe ser menor que epoch-length).",
    )
    parser.add_argument(
        "--prefilter",
        dest="prefilter",
        action="store_true",
        help="Aplicar detrend + band-pass + notch antes de extraer features (default: on).",
    )
    parser.add_argument(
        "--no-prefilter",
        dest="prefilter",
        action="store_false",
        help="Desactivar filtrado/notch previo a la extracción de features.",
    )
    parser.set_defaults(prefilter=True)
    parser.add_argument(
        "--bandpass-low",
        type=float,
        default=None,
        help="Corte inferior del band-pass para features (Hz). Default: 0.3 Hz.",
    )
    parser.add_argument(
        "--bandpass-high",
        type=float,
        default=None,
        help="Corte superior del band-pass para features (Hz). Default: 45 Hz.",
    )
    parser.add_argument(
        "--notch-freqs",
        type=float,
        nargs="+",
        default=None,
        help="Frecuencias de notch (Hz), ej. 50 60. Default: 50 y 60 Hz.",
    )
    parser.add_argument(
        "--psd-method",
        choices=["welch", "multitaper"],
        default="welch",
        help="Método para PSD en features espectrales (welch o multitaper).",
    )
    parser.add_argument(
        "--no-stage-stratify",
        dest="stage_stratify",
        action="store_false",
        help="Desactivar estratificación por estadio a nivel sujeto_core en los splits.",
    )
    parser.add_argument(
        "--keep-cross-single-eeg",
        dest="skip_cross_if_single_eeg",
        action="store_false",
        help="Mantener features cross-channel aunque solo haya un EEG (por defecto se omiten).",
    )
    parser.set_defaults(stage_stratify=True, skip_cross_if_single_eeg=True)
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
        "Si se proporciona, se omite la extracción de features. Requerido para LSTM.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Longitud de secuencias para LSTM (número de epochs consecutivos, default: 5)",
    )
    parser.add_argument(
        "--temporal-split",
        action="store_true",
        help="Usar split temporal por sesión dentro de cada sujeto (holdout de sesiones recientes) "
        "para train/val/test y cross-validation.",
    )
    # Parámetros específicos de Deep Learning
    parser.add_argument(
        "--n-filters",
        type=int,
        default=64,
        help="Número de filtros para CNN1D (default: 64)",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Tamaño del kernel convolucional para CNN1D (default: 3)",
    )
    parser.add_argument(
        "--lstm-units",
        type=int,
        default=128,
        help="Número de unidades LSTM (default: 128)",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.5,
        help="Tasa de dropout para modelos de deep learning (default: 0.5)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Tasa de aprendizaje para modelos de deep learning (default: 0.001)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamaño del batch para modelos de deep learning (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número de épocas para modelos de deep learning (default: 50)",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosidad para modelos de deep learning: 0=silencioso, 1=barra de progreso, 2=una línea por época (default: 1)",
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
        "--cv-strategy",
        choices=["subject-kfold", "loso", "group-temporal"],
        default="subject-kfold",
        help="Estrategia de cross-validation: subject-kfold (default), "
        "loso (leave-one-subject-out), group-temporal (respeta orden temporal por sujeto).",
    )
    parser.add_argument(
        "--strict-class-coverage",
        action="store_true",
        help="Falla si algún split de CV carece de clases requeridas.",
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
        help="Forzar optimización de hiperparámetros usando Optuna (optimización bayesiana). "
        "Si no se especifica, se detecta automáticamente: primera corrida sin parámetros → optimiza automáticamente.",
    )
    parser.add_argument(
        "--n-iter-optimize",
        type=int,
        default=50,
        help="Número de trials para optimización bayesiana de hiperparámetros (default: 50)",
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

    # Agregar parámetros específicos de deep learning
    if args.model_type in ["cnn1d", "lstm"]:
        model_kwargs.update(
            {
                "n_filters": args.n_filters,
                "kernel_size": args.kernel_size,
                "lstm_units": args.lstm_units,
                "dropout_rate": args.dropout_rate,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "verbose": args.verbose,
            }
        )

    try:
        run_training_pipeline(
            manifest_path=args.manifest,
            model_type=args.model_type,
            output_dir=args.output_dir,
            test_size=args.test_size,
            val_size=args.val_size,
            limit=args.limit,
            epoch_length=args.epoch_length,
            overlap=args.overlap,
            sfreq=args.sfreq,
            prefilter=args.prefilter,
            bandpass_low=args.bandpass_low,
            bandpass_high=args.bandpass_high,
            notch_freqs=args.notch_freqs,
            psd_method=args.psd_method,
            stage_stratify=args.stage_stratify,
            skip_cross_if_single_eeg=args.skip_cross_if_single_eeg,
            features_file=args.features_file,
            sequence_length=args.sequence_length,
            temporal_split=args.temporal_split,
            cv_strategy=args.cv_strategy,
            strict_class_coverage=args.strict_class_coverage,
            **model_kwargs,
        )
        return 0
    except Exception as e:
        logging.exception(f"Error en pipeline: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
