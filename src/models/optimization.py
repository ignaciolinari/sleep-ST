"""Funciones de optimización de hiperparámetros usando Optuna.

Este módulo contiene funciones para optimización bayesiana de hiperparámetros
tanto para modelos de Machine Learning como Deep Learning.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import xgboost as xgb

    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from ..crossval import SubjectTimeSeriesSplit
from .base import TF_AVAILABLE, OPTUNA_AVAILABLE, _configure_tensorflow_cpu_only

if OPTUNA_AVAILABLE:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner

if TF_AVAILABLE:
    from tensorflow import keras
    from .cnn1d import build_cnn1d_model
    from .lstm import build_lstm_model


def optimize_hyperparameters_bayesian(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    groups_train: pd.Series,
    groups_val: pd.Series,
    n_trials: int = 50,
    cv_folds: int = 3,
    timeout: Optional[int] = None,
    show_progress_bar: bool = True,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
) -> dict:
    """Optimiza hiperparámetros usando Optuna (optimización bayesiana).

    Esta función implementa optimización bayesiana con TPE (Tree-structured Parzen Estimator)
    que es más eficiente que RandomizedSearchCV, especialmente para espacios de búsqueda
    grandes y parámetros continuos.

    Parameters
    ----------
    model_type : str
        Tipo de modelo ('random_forest' o 'xgboost')
    X_train : pd.DataFrame
        Features de entrenamiento
    y_train : pd.Series
        Etiquetas de entrenamiento
    X_val : pd.DataFrame
        Features de validación
    y_val : pd.Series
        Etiquetas de validación
    groups_train : pd.Series
        Grupos (subject_core) para train
    groups_val : pd.Series
        Grupos (subject_core) para val
    n_trials : int
        Número de trials para optimización bayesiana (default: 50)
    cv_folds : int
        Número de folds para cross-validation interna
    timeout : int, optional
        Tiempo máximo en segundos para la optimización
    show_progress_bar : bool
        Mostrar barra de progreso

    Returns
    -------
    dict
        Diccionario con mejores parámetros encontrados y resultados
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna no está disponible. Instala con: pip install optuna")

    logging.info("=" * 60)
    logging.info("ETAPA: OPTIMIZACIÓN BAYESIANA DE HIPERPARÁMETROS (Optuna)")
    logging.info("=" * 60)
    logging.info(f"Buscando mejores hiperparámetros para {model_type}...")
    logging.info(f"Número de trials: {n_trials}")
    logging.info(f"Cross-validation folds: {cv_folds}")
    if timeout:
        logging.info(f"Timeout: {timeout} segundos")
    # Configurar storage para reanudar estudios (SQLite por defecto si se pasa ruta)
    resolved_storage = None
    if storage:
        if storage.startswith(("sqlite://", "postgresql://", "mysql://")):
            resolved_storage = storage
        else:
            resolved_storage = f"sqlite:///{Path(storage).absolute()}"
        logging.info(f"Usando storage de Optuna: {resolved_storage}")
    study_name = study_name or f"{model_type}_optimization"

    # Crear cross-validator con subject-level splitting
    cv = SubjectTimeSeriesSplit(n_splits=cv_folds, test_size=0.2)

    # Combinar train y val para cross-validation interna
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)
    groups_combined = pd.concat([groups_train, groups_val], ignore_index=True)

    # Para XGBoost, codificar etiquetas
    label_encoder = None
    if model_type == "xgboost":
        label_encoder = LabelEncoder()
        y_combined_encoded = label_encoder.fit_transform(y_combined)
    else:
        y_combined_encoded = y_combined

    def objective(trial: optuna.Trial) -> float:
        """Función objetivo para Optuna."""
        if model_type == "random_forest":
            # max_depth: incluir None (sin límite) como opción válida
            max_depth_choice = trial.suggest_categorical(
                "max_depth", [None, 10, 15, 20, 30, 40, 50]
            )
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": max_depth_choice,
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical(
                    "max_features", ["sqrt", "log2", None]
                ),
                "class_weight": trial.suggest_categorical(
                    "class_weight", [None, "balanced", "balanced_subsample"]
                ),
                "random_state": 42,
                "n_jobs": -1,
            }
            model = RandomForestClassifier(**params)

        elif model_type == "xgboost":
            if not XGB_AVAILABLE:
                raise ImportError("XGBoost no está disponible.")
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "eval_metric": "mlogloss",
            }
            model = xgb.XGBClassifier(**params)
        else:
            raise ValueError(f"Tipo de modelo desconocido: {model_type}")

        # Cross-validation con subject-level splitting
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            cv.split(X_combined, y_combined_encoded, groups_combined)
        ):
            X_fold_train = X_combined.iloc[train_idx]
            X_fold_val = X_combined.iloc[val_idx]

            if model_type == "xgboost":
                y_fold_train = y_combined_encoded[train_idx]
                y_fold_val = y_combined_encoded[val_idx]

                # Calcular sample_weight para balancear clases
                class_counts = pd.Series(y_fold_train).value_counts()
                total_samples = len(y_fold_train)
                n_classes = len(np.unique(y_fold_train))
                class_weights = {}
                for class_idx in range(n_classes):
                    class_count = class_counts.get(class_idx, 0)
                    if class_count > 0:
                        class_weights[class_idx] = total_samples / (
                            n_classes * class_count
                        )
                    else:
                        class_weights[class_idx] = 1.0
                sample_weight = np.array([class_weights[y] for y in y_fold_train])
                model.fit(X_fold_train, y_fold_train, sample_weight=sample_weight)
            else:
                y_fold_train = y_combined.iloc[train_idx]
                y_fold_val = y_combined.iloc[val_idx]
                model.fit(X_fold_train, y_fold_train)

            y_pred = model.predict(X_fold_val)
            score = f1_score(y_fold_val, y_pred, average="macro", zero_division=0)
            scores.append(score)

            # Pruning: reportar resultado intermedio para early stopping
            trial.report(np.mean(scores), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)

    # Crear estudio de Optuna con sampler TPE y pruner
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=1)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=resolved_storage,
        load_if_exists=resolved_storage is not None,
    )

    # Configurar logging de Optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logging.info("Ejecutando optimización bayesiana...")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
        n_jobs=1,  # No paralelizar trials, cada trial ya usa paralelismo interno
    )

    # Resultados
    best_params = study.best_params
    best_score = study.best_value

    logging.info("\n" + "=" * 60)
    logging.info("RESULTADOS DE OPTIMIZACIÓN BAYESIANA")
    logging.info("=" * 60)
    logging.info(f"Mejor score (F1-macro): {best_score:.4f}")
    logging.info(f"Trials completados: {len(study.trials)}")
    logging.info(
        f"Trials podados (pruned): {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
    )
    logging.info("Mejores parámetros encontrados:")
    for param, value in best_params.items():
        logging.info(f"  {param}: {value}")
    logging.info("=" * 60)

    # Preparar resultados para compatibilidad con el código existente
    return {
        "best_params": best_params,
        "best_score": best_score,
        "study": study,
        "n_trials": len(study.trials),
    }


def optimize_hyperparameters(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    groups_train: pd.Series,
    groups_val: pd.Series,
    n_iter: int = 50,
    cv_folds: int = 3,
    storage: Optional[str] = None,
    study_name: Optional[str] = None,
) -> dict:
    """Optimiza hiperparámetros usando optimización bayesiana (Optuna).

    Esta función es un wrapper que usa optimización bayesiana por defecto
    si Optuna está disponible, de lo contrario usa el enfoque manual.

    Parameters
    ----------
    model_type : str
        Tipo de modelo ('random_forest' o 'xgboost')
    X_train : pd.DataFrame
        Features de entrenamiento
    y_train : pd.Series
        Etiquetas de entrenamiento
    X_val : pd.DataFrame
        Features de validación
    y_val : pd.Series
        Etiquetas de validación
    groups_train : pd.Series
        Grupos (subject_core) para train
    groups_val : pd.Series
        Grupos (subject_core) para val
    n_iter : int
        Número de trials/iteraciones para optimización (default: 50)
    cv_folds : int
        Número de folds para cross-validation interna

    Returns
    -------
    dict
        Diccionario con mejores parámetros encontrados y resultados
    """
    # Usar optimización bayesiana si Optuna está disponible
    if OPTUNA_AVAILABLE:
        return optimize_hyperparameters_bayesian(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            groups_train=groups_train,
            groups_val=groups_val,
            n_trials=n_iter,
            cv_folds=cv_folds,
            storage=storage,
            study_name=study_name,
        )

    # Fallback: optimización manual básica si Optuna no está disponible
    logging.warning(
        "Optuna no disponible. Usando búsqueda manual básica. "
        "Para mejores resultados, instala Optuna: pip install optuna"
    )
    logging.info("=" * 60)
    logging.info("ETAPA: OPTIMIZACIÓN DE HIPERPARÁMETROS (Modo Fallback)")
    logging.info("=" * 60)

    # Combinar train y val para cross-validation interna
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)
    groups_combined = pd.concat([groups_train, groups_val], ignore_index=True)

    # Crear cross-validator
    cv = SubjectTimeSeriesSplit(n_splits=cv_folds, test_size=0.2)

    # Configuraciones predefinidas para probar
    if model_type == "random_forest":
        configs = [
            {"n_estimators": 200, "max_depth": None, "class_weight": "balanced"},
            {"n_estimators": 300, "max_depth": 20, "class_weight": "balanced"},
            {"n_estimators": 400, "max_depth": 30, "class_weight": "balanced"},
        ]
    elif model_type == "xgboost":
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost no está disponible.")
        configs = [
            {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
            {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05},
            {"n_estimators": 400, "max_depth": 7, "learning_rate": 0.1},
        ]
    else:
        raise ValueError(f"Tipo de modelo desconocido: {model_type}")

    best_score = -1
    best_params = configs[0]

    # Codificar etiquetas para XGBoost
    if model_type == "xgboost":
        le = LabelEncoder()
        y_combined_encoded = le.fit_transform(y_combined)
    else:
        y_combined_encoded = y_combined

    for config in configs:
        scores = []
        for train_idx, val_idx in cv.split(
            X_combined, y_combined_encoded, groups_combined
        ):
            X_fold_train = X_combined.iloc[train_idx]
            X_fold_val = X_combined.iloc[val_idx]

            if model_type == "random_forest":
                y_fold_train = y_combined.iloc[train_idx]
                y_fold_val = y_combined.iloc[val_idx]
                model = RandomForestClassifier(**config, random_state=42, n_jobs=-1)
                model.fit(X_fold_train, y_fold_train)
            else:
                y_fold_train = y_combined_encoded[train_idx]
                y_fold_val = y_combined_encoded[val_idx]
                model = xgb.XGBClassifier(
                    **config, random_state=42, n_jobs=-1, eval_metric="mlogloss"
                )
                model.fit(X_fold_train, y_fold_train)

            y_pred = model.predict(X_fold_val)
            score = f1_score(y_fold_val, y_pred, average="macro", zero_division=0)
            scores.append(score)

        mean_score = np.mean(scores)
        logging.info(f"Config {config}: F1-macro = {mean_score:.4f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = config

    logging.info(
        f"\nMejor configuración: {best_params} con F1-macro = {best_score:.4f}"
    )

    return {
        "best_params": best_params,
        "best_score": best_score,
    }


def optimize_hyperparameters_dl(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 30,
    timeout: Optional[int] = None,
    show_progress_bar: bool = True,
) -> dict:
    """Optimiza hiperparámetros de modelos Deep Learning usando Optuna.

    Esta función implementa optimización bayesiana con pruning para CNN1D y LSTM,
    aprovechando callbacks de Keras para terminar trials poco prometedores temprano.

    Parameters
    ----------
    model_type : str
        Tipo de modelo ('cnn1d' o 'lstm')
    X_train : np.ndarray
        Datos de entrenamiento
    y_train : np.ndarray
        Etiquetas de entrenamiento
    X_val : np.ndarray
        Datos de validación
    y_val : np.ndarray
        Etiquetas de validación
    n_trials : int
        Número de trials para optimización (default: 30)
    timeout : int, optional
        Tiempo máximo en segundos
    show_progress_bar : bool
        Mostrar barra de progreso

    Returns
    -------
    dict
        Diccionario con mejores parámetros y resultados
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna no está disponible. Instala con: pip install optuna")

    if not TF_AVAILABLE:
        raise ImportError("TensorFlow no está disponible.")

    # Forzar uso de CPU
    _configure_tensorflow_cpu_only()

    logging.info("=" * 60)
    logging.info(
        f"ETAPA: OPTIMIZACIÓN BAYESIANA DE HIPERPARÁMETROS - {model_type.upper()}"
    )
    logging.info("=" * 60)
    logging.info(f"Número de trials: {n_trials}")

    # Codificar etiquetas
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    n_classes = len(le.classes_)

    # Normalizar datos según el tipo de modelo
    if model_type == "cnn1d":
        # Normalizar por canal
        X_train_norm = np.zeros_like(X_train, dtype=np.float32)
        X_val_norm = np.zeros_like(X_val, dtype=np.float32)
        channel_stats = []
        for ch_idx in range(X_train.shape[1]):
            ch_data = X_train[:, ch_idx, :]
            mean = np.mean(ch_data)
            std = np.std(ch_data)
            channel_stats.append((mean, std))
            if std > 0:
                X_train_norm[:, ch_idx, :] = (ch_data - mean) / std
                X_val_norm[:, ch_idx, :] = (X_val[:, ch_idx, :] - mean) / std
            else:
                X_train_norm[:, ch_idx, :] = ch_data
                X_val_norm[:, ch_idx, :] = X_val[:, ch_idx, :]
        input_shape = (X_train.shape[1], X_train.shape[2])
    else:  # lstm
        # Normalizar con StandardScaler
        scaler = StandardScaler()
        n_sequences, sequence_length, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_train_norm = X_train_scaled.reshape(
            n_sequences, sequence_length, n_features
        ).astype(np.float32)

        n_val_sequences = X_val.shape[0]
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_val_scaled = scaler.transform(X_val_reshaped)
        X_val_norm = X_val_scaled.reshape(
            n_val_sequences, sequence_length, n_features
        ).astype(np.float32)
        input_shape = (sequence_length, n_features)

    # Calcular class weights
    from sklearn.utils.class_weight import compute_class_weight

    class_weights_array = compute_class_weight(
        "balanced", classes=np.unique(y_train_encoded), y=y_train_encoded
    )
    class_weight = dict(enumerate(class_weights_array))

    def objective(trial: optuna.Trial) -> float:
        """Función objetivo para Optuna - modelos DL."""
        # Limpiar sesión de Keras para evitar memory leaks
        keras.backend.clear_session()

        if model_type == "cnn1d":
            # Hiperparámetros para CNN1D
            n_filters = trial.suggest_int("n_filters", 32, 128, step=32)
            kernel_size = trial.suggest_int("kernel_size", 3, 7, step=2)
            dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.6)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

            model = build_cnn1d_model(
                input_shape=input_shape,
                n_classes=n_classes,
                n_filters=n_filters,
                kernel_size=kernel_size,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
            )

        else:  # lstm
            # Hiperparámetros para LSTM
            lstm_units = trial.suggest_int("lstm_units", 64, 256, step=32)
            dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.6)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

            model = build_lstm_model(
                input_shape=input_shape,
                n_classes=n_classes,
                lstm_units=lstm_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
            )

        # Callback de pruning de Optuna
        # Termina trials que no mejoran respecto a la mediana
        class OptunaPruningCallback(keras.callbacks.Callback):
            def __init__(self, trial, monitor="val_loss"):
                super().__init__()
                self.trial = trial
                self.monitor = monitor

            def on_epoch_end(self, epoch, logs=None):
                current_value = logs.get(self.monitor)
                if current_value is None:
                    return
                # Para val_loss, menor es mejor, así que negamos para maximizar
                # Optuna maximiza, pero early stopping en val_loss queremos minimizar
                self.trial.report(-current_value, epoch)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=0,
            ),
            OptunaPruningCallback(trial, monitor="val_loss"),
        ]

        # Entrenar modelo
        model.fit(
            X_train_norm,
            y_train_encoded,
            batch_size=batch_size,
            epochs=30,  # Máximo de epochs para cada trial
            validation_data=(X_val_norm, y_val_encoded),
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluar en validación
        y_pred = np.argmax(model.predict(X_val_norm, verbose=0), axis=1)
        score = f1_score(y_val_encoded, y_pred, average="macro", zero_division=0)

        return score

    # Crear estudio de Optuna
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=5)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"{model_type}_dl_optimization",
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logging.info("Ejecutando optimización bayesiana para Deep Learning...")
    logging.info("(Esto puede tomar varios minutos)")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
        n_jobs=1,
    )

    # Resultados
    best_params = study.best_params
    best_score = study.best_value

    logging.info("\n" + "=" * 60)
    logging.info(f"RESULTADOS DE OPTIMIZACIÓN - {model_type.upper()}")
    logging.info("=" * 60)
    logging.info(f"Mejor score (F1-macro): {best_score:.4f}")
    logging.info(f"Trials completados: {len(study.trials)}")
    logging.info(
        f"Trials podados: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
    )
    logging.info("Mejores parámetros encontrados:")
    for param, value in best_params.items():
        logging.info(f"  {param}: {value}")
    logging.info("=" * 60)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "study": study,
        "n_trials": len(study.trials),
    }
