"""Utilidades base y funciones comunes para modelos de clasificación de sueño.

Este módulo contiene:
- Constantes globales (STAGE_ORDER)
- Funciones de evaluación (evaluate_model, print_evaluation_report)
- Funciones de guardado/carga (save_model, load_model, save_metrics, load_metrics)
- Configuración de TensorFlow
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import LabelEncoder

# Estadios en orden estándar
STAGE_ORDER = ["W", "N1", "N2", "N3", "REM"]

# Intentar importar TensorFlow (opcional para modelos tradicionales)
try:
    from tensorflow import keras

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    keras = None
    logging.warning(
        "TensorFlow no está disponible. Los modelos de deep learning no funcionarán."
    )

# Intentar importar Optuna (opcional, pero recomendado para optimización)
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
    TPESampler = None
    MedianPruner = None
    logging.info(
        "Optuna no esta instalado. La optimizacion de hiperparametros usara "
        "busqueda manual basica. Para mejor rendimiento, instala Optuna: "
        "pip install optuna"
    )


def _configure_tensorflow_cpu_only() -> None:
    """Configura TensorFlow para usar solo CPU.

    Esta función deshabilita GPU/Metal para evitar problemas en macOS,
    especialmente bus errors y segmentation faults en Apple Silicon.

    Debe llamarse al inicio de funciones que usan TensorFlow.
    """
    import os

    import tensorflow as tf

    os.environ["TF_METAL_PLUGIN_LIBRARY_PATH"] = ""
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    tf.config.set_visible_devices([], "GPU")  # Deshabilitar GPU/Metal


def evaluate_model(
    model,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    stage_order: Optional[list[str]] = None,
    dataset_name: str = "TEST",
) -> dict:
    """Evalúa un modelo y retorna métricas.

    Parameters
    ----------
    model
        Modelo entrenado (sklearn, XGBoost, o Keras)
    X_test : pd.DataFrame | np.ndarray
        Features de test (DataFrame para modelos tradicionales, array para DL)
    y_test : pd.Series | np.ndarray
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
    if stage_order is None:
        stage_order = STAGE_ORDER

    logging.info("=" * 60)
    logging.info(f"ETAPA: EVALUACIÓN EN {dataset_name}")
    logging.info("=" * 60)
    logging.info(f"Muestras de {dataset_name.lower()}: {len(y_test)}")
    if isinstance(y_test, pd.Series):
        logging.info(
            f"Distribución de clases en {dataset_name.lower()}:\n"
            f"{y_test.value_counts().sort_index()}"
        )
    else:
        logging.info(
            f"Distribución de clases en {dataset_name.lower()}:\n"
            f"{pd.Series(y_test).value_counts().sort_index()}"
        )
    logging.info("Generando predicciones...")

    # Detectar si es modelo de Keras
    is_keras_model = TF_AVAILABLE and isinstance(model, keras.Model)

    if is_keras_model:
        # Modelo de Keras: necesita normalización y decodificación
        logging.info("Procesando predicciones de modelo Keras...")

        # Normalizar datos de test según el tipo de modelo
        if hasattr(model, "scaler_"):
            # LSTM: usar scaler guardado
            n_sequences, sequence_length, n_features = X_test.shape

            # Validar dimensiones
            if n_features != model.scaler_.n_features_in_:
                raise ValueError(
                    f"Dimensiones no coinciden: X_test tiene {n_features} features, "
                    f"pero el scaler espera {model.scaler_.n_features_in_} features"
                )

            X_test_reshaped = X_test.reshape(-1, n_features)
            X_test_scaled = model.scaler_.transform(X_test_reshaped)
            X_test_norm = X_test_scaled.reshape(
                n_sequences, sequence_length, n_features
            )
        elif hasattr(model, "channel_means_") and hasattr(model, "channel_stds_"):
            # CNN1D: normalizar por canal usando estadísticas de train guardadas
            # Validar que las dimensiones coinciden
            if X_test.shape[1] != len(model.channel_means_):
                raise ValueError(
                    f"Dimensiones no coinciden: X_test tiene {X_test.shape[1]} canales, "
                    f"pero el modelo espera {len(model.channel_means_)} canales"
                )

            # Usar estadísticas de train guardadas
            X_test_norm = np.zeros_like(X_test)
            for ch_idx in range(X_test.shape[1]):
                ch_data = X_test[:, ch_idx, :]
                mean = model.channel_means_[ch_idx]
                std = model.channel_stds_[ch_idx]
                if std > 0:
                    X_test_norm[:, ch_idx, :] = (ch_data - mean) / std
                else:
                    X_test_norm[:, ch_idx, :] = ch_data
        else:
            raise ValueError(
                "El modelo Keras no tiene estadísticas de normalización guardadas "
                "(scaler_ o channel_means_/channel_stds_). Reentrena o guarda el modelo "
                "con dichas estadísticas para evitar data leakage."
            )

        # Predecir (retorna probabilidades)
        y_pred_proba = model.predict(X_test_norm, verbose=0)
        # Tomar clase con mayor probabilidad
        y_pred_encoded = np.argmax(y_pred_proba, axis=1)

        # Decodificar usando label_encoder
        if hasattr(model, "label_encoder_"):
            y_pred = model.label_encoder_.inverse_transform(y_pred_encoded)
        else:
            y_pred = model.classes_[y_pred_encoded]
    else:
        # Modelo tradicional (sklearn/XGBoost)
        y_pred = model.predict(X_test)

        # Si es XGBoost, puede necesitar decodificación
        if (
            hasattr(model, "classes_")
            and len(y_pred) > 0
            and isinstance(y_pred[0], (int, np.integer))
        ):
            logging.info("Decodificando predicciones (XGBoost)...")
            if hasattr(model, "label_encoder_"):
                y_pred = model.label_encoder_.inverse_transform(y_pred)
            else:
                # Validar que los índices estén en rango válido
                if np.any(y_pred < 0) or np.any(y_pred >= len(model.classes_)):
                    invalid_indices = np.where(
                        (y_pred < 0) | (y_pred >= len(model.classes_))
                    )[0]
                    logging.warning(
                        f"Advertencia: {len(invalid_indices)} predicciones con "
                        f"índices fuera de rango."
                    )
                    y_pred = np.clip(y_pred, 0, len(model.classes_) - 1)
                y_pred = model.classes_[y_pred]

    logging.info("Calculando métricas de evaluación...")
    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # F1 por clase
    f1_per_class = f1_score(
        y_test, y_pred, average=None, labels=stage_order, zero_division=0
    )
    f1_dict = {stage: float(score) for stage, score in zip(stage_order, f1_per_class)}

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=stage_order)

    # Reporte de clasificación
    report = classification_report(
        y_test, y_pred, labels=stage_order, output_dict=True, zero_division=0
    )

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
    with open(path) as f:
        metrics = json.load(f)
    logging.info(f"Métricas cargadas desde {path}")
    return metrics


def save_model(model, path: Path | str) -> None:
    """Guarda un modelo entrenado.

    Parameters
    ----------
    model
        Modelo entrenado (sklearn, XGBoost, o Keras)
    path : Path | str
        Ruta donde guardar el modelo

    Warnings
    --------
    Emite warnings si faltan atributos críticos para la correcta
    evaluación del modelo (label_encoder_, estadísticas de normalización).
    """
    path = Path(path)

    # Detectar si es modelo de Keras
    is_keras_model = TF_AVAILABLE and isinstance(model, keras.Model)

    # Verificar atributos críticos y emitir warnings
    if is_keras_model:
        if not hasattr(model, "label_encoder_"):
            logging.warning(
                "Modelo Keras sin label_encoder_: la decodificación de predicciones "
                "puede fallar al cargar el modelo."
            )
        # Verificar estadísticas de normalización según el tipo de modelo
        has_cnn_stats = hasattr(model, "channel_means_") and hasattr(
            model, "channel_stds_"
        )
        has_lstm_stats = hasattr(model, "scaler_")
        if not has_cnn_stats and not has_lstm_stats:
            logging.warning(
                "Modelo Keras sin estadísticas de normalización (channel_means_/stds_ "
                "o scaler_): evaluate_model fallará al evaluar en nuevos datos."
            )

    if is_keras_model:
        # Guardar modelo de Keras
        save_path = Path(path)
        if not save_path.suffix:
            save_path = save_path.with_suffix(".keras")
        elif save_path.suffix not in [".keras", ".h5"]:
            save_path = save_path.with_suffix(".keras")
        model.save(str(save_path))

        # Guardar atributos personalizados por separado
        custom_attrs = {}
        if hasattr(model, "label_encoder_"):
            custom_attrs["label_encoder_classes_"] = (
                model.label_encoder_.classes_.tolist()
            )
        if hasattr(model, "classes_"):
            custom_attrs["classes_"] = (
                model.classes_.tolist()
                if isinstance(model.classes_, np.ndarray)
                else list(model.classes_)
            )
        if hasattr(model, "channel_means_"):
            custom_attrs["channel_means_"] = model.channel_means_.tolist()
        if hasattr(model, "channel_stds_"):
            custom_attrs["channel_stds_"] = model.channel_stds_.tolist()
        if hasattr(model, "scaler_"):
            scaler_path = save_path.parent / f"{save_path.stem}_scaler.pkl"
            with open(scaler_path, "wb") as f:
                pickle.dump(model.scaler_, f)
            custom_attrs["scaler_path"] = str(scaler_path)
        if hasattr(model, "history_"):
            history_serializable = {}
            for key, values in model.history_.items():
                if isinstance(values, (list, np.ndarray)):
                    history_serializable[key] = [
                        float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for v in values
                    ]
                else:
                    history_serializable[key] = values

            history_path = save_path.parent / f"{save_path.stem}_history.json"
            with open(history_path, "w") as f:
                json.dump(history_serializable, f, indent=2)
            custom_attrs["history_path"] = str(history_path)

        if custom_attrs:
            custom_attrs_path = save_path.parent / f"{save_path.stem}_custom_attrs.json"
            with open(custom_attrs_path, "w") as f:
                json.dump(custom_attrs, f, indent=2)
            logging.info(f"Atributos personalizados guardados en {custom_attrs_path}")

        logging.info(f"Modelo Keras guardado en {path}")
    else:
        # Guardar modelo tradicional con pickle
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
    path = Path(path)

    # Intentar cargar como modelo de Keras primero
    if TF_AVAILABLE:
        keras_path = path
        if path.is_dir():
            keras_path = path
        elif path.suffix in [".keras", ".h5"]:
            keras_path = path
        elif not path.suffix:
            keras_path = path.with_suffix(".keras")
        else:
            keras_path = None

        if keras_path and (keras_path.is_dir() or keras_path.exists()):
            try:
                model = keras.models.load_model(str(keras_path))

                # Cargar atributos personalizados si existen
                attrs_name = (
                    keras_path.stem if keras_path.is_file() else keras_path.name
                )
                custom_attrs_path = (
                    keras_path.parent / f"{attrs_name}_custom_attrs.json"
                )

                if custom_attrs_path.exists():
                    with open(custom_attrs_path) as f:
                        custom_attrs = json.load(f)

                    # Restaurar LabelEncoder
                    if "label_encoder_classes_" in custom_attrs:
                        le = LabelEncoder()
                        le.classes_ = np.array(custom_attrs["label_encoder_classes_"])
                        model.label_encoder_ = le

                    # Restaurar clases
                    if "classes_" in custom_attrs:
                        model.classes_ = np.array(custom_attrs["classes_"])

                    # Restaurar estadísticas de normalización de CNN1D
                    if "channel_means_" in custom_attrs:
                        model.channel_means_ = np.array(custom_attrs["channel_means_"])
                    if "channel_stds_" in custom_attrs:
                        model.channel_stds_ = np.array(custom_attrs["channel_stds_"])

                    # Restaurar scaler de LSTM
                    if "scaler_path" in custom_attrs:
                        scaler_path = Path(custom_attrs["scaler_path"])
                        if scaler_path.exists():
                            with open(scaler_path, "rb") as f:
                                model.scaler_ = pickle.load(f)  # nosec B301

                    # Restaurar historial de entrenamiento
                    if "history_path" in custom_attrs:
                        history_path = Path(custom_attrs["history_path"])
                        if history_path.exists():
                            with open(history_path) as f:
                                model.history_ = json.load(f)

                logging.info(f"Modelo Keras cargado desde {keras_path}")
                return model
            except Exception as e:
                logging.error(f"Error cargando modelo Keras desde {path}: {e}")
                logging.info("Intentando cargar como modelo tradicional...")

    # Cargar modelo tradicional con pickle
    with open(path, "rb") as f:
        model = pickle.load(f)  # nosec B301
    logging.info(f"Modelo cargado desde {path}")
    return model
