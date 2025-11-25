"""Modelo LSTM para clasificación de estadios de sueño.

Este módulo contiene las funciones para construir y entrenar modelos LSTM
que procesan secuencias de features extraídas para clasificar estadios de sueño.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .base import TF_AVAILABLE, _configure_tensorflow_cpu_only

if TF_AVAILABLE:
    from tensorflow import keras
    from tensorflow.keras import layers


def build_lstm_model(
    input_shape: tuple[int, int],
    n_classes: int = 5,
    lstm_units: int = 128,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
    bidirectional: bool = True,
    use_attention: bool = False,
) -> "keras.Model":
    """Construye modelo LSTM mejorado para clasificación de estadios de sueño.

    Arquitectura mejorada con:
    - LSTM Bidireccional opcional para capturar contexto pasado y futuro
    - Regularización L2 en capas densas
    - Dropout entre capas (no recurrent_dropout para mejor rendimiento)
    - Capas densas para clasificación final

    NOTA: Se eliminó recurrent_dropout porque deshabilita cuDNN y ralentiza
    el entrenamiento significativamente. Se usa dropout estándar entre capas.

    Parameters
    ----------
    input_shape : tuple[int, int]
        Forma de entrada (sequence_length, n_features)
    n_classes : int
        Número de clases (estadios de sueño)
    lstm_units : int
        Número de unidades LSTM
    dropout_rate : float
        Tasa de dropout para regularización
    learning_rate : float
        Tasa de aprendizaje del optimizador
    bidirectional : bool
        Usar LSTM bidireccional (captura contexto hacia adelante y atrás)
    use_attention : bool
        Usar mecanismo de atención simple (experimental)

    Returns
    -------
    keras.Model
        Modelo LSTM compilado
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow no está disponible.")

    # Input: (sequence_length, n_features)
    input_layer = keras.Input(shape=input_shape, name="input")

    # Primera capa LSTM (retorna secuencias completas)
    lstm_1 = layers.LSTM(
        lstm_units,
        return_sequences=True,
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="lstm_1",
    )

    if bidirectional:
        x = layers.Bidirectional(lstm_1, name="bidirectional_1")(input_layer)
    else:
        x = lstm_1(input_layer)

    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_lstm_1")(x)

    # Segunda capa LSTM (retorna solo el último estado)
    lstm_2 = layers.LSTM(
        lstm_units // 2,
        return_sequences=False,
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="lstm_2",
    )

    if bidirectional:
        x = layers.Bidirectional(lstm_2, name="bidirectional_2")(x)
    else:
        x = lstm_2(x)

    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_lstm_2")(x)

    # Capas densas para clasificación con regularización L2
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="dense_1",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="dense_2",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    # Capa de salida
    output_layer = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(
        inputs=input_layer,
        outputs=output_layer,
        name="BiLSTM_SleepStaging" if bidirectional else "LSTM_SleepStaging",
    )

    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    lstm_units: int = 128,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 50,
    verbose: int = 1,
    class_weight: Optional[dict] = None,
) -> "keras.Model":
    """Entrena un modelo LSTM.

    Parameters
    ----------
    X_train : np.ndarray
        Secuencias de features de entrenamiento de forma (n_sequences, sequence_length, n_features)
    y_train : np.ndarray
        Etiquetas de entrenamiento (strings: W, N1, N2, N3, REM)
    X_val : np.ndarray, optional
        Secuencias de features de validación
    y_val : np.ndarray, optional
        Etiquetas de validación
    lstm_units : int
        Número de unidades LSTM
    dropout_rate : float
        Tasa de dropout
    learning_rate : float
        Tasa de aprendizaje
    batch_size : int
        Tamaño del batch
    epochs : int
        Número de épocas
    verbose : int
        Verbosidad
    class_weight : dict, optional
        Pesos por clase para balancear el dataset

    Returns
    -------
    keras.Model
        Modelo LSTM entrenado
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow no está disponible.")

    # Forzar uso de CPU para evitar problemas con Metal en macOS
    _configure_tensorflow_cpu_only()

    logging.info("=" * 60)
    logging.info("ETAPA: ENTRENAMIENTO - LSTM")
    logging.info("=" * 60)
    logging.info("Configuración del modelo:")
    logging.info(f"  - LSTM units: {lstm_units}")
    logging.info(f"  - Dropout rate: {dropout_rate}")
    logging.info(f"  - Learning rate: {learning_rate}")
    logging.info(f"  - Batch size: {batch_size}")
    logging.info(f"  - Epochs: {epochs}")
    logging.info("Datos de entrenamiento:")
    logging.info(f"  - Secuencias: {len(X_train)}")
    logging.info(f"  - Forma de entrada: {X_train.shape[1:]}")
    logging.info(
        f"  - Distribución de clases:\n{pd.Series(y_train).value_counts().sort_index()}"
    )

    # Codificar etiquetas
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    n_classes = len(le.classes_)

    # Normalizar features (usar StandardScaler)
    scaler = StandardScaler()
    # Reshape para normalizar: (n_sequences * sequence_length, n_features)
    n_sequences, sequence_length, n_features = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_features)
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_norm = X_train_scaled.reshape(n_sequences, sequence_length, n_features)

    # Construir modelo
    input_shape = (sequence_length, n_features)
    model = build_lstm_model(
        input_shape=input_shape,
        n_classes=n_classes,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    # Guardar LabelEncoder y scaler en el modelo
    model.label_encoder_ = le
    model.classes_ = le.classes_
    model.scaler_ = scaler

    # Preparar datos de validación si existen
    validation_data = None
    if X_val is not None and y_val is not None:
        y_val_encoded = le.transform(y_val)
        n_val_sequences = X_val.shape[0]
        X_val_reshaped = X_val.reshape(-1, n_features)
        X_val_scaled = scaler.transform(X_val_reshaped)
        X_val_norm = X_val_scaled.reshape(n_val_sequences, sequence_length, n_features)
        validation_data = (X_val_norm, y_val_encoded)

    # Calcular class_weight si no se proporciona
    if class_weight is None:
        from sklearn.utils.class_weight import compute_class_weight

        class_weights = compute_class_weight(
            "balanced", classes=np.unique(y_train_encoded), y=y_train_encoded
        )
        class_weight = dict(enumerate(class_weights))
        logging.info(f"  - Pesos de clases calculados: {class_weight}")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss" if validation_data else "loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss" if validation_data else "loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    logging.info("Iniciando entrenamiento...")
    history = model.fit(
        X_train_norm,
        y_train_encoded,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=verbose,
    )

    # Guardar historial de entrenamiento en el modelo para análisis posterior
    model.history_ = history.history

    logging.info(
        f"✓ Entrenamiento completado: LSTM entrenado con {len(X_train)} secuencias"
    )
    logging.info("=" * 60)

    return model
