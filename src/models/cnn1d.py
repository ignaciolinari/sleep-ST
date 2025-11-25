"""Modelo CNN1D para clasificación de estadios de sueño.

Este módulo contiene las funciones para construir y entrenar modelos CNN1D
que procesan señales EEG/EOG/EMG crudas para clasificar estadios de sueño.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .base import TF_AVAILABLE, _configure_tensorflow_cpu_only

if TF_AVAILABLE:
    from tensorflow import keras
    from tensorflow.keras import layers


def build_cnn1d_model(
    input_shape: tuple[int, int],
    n_classes: int = 5,
    n_filters: int = 64,
    kernel_size: int = 3,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
    use_residual: bool = True,
    use_augmentation: bool = True,
) -> "keras.Model":
    """Construye modelo CNN1D mejorado para clasificación de estadios de sueño.

    Arquitectura mejorada con:
    - Data augmentation (GaussianNoise durante entrenamiento)
    - Conexiones residuales opcionales para mejor gradiente flow
    - Capas convolucionales 1D para extraer patrones locales
    - Pooling para reducir dimensionalidad
    - Regularización L2 en capas densas
    - Capas densas para clasificación final

    Parameters
    ----------
    input_shape : tuple[int, int]
        Forma de entrada (n_channels, n_samples_per_epoch)
    n_classes : int
        Número de clases (estadios de sueño)
    n_filters : int
        Número de filtros en las capas convolucionales
    kernel_size : int
        Tamaño del kernel convolucional
    dropout_rate : float
        Tasa de dropout para regularización
    learning_rate : float
        Tasa de aprendizaje del optimizador
    use_residual : bool
        Usar conexiones residuales (mejora gradiente flow)
    use_augmentation : bool
        Usar data augmentation (GaussianNoise durante entrenamiento)

    Returns
    -------
    keras.Model
        Modelo CNN1D compilado
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow no está disponible.")

    # Input: (n_channels, n_samples_per_epoch)
    input_layer = keras.Input(shape=input_shape, name="input")

    # Transponer para que las convoluciones operen sobre el tiempo
    # De (n_channels, n_samples) a (n_samples, n_channels)
    x = layers.Permute((2, 1))(input_layer)

    # Data augmentation: añadir ruido gaussiano durante entrenamiento
    if use_augmentation:
        x = layers.GaussianNoise(0.1)(x)  # Solo activo durante training

    # Primera capa convolucional
    x = layers.Conv1D(
        filters=n_filters,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="conv1d_1",
    )(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.MaxPooling1D(pool_size=2, name="maxpool_1")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)

    # Segunda capa convolucional con conexión residual opcional
    conv2_input = x
    x = layers.Conv1D(
        filters=n_filters * 2,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="conv1d_2",
    )(x)
    x = layers.BatchNormalization(name="bn_2")(x)

    if use_residual:
        # Ajustar dimensiones para la conexión residual
        conv2_input_adjusted = layers.Conv1D(
            filters=n_filters * 2,
            kernel_size=1,
            padding="same",
            name="residual_adjust_1",
        )(conv2_input)
        x = layers.Add(name="residual_add_1")([x, conv2_input_adjusted])
        x = layers.Activation("relu")(x)

    x = layers.MaxPooling1D(pool_size=2, name="maxpool_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)

    # Tercera capa convolucional con conexión residual opcional
    conv3_input = x
    x = layers.Conv1D(
        filters=n_filters * 4,
        kernel_size=kernel_size,
        activation="relu",
        padding="same",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="conv1d_3",
    )(x)
    x = layers.BatchNormalization(name="bn_3")(x)

    if use_residual:
        conv3_input_adjusted = layers.Conv1D(
            filters=n_filters * 4,
            kernel_size=1,
            padding="same",
            name="residual_adjust_2",
        )(conv3_input)
        x = layers.Add(name="residual_add_2")([x, conv3_input_adjusted])
        x = layers.Activation("relu")(x)

    # Global pooling
    x = layers.GlobalAveragePooling1D(name="global_avg_pool")(x)

    # Capas densas para clasificación con regularización L2
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="dense_1",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_3")(x)
    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name="dense_2",
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_4")(x)

    # Capa de salida
    output_layer = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(
        inputs=input_layer, outputs=output_layer, name="CNN1D_SleepStaging"
    )

    # Compilar modelo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_cnn1d(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    n_filters: int = 64,
    kernel_size: int = 3,
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    epochs: int = 50,
    verbose: int = 1,
    class_weight: Optional[dict] = None,
) -> "keras.Model":
    """Entrena un modelo CNN1D.

    Parameters
    ----------
    X_train : np.ndarray
        Señales raw de entrenamiento de forma (n_samples, n_channels, n_samples_per_epoch)
    y_train : np.ndarray
        Etiquetas de entrenamiento (strings: W, N1, N2, N3, REM)
    X_val : np.ndarray, optional
        Señales raw de validación
    y_val : np.ndarray, optional
        Etiquetas de validación
    n_filters : int
        Número de filtros en las capas convolucionales
    kernel_size : int
        Tamaño del kernel convolucional
    dropout_rate : float
        Tasa de dropout
    learning_rate : float
        Tasa de aprendizaje
    batch_size : int
        Tamaño del batch
    epochs : int
        Número de épocas
    verbose : int
        Verbosidad (0=silencioso, 1=barra de progreso, 2=una línea por época)
    class_weight : dict, optional
        Pesos por clase para balancear el dataset

    Returns
    -------
    keras.Model
        Modelo CNN1D entrenado
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow no está disponible.")

    # Forzar uso de CPU para evitar problemas con Metal en macOS
    _configure_tensorflow_cpu_only()

    logging.info("=" * 60)
    logging.info("ETAPA: ENTRENAMIENTO - CNN1D")
    logging.info("=" * 60)
    logging.info("Configuración del modelo:")
    logging.info(f"  - Filtros: {n_filters}")
    logging.info(f"  - Kernel size: {kernel_size}")
    logging.info(f"  - Dropout rate: {dropout_rate}")
    logging.info(f"  - Learning rate: {learning_rate}")
    logging.info(f"  - Batch size: {batch_size}")
    logging.info(f"  - Epochs: {epochs}")
    logging.info("Datos de entrenamiento:")
    logging.info(f"  - Muestras: {len(X_train)}")
    logging.info(f"  - Forma de entrada: {X_train.shape[1:]}")
    logging.info(
        f"  - Distribución de clases:\n{pd.Series(y_train).value_counts().sort_index()}"
    )

    # Codificar etiquetas
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    n_classes = len(le.classes_)

    # Normalizar señales (por canal)
    # Normalizar cada canal independientemente
    # Guardar estadísticas de normalización para usar en test
    X_train_norm = np.zeros_like(X_train)
    channel_means = []
    channel_stds = []
    for ch_idx in range(X_train.shape[1]):
        ch_data = X_train[:, ch_idx, :]
        mean = np.mean(ch_data)
        std = np.std(ch_data)
        channel_means.append(mean)
        channel_stds.append(std)
        if std > 0:
            X_train_norm[:, ch_idx, :] = (ch_data - mean) / std
        else:
            X_train_norm[:, ch_idx, :] = ch_data

    # Construir modelo
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn1d_model(
        input_shape=input_shape,
        n_classes=n_classes,
        n_filters=n_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    # Guardar LabelEncoder y estadísticas de normalización en el modelo
    model.label_encoder_ = le
    model.classes_ = le.classes_
    model.channel_means_ = np.array(channel_means)
    model.channel_stds_ = np.array(channel_stds)

    # Preparar datos de validación si existen
    validation_data = None
    if X_val is not None and y_val is not None:
        y_val_encoded = le.transform(y_val)
        X_val_norm = np.zeros_like(X_val)
        # Usar estadísticas de train guardadas para normalizar val
        for ch_idx in range(X_val.shape[1]):
            ch_data = X_val[:, ch_idx, :]
            mean = model.channel_means_[ch_idx]
            std = model.channel_stds_[ch_idx]
            if std > 0:
                X_val_norm[:, ch_idx, :] = (ch_data - mean) / std
            else:
                X_val_norm[:, ch_idx, :] = ch_data
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
        f"✓ Entrenamiento completado: CNN1D entrenado con {len(X_train)} muestras"
    )
    logging.info("=" * 60)

    return model
