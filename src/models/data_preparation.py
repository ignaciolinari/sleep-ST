"""Preparación de datos para entrenamiento de modelos.

Este módulo contiene funciones para:
- Preparar datasets de features desde sesiones PSG
- Preparar datasets de epochs raw para modelos DL
- Preparar secuencias para modelos LSTM
- División train/test/val respetando sujetos
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import (
    assign_stages_to_epochs,
    create_epochs,
    extract_features_from_session,
    load_hypnogram,
    load_psg_data,
)
from src.models.base import STAGE_ORDER, TF_AVAILABLE


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
                f"Archivos faltantes para {row['subject_id']} "
                f"(buscado en {psg_path}), saltando"
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
                subject_id_str = str(row["subject_id"])
                features_df["subject_core"] = (
                    subject_id_str[:5] if len(subject_id_str) >= 5 else subject_id_str
                )
                features_df["session_idx"] = idx
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
    combined = combined.sort_values(
        ["subject_core", "subject_id", "epoch_time_start"]
    ).reset_index(drop=True)
    logging.info(
        f"Dataset completo: {len(combined)} epochs de "
        f"{combined['subject_id'].nunique()} sujetos"
    )

    return combined


def prepare_raw_epochs_dataset(
    manifest_path: Path | str,
    limit: Optional[int] = None,
    epoch_length: float = 30.0,
    sfreq: Optional[float] = None,
    channels: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Prepara dataset de epochs raw (señales) para modelos de deep learning.

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
    channels : list[str], optional
        Canales a usar. Si None, usa DEFAULT_CHANNELS.

    Returns
    -------
    X_raw : np.ndarray
        Array de forma (n_epochs, n_channels, n_samples_per_epoch) con señales raw
    y : np.ndarray
        Array con etiquetas de estadios (W, N1, N2, N3, REM)
    metadata_df : pd.DataFrame
        DataFrame con metadata de cada epoch (subject_id, subject_core, etc.)
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow no está disponible. Instala TensorFlow para usar "
            "modelos de deep learning."
        )

    manifest = pd.read_csv(manifest_path)
    manifest_ok = manifest[manifest["status"] == "ok"].copy()

    if limit:
        manifest_ok = manifest_ok.head(limit)
        logging.info(f"Procesando solo {len(manifest_ok)} sesiones (modo limit)")

    all_epochs = []
    all_stages = []
    all_metadata = []

    for idx, row in manifest_ok.iterrows():
        psg_path_str = row.get("psg_trimmed_path", "")
        hyp_path_str = row.get("hypnogram_trimmed_path", "")

        psg_path = Path(psg_path_str) if psg_path_str else None
        hyp_path = Path(hyp_path_str) if hyp_path_str else None

        if (
            not psg_path
            or not hyp_path
            or not psg_path.exists()
            or not hyp_path.exists()
        ):
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
                f"Archivos faltantes para {row['subject_id']} "
                f"(buscado en {psg_path}), saltando"
            )
            continue

        try:
            # Cargar datos raw
            data, actual_sfreq, ch_names = load_psg_data(psg_path, channels, sfreq)
            hypnogram = load_hypnogram(hyp_path)

            # Crear epochs
            epochs, epochs_times = create_epochs(
                data, actual_sfreq, epoch_length=epoch_length
            )

            if len(epochs) == 0:
                continue

            # Asignar estadios
            stages = assign_stages_to_epochs(epochs_times, hypnogram, epoch_length)

            # Filtrar epochs válidos (con estadio)
            for epoch_idx, (epoch, stage, epoch_time) in enumerate(
                zip(epochs, stages, epochs_times)
            ):
                if stage is None or stage not in STAGE_ORDER:
                    continue

                all_epochs.append(epoch)
                all_stages.append(stage)
                all_metadata.append(
                    {
                        "subject_id": row["subject_id"],
                        "subject_core": str(row["subject_id"])[:5]
                        if len(str(row["subject_id"])) >= 5
                        else str(row["subject_id"]),
                        "session_idx": idx,
                        "epoch_time_start": epoch_time,
                        "epoch_index": epoch_idx,
                    }
                )

            logging.info(
                f"Extraídos {len([s for s in stages if s is not None])} "
                f"epochs raw de {row['subject_id']}"
            )
        except Exception as e:
            logging.exception(f"Error procesando {row['subject_id']}: {e}")
            continue

    if not all_epochs:
        raise ValueError("No se pudieron extraer epochs raw de ninguna sesión")

    # Convertir a arrays numpy
    X_raw = np.array(all_epochs)
    y = np.array(all_stages)

    # Crear DataFrame de metadata
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df = metadata_df.sort_values(
        ["subject_core", "subject_id", "epoch_time_start"]
    ).reset_index(drop=True)

    logging.info(
        f"Dataset raw completo: {len(X_raw)} epochs de "
        f"{metadata_df['subject_id'].nunique()} sujetos"
    )
    logging.info(f"Forma de X_raw: {X_raw.shape}")
    logging.info(
        f"Distribución de estadios:\n{pd.Series(y).value_counts().sort_index()}"
    )

    return X_raw, y, metadata_df


def prepare_sequence_dataset(
    features_df: pd.DataFrame,
    sequence_length: int = 5,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Prepara dataset de secuencias de features para modelos LSTM.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame con features extraídas
    sequence_length : int
        Longitud de cada secuencia (número de epochs consecutivos)
    stride : int
        Paso entre secuencias

    Returns
    -------
    X_seq : np.ndarray
        Array de forma (n_sequences, sequence_length, n_features)
    y : np.ndarray
        Array con etiquetas del último epoch de cada secuencia
    metadata_df : pd.DataFrame
        DataFrame con metadata de cada secuencia
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow no está disponible. Instala TensorFlow para usar "
            "modelos de deep learning."
        )

    # Identificar columnas de features
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

    # Ordenar por sujeto y tiempo
    features_sorted = features_df.sort_values(
        ["subject_core", "subject_id", "epoch_time_start"]
    ).reset_index(drop=True)

    # Filtrar estadios válidos
    valid_stages = set(STAGE_ORDER)
    mask = features_sorted["stage"].isin(valid_stages)
    features_sorted = features_sorted[mask].reset_index(drop=True)

    # Crear secuencias respetando límites de sesión/sujeto
    sequences = []
    labels = []
    metadata_list = []

    for (subject_id, session_idx), group in features_sorted.groupby(
        ["subject_id", "session_idx"]
    ):
        group_features = group[feature_cols].values
        group_stages = group["stage"].values
        group_indices = group.index.values

        for i in range(0, len(group_features) - sequence_length + 1, stride):
            seq = group_features[i : i + sequence_length]
            label = group_stages[i + sequence_length - 1]

            sequences.append(seq)
            labels.append(label)

            metadata_list.append(
                {
                    "subject_id": subject_id,
                    "subject_core": group.iloc[i + sequence_length - 1]["subject_core"],
                    "session_idx": session_idx,
                    "sequence_start_idx": group_indices[i],
                    "sequence_end_idx": group_indices[i + sequence_length - 1],
                }
            )

    if not sequences:
        raise ValueError("No se pudieron crear secuencias válidas")

    X_seq = np.array(sequences)
    y = np.array(labels)
    metadata_df = pd.DataFrame(metadata_list)

    logging.info(
        f"Dataset de secuencias: {len(X_seq)} secuencias de longitud {sequence_length}"
    )
    logging.info(f"Forma de X_seq: {X_seq.shape}")
    logging.info(
        f"Distribución de estadios:\n{pd.Series(y).value_counts().sort_index()}"
    )

    return X_seq, y, metadata_df


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

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame con features
    test_size : float
        Proporción del test set (sobre subject_cores)
    val_size : float, optional
        Proporción del validation set. Si None, no se crea val.
    random_state : int
        Semilla aleatoria
    stratify_by : str, optional
        Columna para estratificar (default: 'subject_core')

    Returns
    -------
    train_df : pd.DataFrame
        DataFrame de entrenamiento
    test_df : pd.DataFrame
        DataFrame de test
    val_df : pd.DataFrame, optional
        DataFrame de validación
    """
    if stratify_by and stratify_by in features_df.columns:
        subject_cores = features_df[stratify_by].unique()
        n_cores = len(subject_cores)

        np.random.seed(random_state)
        shuffled_cores = np.random.permutation(subject_cores)

        # Calcular tamaños
        n_test_cores = max(1, int(n_cores * test_size))
        n_val_cores = max(1, int(n_cores * val_size)) if val_size is not None else 0

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

        # Logging
        total_epochs = len(features_df)
        train_pct = (len(train_df) / total_epochs * 100) if total_epochs > 0 else 0
        test_pct = (len(test_df) / total_epochs * 100) if total_epochs > 0 else 0

        logging.info("División del dataset:")
        logging.info(f"  Total de epochs: {total_epochs}")
        logging.info(f"  Total de subject_cores: {n_cores}")
        logging.info(
            f"Train: {len(train_df)} epochs ({train_pct:.1f}%) de "
            f"{train_df[stratify_by].nunique()} subject_cores"
        )
        logging.info(
            f"Test:  {len(test_df)} epochs ({test_pct:.1f}%) de "
            f"{test_df[stratify_by].nunique()} subject_cores"
        )
        if val_df is not None:
            val_pct = (len(val_df) / total_epochs * 100) if total_epochs > 0 else 0
            logging.info(
                f"Val:   {len(val_df)} epochs ({val_pct:.1f}%) de "
                f"{val_df[stratify_by].nunique()} subject_cores"
            )

        # Advertencias si hay pocos sujetos
        if n_cores < 10:
            logging.warning(
                f"⚠️  Solo hay {n_cores} subject_cores. "
                f"Considera usar cross-validation."
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
        # División simple (no recomendado)
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
