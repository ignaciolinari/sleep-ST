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
from typing import Optional, Sequence

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
    movement_policy: str = "drop",
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
                movement_policy=movement_policy,
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
    movement_policy: str = "drop",
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
            hypnogram = load_hypnogram(hyp_path, movement_policy=movement_policy)

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


def _log_class_distribution(
    split_name: str, df: pd.DataFrame, stage_col: str = "stage"
) -> None:
    if stage_col in df.columns and not df.empty:
        logging.info(
            f"Distribución de clases ({split_name}):\n"
            f"{df[stage_col].value_counts().sort_index()}"
        )


def _find_missing_classes(
    df: pd.DataFrame, required_classes: Sequence[str], stage_col: str, split_name: str
) -> list[str]:
    present = set(df[stage_col].dropna().unique())
    missing = [cls for cls in required_classes if cls not in present]
    if missing:
        logging.warning(
            f"{split_name} sin cobertura de clases {missing}. "
            "Probando un nuevo split..."
        )
    return missing


def _temporal_split_by_session(
    features_df: pd.DataFrame,
    test_size: float,
    val_size: Optional[float],
    stratify_by: str,
    session_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    if (
        session_col not in features_df.columns
        and "epoch_time_start" not in features_df.columns
    ):
        raise ValueError(
            f"Se solicitó temporal_split pero no existen columnas '{session_col}' "
            "ni 'epoch_time_start' para ordenar las sesiones."
        )

    if session_col not in features_df.columns:
        # Derivar una pseudo sesión a partir de epoch_time_start (una sola sesión)
        features_df = features_df.copy()
        features_df[session_col] = 0

    train_indices: list[int] = []
    test_indices: list[int] = []
    val_indices: list[int] = []

    for _, group in features_df.groupby(stratify_by):
        # Ordenar sesiones por sesión o tiempo
        session_order = (
            group.groupby(session_col)["epoch_time_start"].min().sort_values().index
            if "epoch_time_start" in group.columns
            else sorted(group[session_col].unique())
        )
        n_sessions = len(session_order)
        if n_sessions == 0:
            continue

        n_test_sessions = max(1, int(np.ceil(n_sessions * test_size)))
        n_val_sessions = int(np.ceil(n_sessions * val_size)) if val_size else 0

        # Ajustar si no hay suficientes sesiones para val
        if n_test_sessions + n_val_sessions >= n_sessions:
            n_val_sessions = max(0, n_sessions - n_test_sessions - 1)

        test_sessions = set(session_order[-n_test_sessions:])
        val_sessions = (
            set(session_order[-(n_test_sessions + n_val_sessions) : -n_test_sessions])
            if n_val_sessions > 0
            else set()
        )
        train_sessions = set(session_order) - test_sessions - val_sessions

        train_indices.extend(
            group[group[session_col].isin(train_sessions)].index.tolist()
        )
        test_indices.extend(
            group[group[session_col].isin(test_sessions)].index.tolist()
        )
        if val_sessions:
            val_indices.extend(
                group[group[session_col].isin(val_sessions)].index.tolist()
            )

    train_df = features_df.loc[train_indices]
    test_df = features_df.loc[test_indices]
    val_df = features_df.loc[val_indices] if val_indices else None

    return train_df, test_df, val_df


def prepare_train_test_split(
    features_df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: int = 42,
    stratify_by: Optional[str] = "subject_core",
    *,
    ensure_class_coverage: bool = True,
    required_classes: Optional[Sequence[str]] = None,
    max_attempts: int = 100,
    temporal_split: bool = False,
    session_col: str = "session_idx",
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
    ensure_class_coverage : bool
        Si True, garantiza que cada split contenga todas las clases disponibles
        en el dataset (hasta max_attempts intentos). Requiere columna 'stage'.
    required_classes : Sequence[str], optional
        Clases a verificar. Si None, usa STAGE_ORDER. Se ignoran las que no
        estén presentes en el dataset.
    max_attempts : int
        Número máximo de intentos para lograr cobertura de clases.
    temporal_split : bool
        Si True, usa un split temporal por sesión dentro de cada sujeto
        (holdout de las sesiones más recientes) en lugar de un split aleatorio
        por sujetos.
    session_col : str
        Nombre de la columna que identifica la sesión (para temporal_split).

    Returns
    -------
    train_df : pd.DataFrame
        DataFrame de entrenamiento
    test_df : pd.DataFrame
        DataFrame de test
    val_df : pd.DataFrame, optional
        DataFrame de validación
    """
    if required_classes is None:
        required_classes = STAGE_ORDER

    if ensure_class_coverage and "stage" not in features_df.columns:
        raise ValueError(
            "ensure_class_coverage=True requiere la columna 'stage' en features_df."
        )

    if stratify_by and stratify_by in features_df.columns:
        subject_cores = features_df[stratify_by].unique()
        n_cores = len(subject_cores)

        classes_in_data = (
            sorted(features_df["stage"].dropna().unique())
            if ensure_class_coverage and "stage" in features_df.columns
            else []
        )
        classes_to_check = (
            [cls for cls in required_classes if cls in classes_in_data]
            if ensure_class_coverage
            else []
        )
        if ensure_class_coverage and classes_to_check and not temporal_split:
            splits_needed = 3 if val_size else 2
            class_core_counts = features_df.groupby("stage")[stratify_by].nunique()
            impossible = [
                cls
                for cls in classes_to_check
                if class_core_counts.get(cls, 0) < splits_needed
            ]
            if impossible:
                raise ValueError(
                    "No hay suficientes subject_cores por clase para cubrir "
                    f"train/test{'/val' if val_size else ''}: {impossible}"
                )
        if ensure_class_coverage and required_classes:
            missing_overall = [
                cls for cls in required_classes if cls not in classes_in_data
            ]
            if missing_overall:
                logging.warning(
                    "Clases ausentes en el dataset y no se validarán en los splits: "
                    f"{missing_overall}"
                )

        if temporal_split:
            train_df, test_df, val_df = _temporal_split_by_session(
                features_df,
                test_size=test_size,
                val_size=val_size,
                stratify_by=stratify_by,
                session_col=session_col,
            )
            if ensure_class_coverage and classes_to_check:
                missing = {}
                missing_train = _find_missing_classes(
                    train_df, classes_to_check, "stage", "train"
                )
                missing_test = _find_missing_classes(
                    test_df, classes_to_check, "stage", "test"
                )
                missing_val = (
                    _find_missing_classes(val_df, classes_to_check, "stage", "val")
                    if val_df is not None
                    else []
                )
                if missing_train:
                    missing["train"] = missing_train
                if missing_test:
                    missing["test"] = missing_test
                if missing_val:
                    missing["val"] = missing_val
                if missing:
                    raise ValueError(
                        "Split temporal sin cobertura de clases: "
                        f"{missing}. Ajusta tamaños o datos."
                    )
        else:
            rng = np.random.default_rng(random_state)

            # Intentar hasta max_attempts encontrar un split con cobertura de clases
            for attempt in range(1, max_attempts + 1):
                shuffled_cores = rng.permutation(subject_cores)

                n_test_cores = max(1, int(n_cores * test_size))
                n_val_cores = (
                    max(1, int(n_cores * val_size)) if val_size is not None else 0
                )

                test_cores = set(shuffled_cores[:n_test_cores])
                val_cores = (
                    set(shuffled_cores[n_test_cores : n_test_cores + n_val_cores])
                    if val_size
                    else set()
                )
                train_cores = set(shuffled_cores[n_test_cores + n_val_cores :])

                train_df = features_df[features_df[stratify_by].isin(train_cores)]
                test_df = features_df[features_df[stratify_by].isin(test_cores)]
                val_df = (
                    features_df[features_df[stratify_by].isin(val_cores)]
                    if val_size
                    else None
                )

                if not ensure_class_coverage or "stage" not in features_df.columns:
                    break

                missing = {}
                if classes_to_check:
                    missing_train = _find_missing_classes(
                        train_df, classes_to_check, "stage", "train"
                    )
                    missing_test = _find_missing_classes(
                        test_df, classes_to_check, "stage", "test"
                    )
                    if val_df is not None:
                        missing_val = _find_missing_classes(
                            val_df, classes_to_check, "stage", "val"
                        )
                    else:
                        missing_val = []

                    if missing_train:
                        missing["train"] = missing_train
                    if missing_test:
                        missing["test"] = missing_test
                    if missing_val:
                        missing["val"] = missing_val

                if not missing:
                    break

                if attempt == max_attempts:
                    raise ValueError(
                        "No se pudo generar un split con cobertura de clases "
                        f"en {max_attempts} intentos. Faltantes: {missing}"
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

        # Logging de distribución de clases por split
        if "stage" in features_df.columns:
            _log_class_distribution("Train", train_df)
            if val_df is not None:
                _log_class_distribution("Val", val_df)
            _log_class_distribution("Test", test_df)
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
