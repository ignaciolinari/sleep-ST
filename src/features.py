"""Extracción de features para clasificación de estadios de sueño.

Este módulo implementa funciones para extraer características espectrales,
temporales y de relación entre canales usando YASA y MNE, adaptadas para
un setup de pocos canales (2 EEG + EOG + EMG).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd
from scipy import signal
import yasa

# Canales por defecto del dataset Sleep-EDF
DEFAULT_CHANNELS = {
    "EEG": ["EEG Fpz-Cz", "EEG Pz-Oz"],
    "EOG": ["EOG horizontal"],
    "EMG": ["EMG submental"],
}

# Bandas de frecuencia estándar para análisis de sueño
FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "sigma": (12, 15),  # Spindles
    "beta": (13, 30),
    "gamma": (30, 45),
}

# Estadios canónicos
STAGE_CANONICAL = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}


def load_psg_data(
    psg_path: Path | str,
    channels: Optional[list[str]] = None,
    sfreq: Optional[float] = None,
) -> tuple[np.ndarray, float, list[str]]:
    """Carga datos PSG desde archivo .fif.

    Parameters
    ----------
    psg_path : Path | str
        Ruta al archivo PSG en formato .fif
    channels : list[str], optional
        Lista de canales a cargar. Si None, usa DEFAULT_CHANNELS.
    sfreq : float, optional
        Frecuencia de muestreo objetivo. Si None, mantiene la original.

    Returns
    -------
    data : np.ndarray
        Array de forma (n_channels, n_samples)
    sfreq : float
        Frecuencia de muestreo
    ch_names : list[str]
        Nombres de los canales cargados
    """
    raw = mne.io.read_raw_fif(str(psg_path), preload=True, verbose="ERROR")

    if channels is None:
        # Intentar cargar canales por defecto
        available = set(raw.ch_names)
        channels = []
        for ch_group in DEFAULT_CHANNELS.values():
            channels.extend([ch for ch in ch_group if ch in available])

    if not channels:
        raise ValueError(f"No se encontraron canales válidos en {psg_path}")

    raw.pick_channels(channels)
    current_sfreq = raw.info["sfreq"]

    if sfreq is not None and sfreq != current_sfreq:
        raw.resample(sfreq)
        current_sfreq = sfreq

    data = raw.get_data()
    ch_names = raw.ch_names

    return data, current_sfreq, ch_names


def load_hypnogram(
    hypnogram_path: Path | str,
) -> pd.DataFrame:
    """Carga hipnograma desde archivo CSV.

    Parameters
    ----------
    hypnogram_path : Path | str
        Ruta al archivo CSV con anotaciones

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: onset, duration, description, stage
    """
    df = pd.read_csv(hypnogram_path)
    required_cols = {"onset", "duration", "description"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"El hipnograma debe contener las columnas: {required_cols}")

    df = df.copy()
    df["stage"] = df["description"].map(STAGE_CANONICAL)
    df = df[df["stage"].notna()].copy()

    return df


def create_epochs(
    data: np.ndarray,
    sfreq: float,
    epoch_length: float = 30.0,
    overlap: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Segmenta datos en epochs de duración fija.

    Parameters
    ----------
    data : np.ndarray
        Array de forma (n_channels, n_samples)
    sfreq : float
        Frecuencia de muestreo
    epoch_length : float
        Duración de cada epoch en segundos (default: 30s)
    overlap : float
        Solapamiento entre epochs en segundos (default: 0)

    Returns
    -------
    epochs : np.ndarray
        Array de forma (n_epochs, n_channels, n_samples_per_epoch)
    times : np.ndarray
        Array con tiempos de inicio de cada epoch en segundos
    """
    n_channels, n_samples = data.shape
    samples_per_epoch = int(epoch_length * sfreq)
    step = int((epoch_length - overlap) * sfreq)

    epochs = []
    times = []

    for start_idx in range(0, n_samples - samples_per_epoch + 1, step):
        end_idx = start_idx + samples_per_epoch
        epoch = data[:, start_idx:end_idx]
        epochs.append(epoch)
        times.append(start_idx / sfreq)

    if not epochs:
        return np.array([]).reshape(0, n_channels, 0), np.array([])

    return np.array(epochs), np.array(times)


def assign_stages_to_epochs(
    epochs_times: np.ndarray,
    hypnogram: pd.DataFrame,
    epoch_length: float = 30.0,
) -> np.ndarray:
    """Asigna estadios de sueño a cada epoch basado en el hipnograma.

    Parameters
    ----------
    epochs_times : np.ndarray
        Tiempos de inicio de cada epoch en segundos
    hypnogram : pd.DataFrame
        DataFrame con anotaciones (debe tener columnas onset, duration, stage)
    epoch_length : float
        Duración de cada epoch en segundos

    Returns
    -------
    np.ndarray
        Array con el estadio de cada epoch (W, N1, N2, N3, REM)
    """
    stages: list[str | None] = []
    for t_start in epochs_times:
        t_end = t_start + epoch_length
        # Encontrar anotación que cubre el punto medio del epoch
        t_mid = (t_start + t_end) / 2

        matching = hypnogram[
            (hypnogram["onset"] <= t_mid)
            & (hypnogram["onset"] + hypnogram["duration"] > t_mid)
        ]

        if matching.empty:
            stages.append(None)
        else:
            # Si hay múltiples, tomar la primera
            stage = matching.iloc[0]["stage"]
            stages.append(stage)

    return np.array(stages)


def extract_spectral_features(
    data: np.ndarray,
    sfreq: float,
    ch_name: str,
) -> dict[str, float]:
    """Extrae features espectrales usando YASA.

    Parameters
    ----------
    data : np.ndarray
        Señal 1D de forma (n_samples,)
    sfreq : float
        Frecuencia de muestreo
    ch_name : str
        Nombre del canal (para logging)

    Returns
    -------
    dict[str, float]
        Diccionario con features espectrales
    """
    if len(data.shape) > 1:
        raise ValueError("data debe ser 1D")

    features = {}

    # Calcular PSD usando YASA
    try:
        # YASA espera bands como lista de tuplas (fmin, fmax, nombre)
        bands_yasa = [
            (fmin, fmax, band_name.capitalize())
            for band_name, (fmin, fmax) in FREQ_BANDS.items()
        ]

        bandpower = yasa.bandpower(
            data,
            sf=sfreq,
            bands=bands_yasa,
            relative=True,  # Potencia relativa
        )

        # Agregar potencia absoluta también
        bandpower_abs = yasa.bandpower(
            data,
            sf=sfreq,
            bands=bands_yasa,
            relative=False,
        )

        # YASA retorna un DataFrame, extraer valores por nombre de banda
        # Los nombres en YASA son capitalizados (Delta, Theta, etc.)
        band_mapping = {
            "delta": "Delta",
            "theta": "Theta",
            "alpha": "Alpha",
            "sigma": "Sigma",
            "beta": "Beta",
            "gamma": "Gamma",
        }

        # Features de potencia relativa por banda
        # YASA retorna un DataFrame con índice 'Chan' y columnas con nombres de bandas
        for band_key, band_yasa_name in band_mapping.items():
            if (
                isinstance(bandpower, pd.DataFrame)
                and band_yasa_name in bandpower.columns
            ):
                # Tomar el primer canal (índice 'CHAN000' o similar)
                rel_val = float(bandpower[band_yasa_name].iloc[0])
                abs_val = float(bandpower_abs[band_yasa_name].iloc[0])
            else:
                rel_val = 0.0
                abs_val = 0.0

            features[f"{ch_name}_rel_{band_key}"] = rel_val
            features[f"{ch_name}_abs_{band_key}"] = abs_val

        # Ratios importantes para sleep staging
        delta = features.get(f"{ch_name}_rel_delta", 0.0)
        theta = features.get(f"{ch_name}_rel_theta", 0.0)
        alpha = features.get(f"{ch_name}_rel_alpha", 0.0)
        sigma = features.get(f"{ch_name}_rel_sigma", 0.0)

        if delta > 0:
            features[f"{ch_name}_theta_delta_ratio"] = theta / delta
            features[f"{ch_name}_alpha_delta_ratio"] = alpha / delta
        if sigma > 0:
            features[f"{ch_name}_sigma_delta_ratio"] = sigma / delta

        # Frecuencia dominante (usando scipy directamente)
        try:
            freqs_full, psd_full = signal.welch(
                data, sfreq, nperseg=min(1024, len(data))
            )
            # Filtrar banda de interés (0.5-45 Hz)
            freq_mask = (freqs_full >= 0.5) & (freqs_full <= 45)
            if freq_mask.any():
                freqs_filtered = freqs_full[freq_mask]
                psd_filtered = psd_full[freq_mask]
                if psd_filtered.sum() > 0:
                    dominant_freq_idx = np.argmax(psd_filtered)
                    features[f"{ch_name}_dominant_freq"] = float(
                        freqs_filtered[dominant_freq_idx]
                    )
        except Exception:
            pass  # Si falla, simplemente no agregar esta feature

    except Exception as e:
        logging.warning(f"Error extrayendo features espectrales para {ch_name}: {e}")
        # Retornar features vacías en caso de error
        for band in FREQ_BANDS.keys():
            features[f"{ch_name}_rel_{band}"] = 0.0
            features[f"{ch_name}_abs_{band}"] = 0.0

    return features


def extract_temporal_features(
    data: np.ndarray,
    ch_name: str,
) -> dict[str, float]:
    """Extrae features temporales de la señal.

    Parameters
    ----------
    data : np.ndarray
        Señal 1D de forma (n_samples,)
    ch_name : str
        Nombre del canal

    Returns
    -------
    dict[str, float]
        Diccionario con features temporales
    """
    if len(data.shape) > 1:
        raise ValueError("data debe ser 1D")

    features = {}

    # Estadísticas básicas
    features[f"{ch_name}_mean"] = float(np.mean(data))
    features[f"{ch_name}_std"] = float(np.std(data))
    features[f"{ch_name}_var"] = float(np.var(data))
    features[f"{ch_name}_min"] = float(np.min(data))
    features[f"{ch_name}_max"] = float(np.max(data))
    features[f"{ch_name}_range"] = float(np.ptp(data))  # peak-to-peak

    # Hjorth parameters
    try:
        hjorth = yasa.hjorth_params(data)
        features[f"{ch_name}_hjorth_activity"] = float(hjorth[0])
        features[f"{ch_name}_hjorth_mobility"] = float(hjorth[1])
        features[f"{ch_name}_hjorth_complexity"] = float(hjorth[2])
    except Exception:
        features[f"{ch_name}_hjorth_activity"] = 0.0
        features[f"{ch_name}_hjorth_mobility"] = 0.0
        features[f"{ch_name}_hjorth_complexity"] = 0.0

    # Entropía de Shannon (aproximada)
    try:
        # Normalizar y discretizar para calcular entropía
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-10)
        hist, _ = np.histogram(data_norm, bins=20)
        hist = hist[hist > 0]
        if len(hist) > 0:
            prob = hist / hist.sum()
            entropy = -np.sum(prob * np.log2(prob))
            features[f"{ch_name}_entropy"] = float(entropy)
        else:
            features[f"{ch_name}_entropy"] = 0.0
    except Exception:
        features[f"{ch_name}_entropy"] = 0.0

    # Zero crossing rate
    zcr = np.sum(np.diff(np.signbit(data))) / len(data)
    features[f"{ch_name}_zcr"] = float(zcr)

    return features


def extract_cross_channel_features(
    eeg1: np.ndarray,
    eeg2: np.ndarray,
    eog: np.ndarray,
    emg: np.ndarray,
    sfreq: float,
) -> dict[str, float]:
    """Extrae features de relación entre canales.

    Parameters
    ----------
    eeg1 : np.ndarray
        Primer canal EEG (1D)
    eeg2 : np.ndarray
        Segundo canal EEG (1D)
    eog : np.ndarray
        Canal EOG (1D)
    emg : np.ndarray
        Canal EMG (1D)
    sfreq : float
        Frecuencia de muestreo

    Returns
    -------
    dict[str, float]
        Diccionario con features de relación entre canales
    """
    features = {}

    # Correlación entre canales EEG
    if len(eeg1) == len(eeg2) and len(eeg1) > 0:
        corr_eeg = np.corrcoef(eeg1, eeg2)[0, 1]
        features["eeg_eeg_correlation"] = float(corr_eeg)

    # Correlación EEG-EOG (importante para REM)
    if len(eeg1) == len(eog) and len(eeg1) > 0:
        corr_eeg_eog = np.corrcoef(eeg1, eog)[0, 1]
        features["eeg_eog_correlation"] = float(corr_eeg_eog)

    # Ratio EMG/EEG (importante para distinguir REM de vigilia)
    if len(emg) > 0 and len(eeg1) > 0:
        emg_power = np.mean(emg**2)
        eeg_power = np.mean(eeg1**2)
        if eeg_power > 0:
            features["emg_eeg_ratio"] = float(emg_power / eeg_power)
        else:
            features["emg_eeg_ratio"] = 0.0

    # Ratio EOG/EMG (útil para REM: EOG alto, EMG bajo)
    if len(eog) > 0 and len(emg) > 0:
        eog_power = np.mean(eog**2)
        emg_power = np.mean(emg**2)
        if emg_power > 0:
            features["eog_emg_ratio"] = float(eog_power / emg_power)
        else:
            features["eog_emg_ratio"] = 0.0

    # Coherencia en banda theta entre EEG y EOG (simplificada)
    try:
        if len(eeg1) == len(eog) and len(eeg1) > 100:
            # Calcular coherencia aproximada en banda theta
            f, Pxx_eeg = signal.welch(eeg1, sfreq, nperseg=min(1024, len(eeg1)))
            f, Pxx_eog = signal.welch(eog, sfreq, nperseg=min(1024, len(eog)))
            f, Pxy = signal.csd(eeg1, eog, sfreq, nperseg=min(1024, len(eeg1)))

            # Banda theta
            theta_mask = (f >= 4) & (f <= 8)
            if theta_mask.any():
                coh_theta = np.abs(Pxy[theta_mask]) ** 2 / (
                    Pxx_eeg[theta_mask] * Pxx_eog[theta_mask] + 1e-10
                )
                features["eeg_eog_theta_coherence"] = float(np.mean(coh_theta))
            else:
                features["eeg_eog_theta_coherence"] = 0.0
        else:
            features["eeg_eog_theta_coherence"] = 0.0
    except Exception:
        features["eeg_eog_theta_coherence"] = 0.0

    return features


def extract_features_for_epoch(
    epoch: np.ndarray,
    ch_names: list[str],
    sfreq: float,
) -> dict[str, float]:
    """Extrae todas las features para un epoch.

    Parameters
    ----------
    epoch : np.ndarray
        Array de forma (n_channels, n_samples)
    ch_names : list[str]
        Nombres de los canales
    sfreq : float
        Frecuencia de muestreo

    Returns
    -------
    dict[str, float]
        Diccionario con todas las features del epoch
    """
    features = {}

    # Identificar canales por tipo
    eeg_channels = []
    eog_channels = []
    emg_channels = []

    for idx, ch_name in enumerate(ch_names):
        ch_lower = ch_name.lower()
        if "eeg" in ch_lower:
            eeg_channels.append((idx, ch_name))
        elif "eog" in ch_lower:
            eog_channels.append((idx, ch_name))
        elif "emg" in ch_lower:
            emg_channels.append((idx, ch_name))

    # Features por canal
    for idx, ch_name in enumerate(ch_names):
        ch_data = epoch[idx, :]

        # Features espectrales
        spectral = extract_spectral_features(ch_data, sfreq, ch_name)
        features.update(spectral)

        # Features temporales
        temporal = extract_temporal_features(ch_data, ch_name)
        features.update(temporal)

    # Features de relación entre canales
    if eeg_channels and eog_channels and emg_channels:
        eeg1_idx, eeg1_name = eeg_channels[0]
        eog_idx, eog_name = eog_channels[0]
        emg_idx, emg_name = emg_channels[0]

        eeg1_data = epoch[eeg1_idx, :]
        eog_data = epoch[eog_idx, :]
        emg_data = epoch[emg_idx, :]

        # Si hay segundo EEG, usarlo también
        eeg2_data = None
        if len(eeg_channels) > 1:
            eeg2_idx, _ = eeg_channels[1]
            eeg2_data = epoch[eeg2_idx, :]
        else:
            eeg2_data = eeg1_data  # Usar el mismo si solo hay uno

        cross_channel = extract_cross_channel_features(
            eeg1_data, eeg2_data, eog_data, emg_data, sfreq
        )
        features.update(cross_channel)

    return features


def extract_features_from_session(
    psg_path: Path | str,
    hypnogram_path: Path | str,
    epoch_length: float = 30.0,
    sfreq: Optional[float] = None,
    channels: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Extrae features para todos los epochs de una sesión.

    Parameters
    ----------
    psg_path : Path | str
        Ruta al archivo PSG (.fif)
    hypnogram_path : Path | str
        Ruta al archivo de hipnograma (.csv)
    epoch_length : float
        Duración de cada epoch en segundos (default: 30s)
    sfreq : float, optional
        Frecuencia de muestreo objetivo
    channels : list[str], optional
        Canales a usar

    Returns
    -------
    pd.DataFrame
        DataFrame con una fila por epoch y columnas de features + 'stage'
    """
    # Cargar datos
    data, actual_sfreq, ch_names = load_psg_data(psg_path, channels, sfreq)
    hypnogram = load_hypnogram(hypnogram_path)

    # Crear epochs
    epochs, epochs_times = create_epochs(data, actual_sfreq, epoch_length=epoch_length)

    if len(epochs) == 0:
        logging.warning(f"No se pudieron crear epochs para {psg_path}")
        return pd.DataFrame()

    # Asignar estadios
    stages = assign_stages_to_epochs(epochs_times, hypnogram, epoch_length)

    # Extraer features para cada epoch
    features_list = []
    for epoch_idx, (epoch, stage, epoch_time) in enumerate(
        zip(epochs, stages, epochs_times)
    ):
        if stage is None:
            continue  # Saltar epochs sin etiqueta

        epoch_features = extract_features_for_epoch(epoch, ch_names, actual_sfreq)
        epoch_features["stage"] = stage
        epoch_features["epoch_time_start"] = epoch_time  # Tiempo de inicio del epoch
        epoch_features["epoch_index"] = (
            epoch_idx  # Índice del epoch dentro de la sesión
        )
        features_list.append(epoch_features)

    if not features_list:
        logging.warning(f"No se extrajeron features válidas para {psg_path}")
        return pd.DataFrame()

    df = pd.DataFrame(features_list)
    # Ordenar por tiempo para asegurar orden temporal
    df = df.sort_values("epoch_time_start").reset_index(drop=True)
    return df
