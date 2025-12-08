"""Extracción de features para clasificación de estadios de sueño.

Este módulo implementa funciones para extraer características espectrales,
temporales y de relación entre canales usando YASA y MNE, adaptadas para
un setup de pocos canales (2 EEG + EOG + EMG).
"""

from __future__ import annotations

import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import yasa
from scipy import signal
from scipy.stats import entropy as scipy_entropy

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

# Parámetros de pre-filtrado por defecto
DEFAULT_FILTER_BAND = (0.3, 45.0)
DEFAULT_NOTCH_FREQS: tuple[float, ...] = (50.0, 60.0)

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
    channels: list[str] | None = None,
    sfreq: float | None = None,
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
        Array de forma (n_channels, n_samples) en microvolts (µV)
    sfreq : float
        Frecuencia de muestreo
    ch_names : list[str]
        Nombres de los canales cargados

    Notes
    -----
    Los datos se escalan de Voltios (unidad de MNE) a Microvolts (µV),
    ya que YASA y otras librerías de análisis de sueño esperan µV.
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

    # MNE almacena datos en Voltios, pero YASA espera Microvolts (µV)
    # Escalar V → µV (multiplicar por 1e6)
    data = raw.get_data() * 1e6
    ch_names = raw.ch_names

    return data, current_sfreq, ch_names


def load_hypnogram(
    hypnogram_path: Path | str,
    movement_policy: str = "drop",
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

    movement_policy = movement_policy.lower()
    if movement_policy not in {"drop", "map_to_w", "keep_unknown"}:
        raise ValueError("movement_policy debe ser 'drop', 'map_to_w' o 'keep_unknown'")

    df = df.copy()

    def _map_stage(desc: str | float) -> str | None:
        if isinstance(desc, float) and pd.isna(desc):
            return None
        if desc in {"Movement time", "Sleep stage ?"}:
            if movement_policy == "drop":
                return None
            if movement_policy == "map_to_w":
                return "W"
            return "UNK"
        return STAGE_CANONICAL.get(str(desc))

    df["stage"] = df["description"].map(_map_stage)
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

    Notes
    -----
    Valida que `overlap < epoch_length` y que el paso `(epoch_length - overlap) * sfreq`
    sea > 0; en caso contrario se lanza un ValueError con mensaje explícito.
    """
    if epoch_length <= 0:
        raise ValueError("epoch_length debe ser > 0")
    if overlap < 0:
        raise ValueError("overlap debe ser >= 0")
    if overlap >= epoch_length:
        raise ValueError("overlap debe ser menor que epoch_length")

    n_channels, n_samples = data.shape
    samples_per_epoch = int(epoch_length * sfreq)
    step = int((epoch_length - overlap) * sfreq)

    if step <= 0:
        raise ValueError(
            "El paso de epoch debe ser > 0; reduce overlap o aumenta epoch_length"
        )

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


def _preprocess_channel(
    data: np.ndarray,
    sfreq: float,
    bandpass: tuple[float, float] = DEFAULT_FILTER_BAND,
    notch_freqs: tuple[float, ...] | list[float] | None = DEFAULT_NOTCH_FREQS,
    detrend: bool = True,
) -> np.ndarray:
    """Aplicar detrend, band-pass y notch básicos a un canal 1D."""
    cleaned = np.asarray(data, dtype=float)

    if detrend:
        try:
            cleaned = signal.detrend(cleaned, type="linear")
        except Exception as exc:
            logging.debug("No se pudo aplicar detrend: %s", exc)

    l_freq, h_freq = bandpass
    try:
        cleaned = mne.filter.filter_data(
            cleaned,
            sfreq=sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            method="fir",
            fir_design="firwin",
            verbose="ERROR",
        )
    except Exception as exc:
        logging.debug(
            "No se pudo aplicar band-pass %.2f-%.2f Hz: %s", l_freq, h_freq, exc
        )

    if notch_freqs:
        try:
            nyq = sfreq / 2.0 if sfreq else 0.0
            valid_notch = [f for f in notch_freqs if f and f < nyq]
            if valid_notch:
                cleaned = mne.filter.notch_filter(
                    cleaned,
                    Fs=sfreq,
                    freqs=valid_notch,
                    method="fir",
                    fir_design="firwin",
                    verbose="ERROR",
                )
        except Exception as exc:
            logging.debug("No se pudo aplicar notch %s Hz: %s", notch_freqs, exc)

    return cleaned


def extract_spectral_features(
    data: np.ndarray,
    sfreq: float,
    ch_name: str,
    *,
    psd_method: str = "welch",
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

    method = (psd_method or "welch").lower()
    if method not in {"welch", "multitaper"}:
        method = "welch"

    # Calcular PSD usando YASA
    try:
        # YASA espera bands como lista de tuplas (fmin, fmax, nombre)
        bands_yasa = [
            (fmin, fmax, band_name.capitalize())
            for band_name, (fmin, fmax) in FREQ_BANDS.items()
        ]

        try:
            bandpower = yasa.bandpower(
                data,
                sf=sfreq,
                bands=bands_yasa,
                relative=True,  # Potencia relativa
                method=method,
            )
            bandpower_abs = yasa.bandpower(
                data,
                sf=sfreq,
                bands=bands_yasa,
                relative=False,
                method=method,
            )
        except TypeError:
            # Versiones antiguas de YASA no aceptan method -> fallback a default (Welch)
            bandpower = yasa.bandpower(
                data,
                sf=sfreq,
                bands=bands_yasa,
                relative=True,  # Potencia relativa
            )
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

        # Usar epsilon para evitar división por cero
        eps = 1e-10
        features[f"{ch_name}_theta_delta_ratio"] = theta / (delta + eps)
        features[f"{ch_name}_alpha_delta_ratio"] = alpha / (delta + eps)
        features[f"{ch_name}_sigma_delta_ratio"] = sigma / (delta + eps)

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
        except Exception as e:
            logging.debug(f"Error calculando frecuencia dominante para {ch_name}: {e}")

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
    sfreq: float,
    epoch_length: float | None = None,
) -> dict[str, float]:
    """Extrae features temporales de la señal.

    Parameters
    ----------
    data : np.ndarray
        Señal 1D de forma (n_samples,)
    ch_name : str
        Nombre del canal
    sfreq : float
        Frecuencia de muestreo
    epoch_length : float, optional
        Duración del epoch en segundos (usado para entropía espectral si sfreq no está disponible)

    Returns
    -------
    dict[str, float]
        Diccionario con features temporales

    Notes
    -----
    Si `sfreq` no se provee o es 0, la entropía espectral deriva `fs = len(data) / epoch_length`
    para el Welch, manteniendo la escala correcta aunque cambie `epoch_length`.
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
    except Exception as e:
        logging.debug(f"Error calculando parámetros Hjorth para {ch_name}: {e}")
        features[f"{ch_name}_hjorth_activity"] = 0.0
        features[f"{ch_name}_hjorth_mobility"] = 0.0
        features[f"{ch_name}_hjorth_complexity"] = 0.0

    # Entropía de Shannon mejorada con más bins y scipy
    try:
        # Normalizar señal
        data_range = data.max() - data.min()
        if data_range > 1e-10:
            data_norm = (data - data.min()) / data_range
            # Usar 100 bins para mejor resolución (en lugar de 20)
            hist, _ = np.histogram(data_norm, bins=100, density=False)
            # Filtrar bins vacíos y calcular probabilidades
            hist = hist[hist > 0]
            if len(hist) > 1:
                prob = hist / hist.sum()
                # scipy_entropy usa log natural por defecto, base=2 para bits
                features[f"{ch_name}_entropy"] = float(scipy_entropy(prob, base=2))
            else:
                features[f"{ch_name}_entropy"] = 0.0
        else:
            features[f"{ch_name}_entropy"] = 0.0
    except Exception as e:
        logging.debug(f"Error calculando entropía de Shannon para {ch_name}: {e}")
        features[f"{ch_name}_entropy"] = 0.0

    # Entropía espectral (informativa para estados de sueño)
    try:
        effective_epoch_length = (
            epoch_length if epoch_length and epoch_length > 0 else None
        )
        if sfreq and sfreq > 0:
            sfreq_entropy = sfreq
        elif effective_epoch_length:
            sfreq_entropy = len(data) / effective_epoch_length
        else:
            sfreq_entropy = 1.0  # Evitar división por cero en casos extremos

        freqs, psd = signal.welch(data, fs=sfreq_entropy, nperseg=min(256, len(data)))
        # Filtrar a banda de interés (0.5-45 Hz)
        freq_mask = (freqs >= 0.5) & (freqs <= 45)
        if freq_mask.any():
            psd_filtered = psd[freq_mask]
            if psd_filtered.sum() > 1e-10:
                # Normalizar PSD como distribución de probabilidad
                psd_norm = psd_filtered / psd_filtered.sum()
                features[f"{ch_name}_spectral_entropy"] = float(
                    scipy_entropy(psd_norm, base=2)
                )
            else:
                features[f"{ch_name}_spectral_entropy"] = 0.0
        else:
            features[f"{ch_name}_spectral_entropy"] = 0.0
    except Exception as e:
        logging.debug(f"Error calculando entropía espectral para {ch_name}: {e}")
        features[f"{ch_name}_spectral_entropy"] = 0.0

    # Zero crossing rate
    zcr = np.sum(np.diff(np.signbit(data))) / len(data)
    features[f"{ch_name}_zcr"] = float(zcr)

    return features


def extract_cross_channel_features(
    eeg1: np.ndarray,
    eeg2: np.ndarray | None,
    eog: np.ndarray,
    emg: np.ndarray,
    sfreq: float,
    *,
    require_eeg2: bool = False,
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

    # Correlación entre canales EEG (solo si hay dos EEG)
    if eeg2 is not None and len(eeg1) == len(eeg2) and len(eeg1) > 0:
        corr_eeg = np.corrcoef(eeg1, eeg2)[0, 1]
        features["eeg_eeg_correlation"] = float(corr_eeg)
    elif require_eeg2:
        # Si se exige EEG2 y no está disponible, omitir la feature
        return features
    else:
        # Sin segundo EEG: conservar la columna pero sin introducir NaNs para evitar
        # romper modelos clásicos (RF/XGB). Interpretamos “sin información” como 0.0.
        features["eeg_eeg_correlation"] = 0.0

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

    # Coherencia en banda theta entre EEG y EOG usando scipy.signal.coherence
    try:
        if len(eeg1) == len(eog) and len(eeg1) > 100:
            # Usar scipy.signal.coherence (método correcto)
            nperseg = min(
                256, len(eeg1) // 4
            )  # Segmentos más pequeños para epochs de 30s
            f, coh = signal.coherence(eeg1, eog, fs=sfreq, nperseg=nperseg)

            # Coherencia en banda theta (4-8 Hz)
            theta_mask = (f >= 4) & (f <= 8)
            if theta_mask.any():
                features["eeg_eog_theta_coherence"] = float(np.mean(coh[theta_mask]))
            else:
                features["eeg_eog_theta_coherence"] = 0.0

            # Coherencia en banda delta (0.5-4 Hz) - útil para N3
            delta_mask = (f >= 0.5) & (f <= 4)
            if delta_mask.any():
                features["eeg_eog_delta_coherence"] = float(np.mean(coh[delta_mask]))
            else:
                features["eeg_eog_delta_coherence"] = 0.0

            # Coherencia en banda sigma (12-15 Hz) - útil para spindles en N2
            sigma_mask = (f >= 12) & (f <= 15)
            if sigma_mask.any():
                features["eeg_eog_sigma_coherence"] = float(np.mean(coh[sigma_mask]))
            else:
                features["eeg_eog_sigma_coherence"] = 0.0
        else:
            features["eeg_eog_theta_coherence"] = 0.0
            features["eeg_eog_delta_coherence"] = 0.0
            features["eeg_eog_sigma_coherence"] = 0.0
    except Exception as e:
        logging.debug(f"Error calculando coherencia EEG-EOG: {e}")
        features["eeg_eog_theta_coherence"] = 0.0
        features["eeg_eog_delta_coherence"] = 0.0
        features["eeg_eog_sigma_coherence"] = 0.0

    return features


def extract_spindle_features(
    data: np.ndarray,
    sfreq: float,
    ch_name: str,
) -> dict[str, float]:
    """Extrae features relacionadas con spindles de sueño usando YASA.

    Los spindles son característicos del estadio N2 y se definen según los
    criterios AASM como oscilaciones en la banda sigma (11-16 Hz, típicamente
    12-15 Hz) con duración de 0.5-2 segundos.

    Parameters
    ----------
    data : np.ndarray
        Señal 1D de forma (n_samples,)
    sfreq : float
        Frecuencia de muestreo
    ch_name : str
        Nombre del canal

    Returns
    -------
    dict[str, float]
        Diccionario con features de spindles:
        - spindle_count: número de spindles detectados
        - spindle_density: spindles por minuto
        - spindle_mean_duration: duración media (segundos)
        - spindle_mean_amplitude: amplitud media (uV)

    Notes
    -----
    La detección usa YASA (Vallat & Walker, 2021) con parámetros basados en:

    - **Banda de frecuencia**: 12-15 Hz (banda sigma según AASM, 2007)
    - **Duración**: 0.5-2 segundos (Warby et al., 2014; Purcell et al., 2017)
    - **Umbrales de detección**:
      - rel_pow=0.2: potencia relativa mínima en sigma vs. banda ancha
      - corr=0.65: correlación mínima con oscilación sigma pura
      - rms=1.5: umbral RMS en desviaciones estándar

    Estos umbrales son los valores por defecto de YASA, validados contra
    scoring manual en el dataset MASS (Vallat & Walker, 2021).

    References
    ----------
    - AASM (2007). The AASM Manual for the Scoring of Sleep.
    - Warby et al. (2014). Sleep spindle measurements. Sleep, 37(9), 1469-1479.
    - Purcell et al. (2017). Characterizing sleep spindles. Sleep, 40(1).
    - Vallat & Walker (2021). An open-source, high-performance tool for
      automated sleep staging. eLife, 10:e70092.
    """
    features = {
        f"{ch_name}_spindle_count": 0.0,
        f"{ch_name}_spindle_density": 0.0,
        f"{ch_name}_spindle_mean_duration": 0.0,
        f"{ch_name}_spindle_mean_amplitude": 0.0,
    }

    if len(data) < sfreq * 2:  # Mínimo 2 segundos de datos
        return features

    try:
        # YASA spindles_detect aplica internamente filtrado en freq_sp,
        # por lo que no es necesario pre-filtrar los datos.
        # Parámetros según criterios AASM y validación YASA (ver docstring).
        sp = yasa.spindles_detect(
            data,
            sf=sfreq,
            freq_sp=(12, 15),  # Banda sigma estándar AASM
            duration=(0.5, 2),  # Duración según Warby et al. (2014)
            thresh={"rel_pow": 0.2, "corr": 0.65, "rms": 1.5},  # Defaults YASA
            remove_outliers=True,
            verbose=False,
        )

        if sp is not None:
            summary = sp.summary()
            if summary is not None and len(summary) > 0:
                n_spindles = len(summary)
                epoch_duration_min = len(data) / sfreq / 60

                features[f"{ch_name}_spindle_count"] = float(n_spindles)
                features[f"{ch_name}_spindle_density"] = (
                    float(n_spindles / epoch_duration_min)
                    if epoch_duration_min > 0
                    else 0.0
                )
                features[f"{ch_name}_spindle_mean_duration"] = float(
                    summary["Duration"].mean()
                )
                features[f"{ch_name}_spindle_mean_amplitude"] = float(
                    summary["Amplitude"].mean()
                )
    except Exception as e:
        logging.debug(f"Error detectando spindles en {ch_name}: {e}")

    return features


def extract_slow_wave_features(
    data: np.ndarray,
    sfreq: float,
    ch_name: str,
) -> dict[str, float]:
    """Extrae features de ondas lentas (slow waves) usando análisis de potencia delta.

    Las ondas lentas son características del estadio N3 (0.5-4 Hz, alta amplitud).

    Parameters
    ----------
    data : np.ndarray
        Señal 1D de forma (n_samples,)
    sfreq : float
        Frecuencia de muestreo
    ch_name : str
        Nombre del canal

    Returns
    -------
    dict[str, float]
        Diccionario con features de ondas lentas:
        - slow_wave_power: potencia absoluta en banda delta
        - slow_wave_ratio: ratio de potencia delta vs total
        - slow_wave_peak_amplitude: amplitud pico-a-pico en banda delta
    """
    features = {
        f"{ch_name}_slow_wave_power": 0.0,
        f"{ch_name}_slow_wave_ratio": 0.0,
        f"{ch_name}_slow_wave_peak_amplitude": 0.0,
    }

    if len(data) < sfreq * 2:
        return features

    try:
        try:
            data = signal.detrend(data, type="linear")
        except Exception as e:
            logging.debug(f"Error en detrend para slow wave features {ch_name}: {e}")

        # Filtrar en banda delta (0.5-4 Hz)
        nyq = sfreq / 2
        low = 0.5 / nyq
        high = 4.0 / nyq

        if high < 1.0:  # Verificar que la frecuencia de corte es válida
            b, a = signal.butter(4, [low, high], btype="band")
            delta_filtered = signal.filtfilt(b, a, data)

            # Potencia en banda delta
            features[f"{ch_name}_slow_wave_power"] = float(np.mean(delta_filtered**2))

            # Amplitud pico-a-pico
            features[f"{ch_name}_slow_wave_peak_amplitude"] = float(
                np.ptp(delta_filtered)
            )

            # Ratio delta vs total
            total_power = np.mean(data**2)
            if total_power > 1e-10:
                features[f"{ch_name}_slow_wave_ratio"] = float(
                    np.mean(delta_filtered**2) / total_power
                )
    except Exception as e:
        logging.debug(f"Error extrayendo slow wave features en {ch_name}: {e}")

    return features


def extract_features_for_epoch(
    epoch: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    epoch_length: float = 30.0,
    apply_prefilter: bool = True,
    bandpass: tuple[float, float] = DEFAULT_FILTER_BAND,
    notch_freqs: tuple[float, ...] | list[float] | None = DEFAULT_NOTCH_FREQS,
    psd_method: str = "welch",
    skip_cross_if_single_eeg: bool = True,
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
    epoch_length : float
        Duración del epoch en segundos
    apply_prefilter : bool
        Si True, aplica detrend + band-pass + notch antes de extraer features
    bandpass : tuple[float, float]
        Banda del filtro pasa banda previa a la extracción
    notch_freqs : tuple[float, ...] | list[float] | None
        Frecuencias de notch a aplicar antes de las features

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
    cleaned_epoch: list[np.ndarray] = []

    for idx, ch_name in enumerate(ch_names):
        ch_data = epoch[idx, :]
        preprocessed = (
            _preprocess_channel(
                ch_data, sfreq, bandpass=bandpass, notch_freqs=notch_freqs
            )
            if apply_prefilter
            else ch_data
        )
        cleaned_epoch.append(preprocessed)

        # Features espectrales
        spectral = extract_spectral_features(
            preprocessed, sfreq, ch_name, psd_method=psd_method
        )
        features.update(spectral)

        # Features temporales
        temporal = extract_temporal_features(
            preprocessed, ch_name, sfreq, epoch_length=epoch_length
        )
        features.update(temporal)

    # Features de spindles y ondas lentas (solo para canales EEG)
    for idx, ch_name in eeg_channels:
        ch_data = cleaned_epoch[idx]

        # Detección de spindles (importantes para N2)
        spindle_feats = extract_spindle_features(ch_data, sfreq, ch_name)
        features.update(spindle_feats)

        # Features de ondas lentas (importantes para N3)
        slow_wave_feats = extract_slow_wave_features(ch_data, sfreq, ch_name)
        features.update(slow_wave_feats)

    # Features de relación entre canales
    if eeg_channels and eog_channels and emg_channels:
        eeg1_idx, eeg1_name = eeg_channels[0]
        eog_idx, eog_name = eog_channels[0]
        emg_idx, emg_name = emg_channels[0]

        eeg1_data = cleaned_epoch[eeg1_idx]
        eog_data = cleaned_epoch[eog_idx]
        emg_data = cleaned_epoch[emg_idx]

        # Si hay segundo EEG, usarlo también
        has_second_eeg = len(eeg_channels) > 1
        eeg2_data = None
        if has_second_eeg:
            eeg2_idx, _ = eeg_channels[1]
            eeg2_data = cleaned_epoch[eeg2_idx]

        if not has_second_eeg and skip_cross_if_single_eeg:
            cross_channel = {}
        else:
            cross_channel = extract_cross_channel_features(
                eeg1_data,
                eeg2_data,
                eog_data,
                emg_data,
                sfreq,
                require_eeg2=skip_cross_if_single_eeg,
            )
        features.update(cross_channel)

    return features


def extract_features_from_session(
    psg_path: Path | str,
    hypnogram_path: Path | str,
    epoch_length: float = 30.0,
    sfreq: float | None = None,
    channels: list[str] | None = None,
    movement_policy: str = "drop",
    overlap: float = 0.0,
    apply_prefilter: bool = True,
    bandpass: tuple[float, float] = DEFAULT_FILTER_BAND,
    notch_freqs: tuple[float, ...] | list[float] | None = DEFAULT_NOTCH_FREQS,
    psd_method: str = "welch",
    skip_cross_if_single_eeg: bool = True,
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
    overlap : float
        Solapamiento entre epochs en segundos (default: 0)
    apply_prefilter : bool
        Si True aplica detrend + band-pass + notch antes de extraer features
    bandpass : tuple[float, float]
        Banda del filtro pasa-banda previa a extracción (default: 0.3-45 Hz)
    notch_freqs : tuple | list | None
        Frecuencias de notch a aplicar (default: 50, 60 Hz). None desactiva notch.
    psd_method : {"welch", "multitaper"}
        Método para el cálculo de PSD en features espectrales (default: welch)
    skip_cross_if_single_eeg : bool
        Si True, omite las features EEG-EEG/coherencia cuando sólo hay un EEG

    Returns
    -------
    pd.DataFrame
        DataFrame con una fila por epoch y columnas de features + 'stage'

    Notes
    -----
    Los epochs sin etiqueta se omiten; antes de descartarlos se loggea cuántos y
    el porcentaje de pérdida sobre la sesión.
    """
    # Cargar datos
    data, actual_sfreq, ch_names = load_psg_data(psg_path, channels, sfreq)
    hypnogram = load_hypnogram(hypnogram_path, movement_policy=movement_policy)

    # Crear epochs
    epochs, epochs_times = create_epochs(
        data, actual_sfreq, epoch_length=epoch_length, overlap=overlap
    )

    if len(epochs) == 0:
        logging.warning(f"No se pudieron crear epochs para {psg_path}")
        return pd.DataFrame()

    # Asignar estadios
    stages = assign_stages_to_epochs(epochs_times, hypnogram, epoch_length)

    total_epochs = len(stages)
    missing_labels = int(np.sum(pd.isna(stages)))
    if missing_labels:
        pct_missing = (missing_labels / total_epochs * 100) if total_epochs else 0.0
        logging.info(
            "Sesión %s: descartados %s epochs sin etiqueta (%.1f%%)",
            psg_path,
            missing_labels,
            pct_missing,
        )

    # Extraer features para cada epoch
    features_list = []
    for epoch_idx, (epoch, stage, epoch_time) in enumerate(
        zip(epochs, stages, epochs_times)
    ):
        if stage is None:
            continue  # Saltar epochs sin etiqueta

        epoch_features = extract_features_for_epoch(
            epoch,
            ch_names,
            actual_sfreq,
            epoch_length=epoch_length,
            apply_prefilter=apply_prefilter,
            bandpass=bandpass,
            notch_freqs=notch_freqs,
            psd_method=psd_method,
            skip_cross_if_single_eeg=skip_cross_if_single_eeg,
        )
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
