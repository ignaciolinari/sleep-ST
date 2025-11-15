"""Fixtures compartidas para los tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    """Crea un directorio temporal para tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_eeg_data():
    """Genera datos EEG sintéticos para tests."""
    sfreq = 100.0  # 100 Hz
    duration = 60.0  # 60 segundos
    n_samples = int(sfreq * duration)

    # Generar señal sintética con múltiples frecuencias
    t = np.linspace(0, duration, n_samples)
    signal = (
        np.sin(2 * np.pi * 1 * t)  # Delta
        + 0.5 * np.sin(2 * np.pi * 6 * t)  # Theta
        + 0.3 * np.sin(2 * np.pi * 10 * t)  # Alpha
        + 0.2 * np.sin(2 * np.pi * 20 * t)  # Beta
        + np.random.randn(n_samples) * 0.1  # Ruido
    )

    return signal, sfreq


@pytest.fixture
def sample_psg_data():
    """Genera datos PSG multi-canal sintéticos."""
    sfreq = 100.0
    duration = 300.0  # 5 minutos
    n_samples = int(sfreq * duration)
    n_channels = 4

    t = np.linspace(0, duration, n_samples)
    data = np.zeros((n_channels, n_samples))

    # Canal 0: EEG Fpz-Cz (simulado)
    data[0, :] = (
        np.sin(2 * np.pi * 1 * t)
        + 0.5 * np.sin(2 * np.pi * 6 * t)
        + np.random.randn(n_samples) * 0.1
    )

    # Canal 1: EEG Pz-Oz (simulado)
    data[1, :] = (
        np.sin(2 * np.pi * 1 * t)
        + 0.3 * np.sin(2 * np.pi * 10 * t)
        + np.random.randn(n_samples) * 0.1
    )

    # Canal 2: EOG horizontal (simulado)
    data[2, :] = 0.5 * np.sin(2 * np.pi * 2 * t) + np.random.randn(n_samples) * 0.2

    # Canal 3: EMG submental (simulado)
    data[3, :] = 0.3 * np.sin(2 * np.pi * 30 * t) + np.random.randn(n_samples) * 0.3

    return data, sfreq


@pytest.fixture
def sample_fif_file(temp_dir, sample_psg_data):
    """Crea un archivo .fif temporal con datos PSG."""
    data, sfreq = sample_psg_data

    # Crear info para MNE
    ch_names = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]
    ch_types = ["eeg", "eeg", "eog", "emg"]

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(data, info)

    # Guardar en archivo temporal (usar nombre que sigue convenciones MNE)
    fif_path = temp_dir / "test_psg_raw.fif"
    raw.save(str(fif_path), overwrite=True)

    return fif_path


@pytest.fixture
def sample_hypnogram_csv(temp_dir):
    """Crea un archivo CSV temporal con hipnograma."""
    # Crear hipnograma sintético con estadios de sueño
    stages = ["W", "N1", "N2", "N3", "N2", "REM", "N2", "N3", "N2", "REM"]
    descriptions = [
        "Sleep stage W",
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 2",
        "Sleep stage R",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 2",
        "Sleep stage R",
    ]

    epoch_length = 30.0  # 30 segundos por epoch
    hypnogram_data = []

    for i, (stage, desc) in enumerate(zip(stages, descriptions)):
        hypnogram_data.append(
            {
                "onset": i * epoch_length,
                "duration": epoch_length,
                "description": desc,
            }
        )

    df = pd.DataFrame(hypnogram_data)
    csv_path = temp_dir / "test_hypnogram.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def sample_annotations():
    """Crea anotaciones MNE sintéticas."""
    stages = ["W", "N1", "N2", "N3", "N2", "REM"]
    descriptions = [
        "Sleep stage W",
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 2",
        "Sleep stage R",
    ]

    epoch_length = 30.0
    onset = [i * epoch_length for i in range(len(stages))]
    duration = [epoch_length] * len(stages)

    annotations = mne.Annotations(
        onset=onset,
        duration=duration,
        description=descriptions,
    )

    return annotations


@pytest.fixture
def sample_features_df():
    """Crea un DataFrame de features sintético para tests."""
    n_epochs = 100
    n_subjects = 5

    # Crear features sintéticas
    features = {}

    # Features espectrales (ejemplo)
    for ch in ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]:
        for band in ["delta", "theta", "alpha", "sigma", "beta", "gamma"]:
            features[f"{ch}_rel_{band}"] = np.random.rand(n_epochs)
            features[f"{ch}_abs_{band}"] = np.random.rand(n_epochs) * 100

    # Features temporales (ejemplo)
    for ch in ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]:
        features[f"{ch}_mean"] = np.random.randn(n_epochs)
        features[f"{ch}_std"] = np.random.rand(n_epochs) * 10
        features[f"{ch}_entropy"] = np.random.rand(n_epochs) * 5

    # Features entre canales
    features["eeg_eeg_correlation"] = np.random.rand(n_epochs)
    features["eeg_eog_correlation"] = np.random.rand(n_epochs)
    features["emg_eeg_ratio"] = np.random.rand(n_epochs)

    # Metadata
    features["stage"] = np.random.choice(["W", "N1", "N2", "N3", "REM"], n_epochs)
    features["subject_id"] = [f"SC400{i}" for i in range(1, n_subjects + 1)] * (
        n_epochs // n_subjects
    )
    features["subject_core"] = [s[:5] for s in features["subject_id"]]
    features["epoch_index"] = np.arange(n_epochs)
    features["epoch_time_start"] = np.arange(n_epochs) * 30.0

    df = pd.DataFrame(features)
    return df
