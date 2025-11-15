"""Tests para el módulo features.py."""

from __future__ import annotations


import mne
import numpy as np
import pandas as pd
import pytest

from src.features import (
    assign_stages_to_epochs,
    create_epochs,
    extract_cross_channel_features,
    extract_features_for_epoch,
    extract_features_from_session,
    extract_spectral_features,
    extract_temporal_features,
    load_hypnogram,
    load_psg_data,
    FREQ_BANDS,
)


class TestLoadPSGData:
    """Tests para load_psg_data."""

    def test_load_psg_data_basic(self, sample_fif_file):
        """Test carga básica de datos PSG."""
        data, sfreq, ch_names = load_psg_data(sample_fif_file)

        assert data.shape[0] == 4  # 4 canales
        assert sfreq == 100.0
        assert len(ch_names) == 4
        assert "EEG Fpz-Cz" in ch_names

    def test_load_psg_data_with_channels(self, sample_fif_file):
        """Test carga con selección de canales específicos."""
        channels = ["EEG Fpz-Cz", "EOG horizontal"]
        data, sfreq, ch_names = load_psg_data(sample_fif_file, channels=channels)

        assert data.shape[0] == 2
        assert set(ch_names) == set(channels)

    def test_load_psg_data_with_resample(self, sample_fif_file):
        """Test resampleo de datos."""
        target_sfreq = 50.0
        data, sfreq, _ = load_psg_data(sample_fif_file, sfreq=target_sfreq)

        assert sfreq == target_sfreq
        # Verificar que el número de muestras se redujo aproximadamente a la mitad
        assert data.shape[1] < 30000  # Original tiene ~30000 muestras a 100Hz

    def test_load_psg_data_invalid_file(self, temp_dir):
        """Test manejo de archivo inválido."""
        invalid_path = temp_dir / "nonexistent.fif"

        with pytest.raises(
            Exception
        ):  # Puede ser FileNotFoundError u otro error de MNE
            load_psg_data(invalid_path)

    def test_load_psg_data_no_channels(self, temp_dir, sample_psg_data):
        """Test cuando no hay canales disponibles."""
        data, sfreq = sample_psg_data
        # Crear archivo con canales que no coinciden con DEFAULT_CHANNELS
        ch_names = ["CH1", "CH2", "CH3", "CH4"]
        ch_types = ["eeg"] * 4

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data, info)

        fif_path = temp_dir / "test_no_channels_raw.fif"
        raw.save(str(fif_path), overwrite=True)

        with pytest.raises(ValueError, match="No se encontraron canales válidos"):
            load_psg_data(fif_path)


class TestLoadHypnogram:
    """Tests para load_hypnogram."""

    def test_load_hypnogram_basic(self, sample_hypnogram_csv):
        """Test carga básica de hipnograma."""
        df = load_hypnogram(sample_hypnogram_csv)

        assert "onset" in df.columns
        assert "duration" in df.columns
        assert "description" in df.columns
        assert "stage" in df.columns
        assert len(df) > 0

    def test_load_hypnogram_stage_mapping(self, temp_dir):
        """Test mapeo correcto de estadios."""
        # Crear CSV con diferentes estadios
        data = {
            "onset": [0, 30, 60, 90],
            "duration": [30, 30, 30, 30],
            "description": [
                "Sleep stage W",
                "Sleep stage 1",
                "Sleep stage 2",
                "Sleep stage R",
            ],
        }
        df = pd.DataFrame(data)
        csv_path = temp_dir / "test_stages.csv"
        df.to_csv(csv_path, index=False)

        result = load_hypnogram(csv_path)

        assert result["stage"].iloc[0] == "W"
        assert result["stage"].iloc[1] == "N1"
        assert result["stage"].iloc[2] == "N2"
        assert result["stage"].iloc[3] == "REM"

    def test_load_hypnogram_missing_columns(self, temp_dir):
        """Test manejo de columnas faltantes."""
        df = pd.DataFrame({"onset": [0], "duration": [30]})  # Falta 'description'
        csv_path = temp_dir / "test_invalid.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="debe contener las columnas"):
            load_hypnogram(csv_path)

    def test_load_hypnogram_filters_invalid_stages(self, temp_dir):
        """Test que filtra estadios inválidos."""
        data = {
            "onset": [0, 30, 60],
            "duration": [30, 30, 30],
            "description": [
                "Sleep stage W",
                "Invalid stage",
                "Sleep stage 2",
            ],
        }
        df = pd.DataFrame(data)
        csv_path = temp_dir / "test_filter.csv"
        df.to_csv(csv_path, index=False)

        result = load_hypnogram(csv_path)

        # Debe filtrar "Invalid stage" que no está en STAGE_CANONICAL
        assert len(result) == 2
        assert "Invalid stage" not in result["description"].values


class TestCreateEpochs:
    """Tests para create_epochs."""

    def test_create_epochs_basic(self, sample_psg_data):
        """Test creación básica de epochs."""
        data, sfreq = sample_psg_data
        epochs, times = create_epochs(data, sfreq, epoch_length=30.0)

        assert len(epochs) > 0
        assert len(times) == len(epochs)
        assert epochs.shape[1] == data.shape[0]  # Mismo número de canales
        assert epochs.shape[2] == int(
            30.0 * sfreq
        )  # 30 segundos a 100 Hz = 3000 muestras

    def test_create_epochs_custom_length(self, sample_psg_data):
        """Test creación de epochs con longitud personalizada."""
        data, sfreq = sample_psg_data
        epoch_length = 60.0  # 60 segundos
        epochs, times = create_epochs(data, sfreq, epoch_length=epoch_length)

        assert epochs.shape[2] == int(epoch_length * sfreq)

    def test_create_epochs_overlap(self, sample_psg_data):
        """Test creación de epochs con solapamiento."""
        data, sfreq = sample_psg_data
        epochs_no_overlap, _ = create_epochs(
            data, sfreq, epoch_length=30.0, overlap=0.0
        )
        epochs_overlap, _ = create_epochs(data, sfreq, epoch_length=30.0, overlap=10.0)

        # Con solapamiento debe haber más epochs
        assert len(epochs_overlap) > len(epochs_no_overlap)

    def test_create_epochs_empty(self):
        """Test creación de epochs con datos vacíos."""
        data = np.zeros((4, 100))  # Solo 100 muestras, menos de un epoch de 30s a 100Hz
        sfreq = 100.0
        epochs, times = create_epochs(data, sfreq, epoch_length=30.0)

        assert len(epochs) == 0
        assert len(times) == 0


class TestAssignStagesToEpochs:
    """Tests para assign_stages_to_epochs."""

    def test_assign_stages_basic(self, sample_hypnogram_csv):
        """Test asignación básica de estadios."""
        hypnogram = load_hypnogram(sample_hypnogram_csv)
        epochs_times = np.array([0, 30, 60, 90, 120])

        stages = assign_stages_to_epochs(epochs_times, hypnogram, epoch_length=30.0)

        assert len(stages) == len(epochs_times)
        assert stages[0] == "W"  # Primer epoch debe ser W según el hipnograma

    def test_assign_stages_no_match(self, temp_dir):
        """Test cuando no hay coincidencia de estadios."""
        # Crear hipnograma que no cubre los epochs
        data = {
            "onset": [1000, 1030],  # Muy tarde
            "duration": [30, 30],
            "description": ["Sleep stage W", "Sleep stage 1"],
        }
        df = pd.DataFrame(data)
        csv_path = temp_dir / "test_no_match.csv"
        df.to_csv(csv_path, index=False)

        hypnogram = load_hypnogram(csv_path)
        epochs_times = np.array([0, 30, 60])  # Epochs tempranos

        stages = assign_stages_to_epochs(epochs_times, hypnogram, epoch_length=30.0)

        # Debe retornar None para epochs sin coincidencia
        assert all(s is None for s in stages)


class TestExtractSpectralFeatures:
    """Tests para extract_spectral_features."""

    def test_extract_spectral_features_basic(self, sample_eeg_data):
        """Test extracción básica de features espectrales."""
        signal, sfreq = sample_eeg_data
        features = extract_spectral_features(signal, sfreq, "EEG Fpz-Cz")

        # Verificar que se extrajeron features para todas las bandas
        for band in FREQ_BANDS.keys():
            assert f"EEG Fpz-Cz_rel_{band}" in features
            assert f"EEG Fpz-Cz_abs_{band}" in features

        # Verificar ratios
        assert "EEG Fpz-Cz_theta_delta_ratio" in features
        assert "EEG Fpz-Cz_alpha_delta_ratio" in features

        # Verificar frecuencia dominante
        assert "EEG Fpz-Cz_dominant_freq" in features

    def test_extract_spectral_features_2d_error(self):
        """Test que lanza error con datos 2D."""
        data_2d = np.random.randn(10, 100)
        sfreq = 100.0

        with pytest.raises(ValueError, match="data debe ser 1D"):
            extract_spectral_features(data_2d, sfreq, "test")


class TestExtractTemporalFeatures:
    """Tests para extract_temporal_features."""

    def test_extract_temporal_features_basic(self, sample_eeg_data):
        """Test extracción básica de features temporales."""
        signal, _ = sample_eeg_data
        features = extract_temporal_features(signal, "EEG Fpz-Cz")

        # Verificar estadísticas básicas
        assert "EEG Fpz-Cz_mean" in features
        assert "EEG Fpz-Cz_std" in features
        assert "EEG Fpz-Cz_var" in features
        assert "EEG Fpz-Cz_min" in features
        assert "EEG Fpz-Cz_max" in features
        assert "EEG Fpz-Cz_range" in features

        # Verificar parámetros de Hjorth
        assert "EEG Fpz-Cz_hjorth_activity" in features
        assert "EEG Fpz-Cz_hjorth_mobility" in features
        assert "EEG Fpz-Cz_hjorth_complexity" in features

        # Verificar entropía y ZCR
        assert "EEG Fpz-Cz_entropy" in features
        assert "EEG Fpz-Cz_zcr" in features

    def test_extract_temporal_features_2d_error(self):
        """Test que lanza error con datos 2D."""
        data_2d = np.random.randn(10, 100)

        with pytest.raises(ValueError, match="data debe ser 1D"):
            extract_temporal_features(data_2d, "test")


class TestExtractCrossChannelFeatures:
    """Tests para extract_cross_channel_features."""

    def test_extract_cross_channel_features_basic(self, sample_eeg_data):
        """Test extracción básica de features entre canales."""
        signal, sfreq = sample_eeg_data
        eeg1 = signal
        eeg2 = signal * 0.9  # Similar pero diferente
        eog = signal * 0.5
        emg = signal * 0.3

        features = extract_cross_channel_features(eeg1, eeg2, eog, emg, sfreq)

        assert "eeg_eeg_correlation" in features
        assert "eeg_eog_correlation" in features
        assert "emg_eeg_ratio" in features
        assert "eog_emg_ratio" in features
        assert "eeg_eog_theta_coherence" in features

    def test_extract_cross_channel_features_different_lengths(self):
        """Test manejo de canales con longitudes diferentes."""
        eeg1 = np.random.randn(100)
        eeg2 = np.random.randn(50)  # Diferente longitud
        eog = np.random.randn(100)
        emg = np.random.randn(100)

        features = extract_cross_channel_features(eeg1, eeg2, eog, emg, 100.0)

        # Debe manejar graciosamente las diferencias de longitud
        # (algunas features pueden no calcularse)
        assert isinstance(features, dict)


class TestExtractFeaturesForEpoch:
    """Tests para extract_features_for_epoch."""

    def test_extract_features_for_epoch_basic(self, sample_psg_data):
        """Test extracción de features para un epoch."""
        data, sfreq = sample_psg_data
        epoch_length = 30.0
        n_samples = int(epoch_length * sfreq)

        # Tomar un epoch
        epoch = data[:, :n_samples]
        ch_names = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]

        features = extract_features_for_epoch(epoch, ch_names, sfreq)

        # Verificar que se extrajeron features para todos los canales
        assert len(features) > 0
        # Verificar que hay features espectrales y temporales
        assert any("rel_delta" in k for k in features.keys())
        assert any("mean" in k for k in features.keys())


class TestExtractFeaturesFromSession:
    """Tests para extract_features_from_session."""

    def test_extract_features_from_session_basic(
        self, sample_fif_file, sample_hypnogram_csv
    ):
        """Test extracción completa de features desde una sesión."""
        df = extract_features_from_session(
            sample_fif_file,
            sample_hypnogram_csv,
            epoch_length=30.0,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "stage" in df.columns
        assert "epoch_time_start" in df.columns
        assert "epoch_index" in df.columns

        # Verificar que hay features espectrales
        assert any("rel_delta" in col for col in df.columns)

    def test_extract_features_from_session_no_epochs(self, temp_dir, sample_psg_data):
        """Test cuando no se pueden crear epochs."""
        # Crear archivo con muy pocos datos
        data, sfreq = sample_psg_data
        data_short = data[:, :100]  # Solo 100 muestras

        ch_names = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"]
        ch_types = ["eeg", "eeg", "eog", "emg"]

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
        raw = mne.io.RawArray(data_short, info)

        fif_path = temp_dir / "test_short_raw.fif"
        raw.save(str(fif_path), overwrite=True)

        # Crear hipnograma vacío
        hyp_path = temp_dir / "test_empty_hyp.csv"
        pd.DataFrame(columns=["onset", "duration", "description"]).to_csv(
            hyp_path, index=False
        )

        df = extract_features_from_session(fif_path, hyp_path, epoch_length=30.0)

        # Debe retornar DataFrame vacío
        assert df.empty
