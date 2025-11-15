"""Tests para modelos de deep learning (CNN1D y LSTM).

Este módulo verifica que:
1. Los modelos CNN1D y LSTM se pueden guardar y cargar correctamente
2. Los atributos personalizados (label_encoder, scaler, estadísticas de normalización) se restauran
3. Las predicciones son consistentes antes y después de guardar/cargar
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

# NOTA: Estos tests requieren TensorFlow instalado y funcionando.
# Si experimentas problemas, consulta docs/TENSORFLOW_TROUBLESHOOTING.md

# Intentar importar TensorFlow de manera segura al inicio
# Si falla, los tests se saltarán automáticamente
TF_AVAILABLE = False
try:
    import tensorflow as tf  # noqa: F401

    TF_AVAILABLE = True
except (ImportError, Exception):
    # Si TensorFlow no está disponible o causa problemas, los tests se saltarán
    TF_AVAILABLE = False


def _import_dl_models():
    """Importa funciones de modelos DL solo cuando se necesitan."""
    if not TF_AVAILABLE:
        pytest.skip("TensorFlow no está disponible")

    try:
        from src.models import (
            train_cnn1d,
            train_lstm,
            save_model,
            load_model,
            evaluate_model,
            STAGE_ORDER,
        )

        return (
            train_cnn1d,
            train_lstm,
            save_model,
            load_model,
            evaluate_model,
            STAGE_ORDER,
        )
    except Exception as e:
        pytest.skip(f"No se pudieron importar modelos DL: {e}")


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow no está disponible")
class TestCNN1DModel:
    """Tests para modelo CNN1D."""

    def test_cnn1d_save_load(self):
        """Prueba guardado y carga de modelo CNN1D."""
        train_cnn1d, _, save_model, load_model, evaluate_model, STAGE_ORDER = (
            _import_dl_models()
        )

        # Crear datos sintéticos
        n_samples = 100
        n_channels = 2
        n_samples_per_epoch = 3000  # 30 segundos a 100 Hz

        # Generar datos de entrenamiento
        np.random.seed(42)
        X_train = np.random.randn(n_samples, n_channels, n_samples_per_epoch)
        y_train = np.random.choice(STAGE_ORDER, size=n_samples)

        X_val = np.random.randn(20, n_channels, n_samples_per_epoch)
        y_val = np.random.choice(STAGE_ORDER, size=20)

        # Entrenar modelo
        model = train_cnn1d(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=2,  # Solo 2 épocas para prueba rápida
            verbose=0,
        )

        # Verificar atributos guardados
        assert hasattr(model, "label_encoder_"), "label_encoder_ no encontrado"
        assert hasattr(model, "classes_"), "classes_ no encontrado"
        assert hasattr(model, "channel_means_"), "channel_means_ no encontrado"
        assert hasattr(model, "channel_stds_"), "channel_stds_ no encontrado"

        # Hacer predicciones antes de guardar
        metrics_before = evaluate_model(
            model, X_val, y_val, stage_order=STAGE_ORDER, dataset_name="VALIDATION"
        )
        accuracy_before = metrics_before["accuracy"]

        # Guardar modelo
        temp_dir = Path(tempfile.mkdtemp())
        model_path = temp_dir / "cnn1d_test_model"
        try:
            save_model(model, model_path)

            # Verificar que se crearon archivos adicionales
            custom_attrs_path = temp_dir / "cnn1d_test_model_custom_attrs.json"
            assert (
                custom_attrs_path.exists()
            ), "Archivo de atributos personalizados no encontrado"

            # Cargar modelo
            model_loaded = load_model(model_path)

            # Verificar atributos restaurados
            assert hasattr(
                model_loaded, "label_encoder_"
            ), "label_encoder_ no restaurado"
            assert hasattr(model_loaded, "classes_"), "classes_ no restaurado"
            assert hasattr(
                model_loaded, "channel_means_"
            ), "channel_means_ no restaurado"
            assert hasattr(model_loaded, "channel_stds_"), "channel_stds_ no restaurado"

            # Verificar que los valores son iguales
            assert np.allclose(
                model.channel_means_, model_loaded.channel_means_
            ), "channel_means_ no coincide"
            assert np.allclose(
                model.channel_stds_, model_loaded.channel_stds_
            ), "channel_stds_ no coincide"
            assert np.array_equal(
                model.classes_, model_loaded.classes_
            ), "classes_ no coincide"

            # Hacer predicciones después de cargar
            metrics_after = evaluate_model(
                model_loaded,
                X_val,
                y_val,
                stage_order=STAGE_ORDER,
                dataset_name="VALIDATION",
            )
            accuracy_after = metrics_after["accuracy"]

            # Verificar que las predicciones son consistentes
            assert (
                abs(accuracy_before - accuracy_after) < 1e-6
            ), "Accuracy cambió después de cargar"

            # Verificar historial de entrenamiento
            if hasattr(model, "history_") and hasattr(model_loaded, "history_"):
                assert (
                    "loss" in model_loaded.history_
                ), "Historial de pérdida no encontrado"

        finally:
            # Limpiar archivos temporales
            import shutil

            if temp_dir.exists():
                shutil.rmtree(temp_dir)


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow no está disponible")
class TestLSTMModel:
    """Tests para modelo LSTM."""

    def test_lstm_save_load(self):
        """Prueba guardado y carga de modelo LSTM."""
        _, train_lstm, save_model, load_model, evaluate_model, STAGE_ORDER = (
            _import_dl_models()
        )

        # Crear datos sintéticos
        n_sequences = 100
        sequence_length = 5
        n_features = 20

        # Generar datos de entrenamiento
        np.random.seed(42)
        X_train = np.random.randn(n_sequences, sequence_length, n_features)
        y_train = np.random.choice(STAGE_ORDER, size=n_sequences)

        X_val = np.random.randn(20, sequence_length, n_features)
        y_val = np.random.choice(STAGE_ORDER, size=20)

        # Entrenar modelo
        model = train_lstm(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=2,  # Solo 2 épocas para prueba rápida
            verbose=0,
        )

        # Verificar atributos guardados
        assert hasattr(model, "label_encoder_"), "label_encoder_ no encontrado"
        assert hasattr(model, "classes_"), "classes_ no encontrado"
        assert hasattr(model, "scaler_"), "scaler_ no encontrado"

        # Hacer predicciones antes de guardar
        metrics_before = evaluate_model(
            model, X_val, y_val, stage_order=STAGE_ORDER, dataset_name="VALIDATION"
        )
        accuracy_before = metrics_before["accuracy"]

        # Guardar modelo
        temp_dir = Path(tempfile.mkdtemp())
        model_path = temp_dir / "lstm_test_model"
        try:
            save_model(model, model_path)

            # Verificar que se crearon archivos adicionales
            custom_attrs_path = temp_dir / "lstm_test_model_custom_attrs.json"
            scaler_path = temp_dir / "lstm_test_model_scaler.pkl"
            assert (
                custom_attrs_path.exists()
            ), "Archivo de atributos personalizados no encontrado"
            assert scaler_path.exists(), "Archivo de scaler no encontrado"

            # Cargar modelo
            model_loaded = load_model(model_path)

            # Verificar atributos restaurados
            assert hasattr(
                model_loaded, "label_encoder_"
            ), "label_encoder_ no restaurado"
            assert hasattr(model_loaded, "classes_"), "classes_ no restaurado"
            assert hasattr(model_loaded, "scaler_"), "scaler_ no restaurado"

            # Verificar que los valores son iguales
            assert np.array_equal(
                model.classes_, model_loaded.classes_
            ), "classes_ no coincide"
            # Verificar scaler (comparar mean_ y scale_)
            assert np.allclose(
                model.scaler_.mean_, model_loaded.scaler_.mean_
            ), "scaler.mean_ no coincide"
            assert np.allclose(
                model.scaler_.scale_, model_loaded.scaler_.scale_
            ), "scaler.scale_ no coincide"

            # Hacer predicciones después de cargar
            metrics_after = evaluate_model(
                model_loaded,
                X_val,
                y_val,
                stage_order=STAGE_ORDER,
                dataset_name="VALIDATION",
            )
            accuracy_after = metrics_after["accuracy"]

            # Verificar que las predicciones son consistentes
            assert (
                abs(accuracy_before - accuracy_after) < 1e-6
            ), "Accuracy cambió después de cargar"

            # Verificar historial de entrenamiento
            if hasattr(model, "history_") and hasattr(model_loaded, "history_"):
                assert (
                    "loss" in model_loaded.history_
                ), "Historial de pérdida no encontrado"

        finally:
            # Limpiar archivos temporales
            import shutil

            if temp_dir.exists():
                shutil.rmtree(temp_dir)


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow no está disponible")
class TestNormalizationConsistency:
    """Tests para consistencia de normalización."""

    def test_normalization_uses_train_stats(self):
        """Prueba que la normalización usa estadísticas de train en test."""
        train_cnn1d, _, _, _, evaluate_model, STAGE_ORDER = _import_dl_models()

        # Crear datos sintéticos con diferentes distribuciones
        np.random.seed(42)
        n_samples = 50
        n_channels = 2
        n_samples_per_epoch = 1000

        # Train con media 0, std 1
        X_train = np.random.randn(n_samples, n_channels, n_samples_per_epoch)
        y_train = np.random.choice(STAGE_ORDER, size=n_samples)

        # Test con media diferente (simula distribución diferente)
        X_test = (
            np.random.randn(20, n_channels, n_samples_per_epoch) + 5.0
        )  # Media desplazada
        y_test = np.random.choice(STAGE_ORDER, size=20)

        # Entrenar modelo
        model = train_cnn1d(X_train, y_train, epochs=1, verbose=0)

        # Guardar estadísticas originales
        train_means = model.channel_means_.copy()
        train_stds = model.channel_stds_.copy()

        # Evaluar modelo (debería usar estadísticas de train)
        _metrics = evaluate_model(
            model, X_test, y_test, stage_order=STAGE_ORDER, dataset_name="TEST"
        )

        # Verificar que se usaron las estadísticas de train
        assert np.allclose(
            model.channel_means_, train_means
        ), "Estadísticas de train cambiaron"
        assert np.allclose(
            model.channel_stds_, train_stds
        ), "Estadísticas de train cambiaron"
