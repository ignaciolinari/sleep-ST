"""Script de prueba para verificar guardado y carga de modelos de deep learning.

Este script verifica que:
1. Los modelos CNN1D y LSTM se pueden guardar y cargar correctamente
2. Los atributos personalizados (label_encoder, scaler, estadísticas de normalización) se restauran
3. Las predicciones son consistentes antes y después de guardar/cargar
"""

from __future__ import annotations

import logging
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.models import (
    train_cnn1d,
    train_lstm,
    save_model,
    load_model,
    evaluate_model,
    STAGE_ORDER,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def test_cnn1d_save_load():
    """Prueba guardado y carga de modelo CNN1D."""
    print("\n" + "=" * 60)
    print("PRUEBA: Guardado y carga de CNN1D")
    print("=" * 60)

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
    print("\n1. Entrenando modelo CNN1D...")
    model = train_cnn1d(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=2,  # Solo 2 épocas para prueba rápida
        verbose=0,
    )

    # Verificar atributos guardados
    print("\n2. Verificando atributos del modelo entrenado...")
    assert hasattr(model, "label_encoder_"), "label_encoder_ no encontrado"
    assert hasattr(model, "classes_"), "classes_ no encontrado"
    assert hasattr(model, "channel_means_"), "channel_means_ no encontrado"
    assert hasattr(model, "channel_stds_"), "channel_stds_ no encontrado"
    print("  ✓ Todos los atributos presentes")

    # Hacer predicciones antes de guardar
    print("\n3. Haciendo predicciones antes de guardar...")
    metrics_before = evaluate_model(
        model, X_val, y_val, stage_order=STAGE_ORDER, dataset_name="VALIDATION"
    )
    accuracy_before = metrics_before["accuracy"]
    print(f"  Accuracy antes de guardar: {accuracy_before:.4f}")

    # Guardar modelo
    print("\n4. Guardando modelo...")
    temp_dir = Path(tempfile.mkdtemp())
    model_path = temp_dir / "cnn1d_test_model"
    try:
        save_model(model, model_path)
        print(f"  ✓ Modelo guardado en {model_path}")

        # Verificar que se crearon archivos adicionales
        custom_attrs_path = temp_dir / "cnn1d_test_model_custom_attrs.json"
        assert (
            custom_attrs_path.exists()
        ), "Archivo de atributos personalizados no encontrado"
        print("  ✓ Archivo de atributos personalizados creado")

        # Cargar modelo
        print("\n5. Cargando modelo...")
        model_loaded = load_model(model_path)
        print(f"  ✓ Modelo cargado desde {model_path}")

        # Verificar atributos restaurados
        print("\n6. Verificando atributos restaurados...")
        assert hasattr(model_loaded, "label_encoder_"), "label_encoder_ no restaurado"
        assert hasattr(model_loaded, "classes_"), "classes_ no restaurado"
        assert hasattr(model_loaded, "channel_means_"), "channel_means_ no restaurado"
        assert hasattr(model_loaded, "channel_stds_"), "channel_stds_ no restaurado"
        print("  ✓ Todos los atributos restaurados")

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
        print("  ✓ Valores de atributos coinciden")

        # Hacer predicciones después de cargar
        print("\n7. Haciendo predicciones después de cargar...")
        metrics_after = evaluate_model(
            model_loaded,
            X_val,
            y_val,
            stage_order=STAGE_ORDER,
            dataset_name="VALIDATION",
        )
        accuracy_after = metrics_after["accuracy"]
        print(f"  Accuracy después de cargar: {accuracy_after:.4f}")

        # Verificar que las predicciones son consistentes
        assert (
            abs(accuracy_before - accuracy_after) < 1e-6
        ), "Accuracy cambió después de cargar"
        print("  ✓ Predicciones consistentes")

        # Verificar historial de entrenamiento
        if hasattr(model, "history_") and hasattr(model_loaded, "history_"):
            print("\n8. Verificando historial de entrenamiento...")
            assert "loss" in model_loaded.history_, "Historial de pérdida no encontrado"
            print("  ✓ Historial de entrenamiento restaurado")

        print("\n" + "=" * 60)
        print("✓ PRUEBA CNN1D EXITOSA")
        print("=" * 60)
        return True

    finally:
        # Limpiar archivos temporales
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def test_lstm_save_load():
    """Prueba guardado y carga de modelo LSTM."""
    print("\n" + "=" * 60)
    print("PRUEBA: Guardado y carga de LSTM")
    print("=" * 60)

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
    print("\n1. Entrenando modelo LSTM...")
    model = train_lstm(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=2,  # Solo 2 épocas para prueba rápida
        verbose=0,
    )

    # Verificar atributos guardados
    print("\n2. Verificando atributos del modelo entrenado...")
    assert hasattr(model, "label_encoder_"), "label_encoder_ no encontrado"
    assert hasattr(model, "classes_"), "classes_ no encontrado"
    assert hasattr(model, "scaler_"), "scaler_ no encontrado"
    print("  ✓ Todos los atributos presentes")

    # Hacer predicciones antes de guardar
    print("\n3. Haciendo predicciones antes de guardar...")
    metrics_before = evaluate_model(
        model, X_val, y_val, stage_order=STAGE_ORDER, dataset_name="VALIDATION"
    )
    accuracy_before = metrics_before["accuracy"]
    print(f"  Accuracy antes de guardar: {accuracy_before:.4f}")

    # Guardar modelo
    print("\n4. Guardando modelo...")
    temp_dir = Path(tempfile.mkdtemp())
    model_path = temp_dir / "lstm_test_model"
    try:
        save_model(model, model_path)
        print(f"  ✓ Modelo guardado en {model_path}")

        # Verificar que se crearon archivos adicionales
        custom_attrs_path = temp_dir / "lstm_test_model_custom_attrs.json"
        scaler_path = temp_dir / "lstm_test_model_scaler.pkl"
        assert (
            custom_attrs_path.exists()
        ), "Archivo de atributos personalizados no encontrado"
        assert scaler_path.exists(), "Archivo de scaler no encontrado"
        print("  ✓ Archivos de atributos personalizados creados")

        # Cargar modelo
        print("\n5. Cargando modelo...")
        model_loaded = load_model(model_path)
        print(f"  ✓ Modelo cargado desde {model_path}")

        # Verificar atributos restaurados
        print("\n6. Verificando atributos restaurados...")
        assert hasattr(model_loaded, "label_encoder_"), "label_encoder_ no restaurado"
        assert hasattr(model_loaded, "classes_"), "classes_ no restaurado"
        assert hasattr(model_loaded, "scaler_"), "scaler_ no restaurado"
        print("  ✓ Todos los atributos restaurados")

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
        print("  ✓ Valores de atributos coinciden")

        # Hacer predicciones después de cargar
        print("\n7. Haciendo predicciones después de cargar...")
        metrics_after = evaluate_model(
            model_loaded,
            X_val,
            y_val,
            stage_order=STAGE_ORDER,
            dataset_name="VALIDATION",
        )
        accuracy_after = metrics_after["accuracy"]
        print(f"  Accuracy después de cargar: {accuracy_after:.4f}")

        # Verificar que las predicciones son consistentes
        assert (
            abs(accuracy_before - accuracy_after) < 1e-6
        ), "Accuracy cambió después de cargar"
        print("  ✓ Predicciones consistentes")

        # Verificar historial de entrenamiento
        if hasattr(model, "history_") and hasattr(model_loaded, "history_"):
            print("\n8. Verificando historial de entrenamiento...")
            assert "loss" in model_loaded.history_, "Historial de pérdida no encontrado"
            print("  ✓ Historial de entrenamiento restaurado")

        print("\n" + "=" * 60)
        print("✓ PRUEBA LSTM EXITOSA")
        print("=" * 60)
        return True

    finally:
        # Limpiar archivos temporales
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def test_normalization_consistency():
    """Prueba que la normalización usa estadísticas de train en test."""
    print("\n" + "=" * 60)
    print("PRUEBA: Consistencia de normalización")
    print("=" * 60)

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
    print("\n1. Entrenando modelo con datos de train...")
    model = train_cnn1d(X_train, y_train, epochs=1, verbose=0)

    # Guardar estadísticas originales
    train_means = model.channel_means_.copy()
    train_stds = model.channel_stds_.copy()

    print("\n2. Estadísticas de train guardadas:")
    print(f"   Channel 0: mean={train_means[0]:.4f}, std={train_stds[0]:.4f}")
    print(f"   Channel 1: mean={train_means[1]:.4f}, std={train_stds[1]:.4f}")

    # Calcular estadísticas de test (no deberían usarse)
    test_means = [np.mean(X_test[:, i, :]) for i in range(n_channels)]
    test_stds = [np.std(X_test[:, i, :]) for i in range(n_channels)]

    print("\n3. Estadísticas de test (NO deberían usarse):")
    print(f"   Channel 0: mean={test_means[0]:.4f}, std={test_stds[0]:.4f}")
    print(f"   Channel 1: mean={test_means[1]:.4f}, std={test_stds[1]:.4f}")

    # Evaluar modelo (debería usar estadísticas de train)
    print("\n4. Evaluando modelo en test...")
    _metrics = evaluate_model(
        model, X_test, y_test, stage_order=STAGE_ORDER, dataset_name="TEST"
    )

    # Verificar que se usaron las estadísticas de train
    print("\n5. Verificando que se usaron estadísticas de train...")
    assert np.allclose(
        model.channel_means_, train_means
    ), "Estadísticas de train cambiaron"
    assert np.allclose(
        model.channel_stds_, train_stds
    ), "Estadísticas de train cambiaron"
    print("  ✓ Se usaron estadísticas de train (no de test)")

    print("\n" + "=" * 60)
    print("✓ PRUEBA DE NORMALIZACIÓN EXITOSA")
    print("=" * 60)
    return True


def main():
    """Ejecuta todas las pruebas."""
    print("\n" + "=" * 60)
    print("SUITE DE PRUEBAS: Modelos de Deep Learning")
    print("=" * 60)

    try:
        # Verificar que TensorFlow está disponible
        from src.models import TF_AVAILABLE

        if not TF_AVAILABLE:
            print("\n❌ ERROR: TensorFlow no está disponible")
            print("Instala TensorFlow para ejecutar estas pruebas:")
            print("  conda install tensorflow")
            return 1

        # Ejecutar pruebas
        test_cnn1d_save_load()
        test_lstm_save_load()
        test_normalization_consistency()

        print("\n" + "=" * 60)
        print("✓ TODAS LAS PRUEBAS EXITOSAS")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n❌ PRUEBA FALLIDA: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
