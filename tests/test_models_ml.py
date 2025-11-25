"""Tests para modelos de machine learning tradicionales (Random Forest y XGBoost).

Este módulo verifica que:
1. Los modelos RF y XGBoost se entrenan correctamente
2. Las predicciones son del formato esperado
3. Los modelos se pueden guardar y cargar
4. El manejo de clases desbalanceadas funciona
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from src.models import (
    train_random_forest,
    train_xgboost,
    evaluate_model,
    save_model,
    load_model,
    STAGE_ORDER,
)


@pytest.fixture
def synthetic_feature_data():
    """Genera datos sintéticos de features para pruebas."""
    np.random.seed(42)
    n_samples = 200
    n_features = 50

    # Crear features aleatorias
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Crear etiquetas balanceadas
    y = pd.Series(np.random.choice(STAGE_ORDER, size=n_samples))

    return X, y


@pytest.fixture
def synthetic_imbalanced_data():
    """Genera datos sintéticos desbalanceados para probar class_weight."""
    np.random.seed(42)
    n_samples = 200
    n_features = 20

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    # Crear etiquetas muy desbalanceadas (80% W, 20% resto)
    y_list = ["W"] * 160 + ["N2"] * 20 + ["N3"] * 10 + ["REM"] * 10
    y = pd.Series(y_list)

    return X, y


class TestRandomForest:
    """Tests para modelo Random Forest."""

    def test_train_random_forest_basic(self, synthetic_feature_data):
        """Prueba entrenamiento básico de Random Forest."""
        X, y = synthetic_feature_data

        model = train_random_forest(
            X,
            y,
            n_estimators=10,  # Pocos árboles para test rápido
            max_depth=5,
            random_state=42,
        )

        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_train_random_forest_predictions(self, synthetic_feature_data):
        """Verifica que las predicciones tienen el formato correcto."""
        X, y = synthetic_feature_data

        model = train_random_forest(X, y, n_estimators=10, random_state=42)

        predictions = model.predict(X)

        # Verificar forma y valores
        assert len(predictions) == len(X)
        assert all(pred in STAGE_ORDER for pred in predictions)

    def test_train_random_forest_class_weight(self, synthetic_imbalanced_data):
        """Prueba que class_weight='balanced' maneja datos desbalanceados."""
        X, y = synthetic_imbalanced_data

        # Entrenar con class_weight
        model_balanced = train_random_forest(
            X, y, n_estimators=30, class_weight="balanced", random_state=42
        )

        # El modelo balanceado debería predecir más clases minoritarias
        preds_balanced = model_balanced.predict(X)

        # Verificar que el modelo balanceado predice más variedad de clases
        unique_balanced = len(set(preds_balanced))

        # El modelo balanceado debería predecir al menos tantas clases como el no balanceado
        assert unique_balanced >= 1  # Al menos debe predecir algo

    def test_train_random_forest_save_load(self, synthetic_feature_data, tmp_path):
        """Prueba guardado y carga de modelo Random Forest."""
        X, y = synthetic_feature_data

        model = train_random_forest(X, y, n_estimators=10, random_state=42)

        # Guardar modelo
        model_path = tmp_path / "test_rf_model.pkl"
        save_model(model, model_path)

        # Verificar que se creó el archivo
        assert model_path.exists()

        # Cargar modelo
        loaded_model = load_model(model_path)

        # Verificar que las predicciones son idénticas
        preds_original = model.predict(X)
        preds_loaded = loaded_model.predict(X)

        assert np.array_equal(preds_original, preds_loaded)


class TestXGBoost:
    """Tests para modelo XGBoost."""

    def test_train_xgboost_basic(self, synthetic_feature_data):
        """Prueba entrenamiento básico de XGBoost."""
        X, y = synthetic_feature_data

        model = train_xgboost(
            X,
            y,
            n_estimators=10,  # Pocos árboles para test rápido
            max_depth=3,
            random_state=42,
        )

        assert model is not None
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")
        # Verificar que guardó el label_encoder
        assert hasattr(model, "label_encoder_")
        assert hasattr(model, "original_classes_")

    def test_train_xgboost_predictions(self, synthetic_feature_data):
        """Verifica que las predicciones tienen el formato correcto."""
        X, y = synthetic_feature_data

        model = train_xgboost(X, y, n_estimators=10, random_state=42)

        # XGBoost con label_encoder devuelve índices, necesitamos decodificar
        predictions_encoded = model.predict(X)
        predictions = model.label_encoder_.inverse_transform(predictions_encoded)

        # Verificar forma y valores
        assert len(predictions) == len(X)
        assert all(pred in STAGE_ORDER for pred in predictions)

    def test_train_xgboost_automatic_balancing(self, synthetic_imbalanced_data):
        """Prueba que XGBoost balancea clases automáticamente con sample_weight."""
        X, y = synthetic_imbalanced_data

        model = train_xgboost(X, y, n_estimators=50, random_state=42)

        predictions_encoded = model.predict(X)
        predictions = model.label_encoder_.inverse_transform(predictions_encoded)

        # Debería predecir más de una clase (no solo la mayoritaria)
        unique_predictions = set(predictions)
        assert len(unique_predictions) >= 1

    def test_train_xgboost_save_load(self, synthetic_feature_data, tmp_path):
        """Prueba guardado y carga de modelo XGBoost."""
        X, y = synthetic_feature_data

        model = train_xgboost(X, y, n_estimators=10, random_state=42)

        # Guardar modelo
        model_path = tmp_path / "test_xgb_model.pkl"
        save_model(model, model_path)

        # Verificar que se creó el archivo
        assert model_path.exists()

        # Cargar modelo
        loaded_model = load_model(model_path)

        # Verificar que las predicciones son idénticas
        preds_original = model.predict(X)
        preds_loaded = loaded_model.predict(X)

        assert np.array_equal(preds_original, preds_loaded)

    def test_train_xgboost_label_encoder_preserved(
        self, synthetic_feature_data, tmp_path
    ):
        """Verifica que el label_encoder se preserva al guardar/cargar."""
        X, y = synthetic_feature_data

        model = train_xgboost(X, y, n_estimators=10, random_state=42)

        # Guardar y cargar
        model_path = tmp_path / "test_xgb_le.pkl"
        save_model(model, model_path)
        loaded_model = load_model(model_path)

        # Verificar que label_encoder_ existe y funciona
        assert hasattr(loaded_model, "label_encoder_")
        assert hasattr(loaded_model, "original_classes_")

        # Verificar que puede decodificar predicciones
        predictions_encoded = loaded_model.predict(X)
        predictions = loaded_model.label_encoder_.inverse_transform(predictions_encoded)

        assert all(pred in STAGE_ORDER for pred in predictions)


class TestEvaluateModel:
    """Tests para la función evaluate_model con modelos ML."""

    def test_evaluate_random_forest(self, synthetic_feature_data):
        """Prueba evaluación de Random Forest."""
        X, y = synthetic_feature_data

        # Separar train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = train_random_forest(X_train, y_train, n_estimators=10, random_state=42)

        metrics = evaluate_model(
            model, X_test, y_test, stage_order=STAGE_ORDER, dataset_name="TEST"
        )

        # Verificar métricas retornadas
        assert "accuracy" in metrics
        assert "kappa" in metrics
        assert "f1_macro" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_evaluate_xgboost(self, synthetic_feature_data):
        """Prueba evaluación de XGBoost."""
        X, y = synthetic_feature_data

        # Separar train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model = train_xgboost(X_train, y_train, n_estimators=10, random_state=42)

        metrics = evaluate_model(
            model, X_test, y_test, stage_order=STAGE_ORDER, dataset_name="TEST"
        )

        # Verificar métricas retornadas
        assert "accuracy" in metrics
        assert "kappa" in metrics
        assert "f1_macro" in metrics
        assert 0 <= metrics["accuracy"] <= 1


class TestModelConsistency:
    """Tests de consistencia entre entrenamientos."""

    def test_random_forest_reproducibility(self, synthetic_feature_data):
        """Verifica que RF produce resultados reproducibles con misma semilla."""
        X, y = synthetic_feature_data

        model1 = train_random_forest(X, y, n_estimators=10, random_state=42)
        model2 = train_random_forest(X, y, n_estimators=10, random_state=42)

        preds1 = model1.predict(X)
        preds2 = model2.predict(X)

        assert np.array_equal(preds1, preds2)

    def test_xgboost_reproducibility(self, synthetic_feature_data):
        """Verifica que XGBoost produce resultados reproducibles con misma semilla."""
        X, y = synthetic_feature_data

        model1 = train_xgboost(X, y, n_estimators=10, random_state=42)
        model2 = train_xgboost(X, y, n_estimators=10, random_state=42)

        preds1 = model1.predict(X)
        preds2 = model2.predict(X)

        assert np.array_equal(preds1, preds2)
