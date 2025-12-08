"""Módulo de modelos para clasificación de estadios de sueño.

Este paquete contiene implementaciones de modelos de Machine Learning y Deep Learning
para la clasificación automática de estadios de sueño a partir de señales PSG.

Modelos disponibles:
- Random Forest: Modelo de ensemble basado en árboles de decisión
- XGBoost: Gradient Boosting con regularización
- CNN1D: Red convolucional 1D para procesamiento de señales raw
- LSTM: Red recurrente bidireccional para secuencias de features

Uso básico:
    from src.models import train_random_forest, evaluate_model

    # Entrenar modelo
    model = train_random_forest(X_train, y_train)

    # Evaluar
    metrics = evaluate_model(model, X_test, y_test)

Para el pipeline completo:
    from src.models import run_training_pipeline

    metrics = run_training_pipeline(
        manifest_path="data/processed/manifest_trimmed.csv",
        model_type="random_forest",
        output_dir="models",
    )

Para optimización de hiperparámetros:
    from src.models import optimize_hyperparameters_bayesian

    results = optimize_hyperparameters_bayesian(
        model_type="random_forest",
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        groups_train=groups_train, groups_val=groups_val,
        n_trials=50,
    )
"""

# Constantes y utilidades base
from .base import (
    OPTUNA_AVAILABLE,
    STAGE_ORDER,
    TF_AVAILABLE,
    evaluate_model,
    load_metrics,
    load_model,
    print_evaluation_report,
    save_metrics,
    save_model,
)

# Preparación de datos
from .data_preparation import (
    prepare_features_dataset,
    prepare_raw_epochs_dataset,
    prepare_sequence_dataset,
    prepare_train_test_split,
)

# Optimización de hiperparámetros
from .optimization import (
    optimize_hyperparameters,
    optimize_hyperparameters_bayesian,
)

# Pipeline principal
from .pipeline import (
    build_parser,
    cross_validate_model,
    main,
    run_training_pipeline,
)

# Modelos de Machine Learning
from .random_forest import train_random_forest
from .xgboost_model import train_xgboost

# Modelos de Deep Learning (condicionales a TensorFlow)
if TF_AVAILABLE:
    from .cnn1d import (
        build_cnn1d_model as build_cnn1d_model,
    )
    from .cnn1d import (
        train_cnn1d as train_cnn1d,
    )
    from .lstm import build_lstm_model as build_lstm_model
    from .lstm import train_lstm as train_lstm
    from .optimization import optimize_hyperparameters_dl as optimize_hyperparameters_dl

__all__ = [
    # Constantes
    "STAGE_ORDER",
    "TF_AVAILABLE",
    "OPTUNA_AVAILABLE",
    # Base
    "evaluate_model",
    "print_evaluation_report",
    "save_metrics",
    "load_metrics",
    "save_model",
    "load_model",
    # Data preparation
    "prepare_features_dataset",
    "prepare_raw_epochs_dataset",
    "prepare_sequence_dataset",
    "prepare_train_test_split",
    # ML Models
    "train_random_forest",
    "train_xgboost",
    # Optimization
    "optimize_hyperparameters",
    "optimize_hyperparameters_bayesian",
    # Pipeline
    "run_training_pipeline",
    "cross_validate_model",
    "build_parser",
    "main",
]

# Agregar exports de DL solo si TensorFlow está disponible
if TF_AVAILABLE:
    __all__.extend(
        [
            "build_cnn1d_model",
            "train_cnn1d",
            "build_lstm_model",
            "train_lstm",
            "optimize_hyperparameters_dl",
        ]
    )
