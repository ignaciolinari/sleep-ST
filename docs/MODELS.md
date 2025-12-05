# Models

Documentación completa sobre entrenamiento de modelos, estrategias de split, cross-validation y optimización de hiperparámetros.

## Modelos disponibles

| Modelo | Tipo | Input | Descripción |
|--------|------|-------|-------------|
| Random Forest | ML | Features | Ensemble de árboles, robusto y rápido |
| XGBoost | ML | Features | Gradient boosting con regularización |
| CNN1D | DL | Señales raw | Red convolucional con bloques residuales |
| BiLSTM | DL | Secuencias de features | LSTM bidireccional con atención |

## Workflow recomendado

### Extraer features una vez, entrenar múltiples modelos

```bash
# 1. Extraer features (~20-30 minutos)
python -m src.extract_features \
  --manifest data/processed/manifest_trimmed.csv \
  --output data/processed/features.parquet \
  --format parquet

# 2. Entrenar modelos rápidamente
python -m src.models \
  --features-file data/processed/features.parquet \
  --model-type random_forest \
  --output-dir models

python -m src.models \
  --features-file data/processed/features.parquet \
  --model-type xgboost \
  --output-dir models
```

> **Nota:** Usá `python -m src.extract_features --help` y `python -m src.models --help` para ver todas las opciones disponibles.

## Entrenamiento desde CLI

### Random Forest

```bash
python -m src.models \
  --features-file data/processed/features.parquet \
  --model-type random_forest \
  --n-estimators 200 \
  --max-depth 20 \
  --output-dir models
```

### XGBoost

```bash
python -m src.models \
  --features-file data/processed/features.parquet \
  --model-type xgboost \
  --n-estimators 300 \
  --max-depth 8 \
  --learning-rate 0.1 \
  --output-dir models
```

### CNN1D (requiere TensorFlow)

```bash
python -m src.models \
  --manifest data/processed/manifest_trimmed.csv \
  --model-type cnn1d \
  --n-filters 64 \
  --epochs 50 \
  --output-dir models
```

### LSTM (requiere TensorFlow)

```bash
python -m src.models \
  --features-file data/processed/features.parquet \
  --model-type lstm \
  --lstm-units 128 \
  --sequence-length 5 \
  --epochs 50 \
  --output-dir models
```

> **Nota:** Cada modelo tiene opciones específicas. Usá `python -m src.models --help` para ver la lista completa de hiperparámetros configurables.

## Uso programático

```python
from src.models import run_training_pipeline

metrics = run_training_pipeline(
    manifest_path="data/processed/manifest_trimmed.csv",
    model_type="random_forest",
    output_dir="models",
    test_size=0.2,
    n_estimators=200,
    max_depth=None,
)
```

### Extracción manual de features

```python
from src.features import extract_features_from_session

features_df = extract_features_from_session(
    psg_path="data/processed/sleep_trimmed/psg/SC4001E_trimmed_raw.fif",
    hypnogram_path="data/processed/sleep_trimmed/hypnograms/SC4001E_trimmed_annotations.csv",
    epoch_length=30.0,
    sfreq=100.0,
)
```

## Estrategias de Split

### Por sujeto (default, recomendado)

Todos los epochs de un sujeto van al mismo conjunto (train/val/test) para evitar data leakage.

```bash
python -m src.models \
  --features-file data/processed/features.parquet \
  --model-type random_forest \
  --test-size 0.2 \
  --val-size 0.2
```

### Split temporal

Las sesiones más recientes de cada sujeto van a test/val.

```bash
python -m src.models \
  --features-file data/processed/features.parquet \
  --model-type random_forest \
  --temporal-split \
  --output-dir models_temporal
```

> **Nota**: Requiere `epoch_time_start` o `epoch_index` en las features.

## Cross-Validation

### Subject K-Fold (default)

```bash
python -m src.models \
  --features-file data/processed/features.parquet \
  --model-type random_forest \
  --cross-validate \
  --cv-strategy subject-kfold \
  --cv-folds 5
```

### Leave-One-Subject-Out (LOSO)

Cada fold deja un sujeto completo fuera. Ideal para evaluar generalización a sujetos no vistos.

```bash
python -m src.models \
  --features-file data/processed/features.parquet \
  --model-type random_forest \
  --cross-validate \
  --cv-strategy loso \
  --strict-class-coverage
```

### Group Temporal

Respeta orden temporal dentro de cada sujeto durante CV.

```bash
python -m src.models \
  --features-file data/processed/features.parquet \
  --model-type random_forest \
  --cross-validate \
  --cv-strategy group-temporal
```

## Optimización de Hiperparámetros

El pipeline detecta automáticamente si es la primera corrida y activa optimización bayesiana con Optuna.

### Forzar optimización

```bash
python -m src.models \
  --features-file data/processed/features.parquet \
  --model-type random_forest \
  --optimize \
  --n-iter-optimize 50
```

### Parámetros de optimización

| Flag | Default | Descripción |
|------|---------|-------------|
| `--optimize` | auto | Forzar optimización |
| `--n-iter-optimize` | 50 | Número de trials de Optuna |
| `--cv-folds-optimize` | 3 | Folds para CV interna |

## Cobertura de clases

El split garantiza que todas las clases (W, N1, N2, N3, REM) estén presentes en cada conjunto.

```bash
# Fallar si falta alguna clase en CV
python -m src.models \
  --features-file data/processed/features.parquet \
  --cross-validate \
  --strict-class-coverage
```

## Modelos de Deep Learning

### Consideraciones para CNN1D

- Input: señales raw `(n_epochs, n_channels, n_samples)`
- Normalización por canal guardada en el modelo
- Data augmentation con GaussianNoise
- Conexiones residuales opcionales

### Consideraciones para LSTM

- Input: secuencias de features `(n_sequences, sequence_length, n_features)`
- `sequence_length`: epochs consecutivos (default: 5 ≈ 2.5 min)
- Normalización con StandardScaler guardada en el modelo
- Arquitectura bidireccional

### Normalización

Los modelos DL guardan estadísticas de normalización del conjunto de entrenamiento:
- CNN1D: `channel_means_`, `channel_stds_`
- LSTM: `scaler_`

**Importante**: `evaluate_model` falla si no están presentes para evitar data leakage.

## Métricas de evaluación

| Métrica | Descripción |
|---------|-------------|
| Accuracy | Precisión global |
| Cohen's Kappa | Acuerdo más allá del azar |
| F1-macro | Promedio de F1 por clase |
| F1-weighted | F1 ponderado por soporte |
| Matriz de confusión | Errores por clase |

## Salida del entrenamiento

```
models/
├── random_forest_model.pkl          # Modelo entrenado
├── random_forest_feature_names.pkl  # Nombres de features
├── random_forest_metrics.json       # Métricas de evaluación
├── xgboost_model.pkl
├── cnn1d_model.keras                # Modelo Keras
├── cnn1d_model_custom_attrs.json    # Atributos (scaler, etc.)
└── lstm_model.keras
```

## Recomendaciones

| Escenario | Recomendación |
|-----------|---------------|
| Dataset pequeño (<15 sujetos) | Usar LOSO CV |
| Dataset mediano (15-50 sujetos) | Subject K-Fold con 5 folds |
| Dataset grande (>50 sujetos) | Train/val/test simple |
| Evaluar generalización | LOSO + strict-class-coverage |
| Señales raw | CNN1D |
| Contexto temporal | LSTM con sequence_length 5-10 |

## Siguiente paso

Para entrenar en GPU con Kaggle, ver [Kaggle Notebooks](KAGGLE_NOTEBOOKS.md).
