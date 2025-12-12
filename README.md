# Sleep Stage Classification

Clasificación automática de estadios de sueño (W, N1, N2, N3, REM) a partir de señales EEG, EOG y EMG usando Machine Learning y Deep Learning.

## Resultados

| Modelo | Cohen's Kappa | F1 Macro | Accuracy | Uso Recomendado |
|--------|---------------|----------|----------|-----------------|
| **CNN1D** | **0.680** | 70.83% | 76.86% | Mejor rendimiento offline |
| LSTM Bi + Attention | 0.651 | 68.07% | 74.64% | Deep Learning secuencial |
| **XGBoost LOSO** | 0.641 | 70.02% | 73.08% | ML interpretable (elegido) |
| Random Forest | 0.635 | 69.50% | 72.82% | Baseline ML |
| LSTM Unidireccional | 0.530 | 58.59% | 66.17% | Inferencia real-time |


## Características

- **Pipeline completo**: descarga → preprocesamiento → extracción de features → entrenamiento
- **133 features** espectrales, temporales, spindles, ondas lentas y cross-channel
- **Modelos entrenados**: Random Forest, XGBoost (LOSO), CNN1D, LSTM (uni/bi/attention)
- **Optimización bayesiana** con Optuna (50 trials)
- **Validación LOSO** (Leave-One-Subject-Out) para generalización a nuevos sujetos

## Quickstart

```bash
# 1. Crear entorno
conda env create -f environment.yml && conda activate sleep-st

# 2. Descargar y procesar datos
python src/download.py --method wget --subset sleep-cassette --out data/raw --clean
python src/manifest.py --version 1.0.0 --subset sleep-cassette --raw-root data/raw --out data/processed/manifest.csv
python src/preprocessing.py --manifest data/processed/manifest.csv --out-root data/processed/sleep_trimmed

# 3. Extraer features y entrenar
python -m src.extract_features --manifest data/processed/manifest_trimmed.csv --output data/processed/features.parquet
python -m src.models --features-file data/processed/features.parquet --model-type xgboost
```

> **Nota:** Usá `--help` en cualquier script para ver todas las opciones disponibles.

## Documentación

| Documento | Descripción |
|-----------|-------------|
| [Getting Started](docs/GETTING_STARTED.md) | Instalación, descarga y validación de datos |
| [Data Pipeline](docs/DATA_PIPELINE.md) | Preprocesamiento, recorte y estrategias de episodios |
| [Features](docs/FEATURES.md) | Descripción detallada de las 133 features extraídas |
| [Models](docs/MODELS.md) | Entrenamiento, splits, cross-validation y optimización |
| [Kaggle Notebooks](docs/KAGGLE_NOTEBOOKS.md) | Entrenar modelos DL en Kaggle con GPU |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | FAQ y solución de problemas comunes |
| [Future Work](docs/FUTURE_WORK.md) | Direcciones futuras: Transformers, LSTM largo, híbridos |

### Reportes de Modelos

| Reporte | Descripción |
|---------|-------------|
| [Análisis Comparativo](docs/reports/COMPARATIVE_ANALYSIS.md) | Comparación final de todos los modelos |
| [XGBoost LOSO](docs/reports/XGBoost_LOSO_Analysis.md) | Análisis detallado del modelo ML elegido |
| [CNN1D](docs/reports/CNN1D_Analysis.md) | Mejor modelo Deep Learning |
| [LSTM Unidireccional](docs/reports/LSTM_Unidir_Analysis.md) | Baseline para inferencia real-time |
| [LSTM Bidireccional](docs/reports/LSTM_Bidir_Analysis.md) | Análisis de bidireccionalidad |
| [LSTM Bi+Attention](docs/reports/LSTM_Bidir_Attention_Analysis.md) | Mejor modelo LSTM |

### Notebooks de Análisis

- [03_model_results.ipynb](notebooks/03_model_results.ipynb): Comparación de métricas y visualizaciones
- [04_feature_analysis.ipynb](notebooks/04_feature_analysis.ipynb): SHAP, feature importance e interpretabilidad

## Estructura del proyecto

```
src/                      Código fuente
├── download.py           Descarga de Sleep-EDF desde PhysioNet
├── manifest.py           Generación de inventario de sesiones
├── preprocessing.py      Recorte alrededor del período de sueño
├── features.py           Extracción de características (133 features)
├── extract_features.py   CLI para extracción batch
├── models/               Modelos ML y DL
└── crossval.py           Cross-validation LOSO

data/                     Datos (no versionados)
├── raw/                  Espejo de PhysioNet
└── processed/            Manifests, PSG recortados y features

notebooks/                Exploración, análisis y entrenamiento DL
docs/                     Documentación técnica
├── reports/              Análisis detallados por modelo
artifacts/                Modelos entrenados, métricas y visualizaciones
```

## Dataset

[Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) de PhysioNet:
- **78 sujetos** (153 sesiones/noches)
- **186,499 epochs** de 30 segundos
- Canales: 2 EEG (Fpz-Cz, Pz-Oz), 1 EOG, 1 EMG
- Hipnogramas anotados manualmente

## Validación

El pipeline usa **Leave-One-Subject-Out (LOSO)** por defecto, garantizando:
- Ningún epoch del sujeto de test aparece en entrenamiento
- Evaluación de generalización a sujetos completamente nuevos
- Sin data leakage entre noches del mismo sujeto

## Licencia

MIT
