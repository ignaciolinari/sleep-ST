# Sleep Stage Classification

Clasificación automática de estadios de sueño (W, N1, N2, N3, REM) a partir de señales EEG, EOG y EMG usando Machine Learning y Deep Learning.

## Características

- **Pipeline completo**: descarga → preprocesamiento → extracción de features → entrenamiento
- **Modelos disponibles**: Random Forest, XGBoost, CNN1D, BiLSTM
- **Optimización bayesiana** con Optuna
- **Cross-validation** respetando sujetos (sin data leakage)
- **~130 features** espectrales, temporales y entre canales

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
python -m src.models --features-file data/processed/features.parquet --model-type random_forest
```

> **Nota:** Usá `--help` en cualquier script para ver todas las opciones disponibles.

## Documentación

| Documento | Descripción |
|-----------|-------------|
| [Getting Started](docs/GETTING_STARTED.md) | Instalación, descarga y validación de datos |
| [Data Pipeline](docs/DATA_PIPELINE.md) | Preprocesamiento, recorte y estrategias de episodios |
| [Features](docs/FEATURES.md) | Descripción detallada de las ~130 features extraídas |
| [Models](docs/MODELS.md) | Entrenamiento, splits, cross-validation y optimización |
| [Kaggle Notebooks](docs/KAGGLE_NOTEBOOKS.md) | Entrenar modelos DL en Kaggle con GPU |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | FAQ y solución de problemas comunes |

> **Validación por defecto (generalización a sujetos nuevos)**
> El pipeline usa `cv_strategy="loso"` (Leave-One-Subject-Out) por defecto, de modo que cada fold deja fuera un sujeto completo. Los splits train/test también se hacen por `subject_core`, evitando leakage entre epochs de un mismo sujeto. Solo usa `group-temporal` si necesitas evaluar separación temporal dentro del mismo sujeto; en ese caso pasa explícitamente `--cv-strategy group-temporal`.

## Estructura del proyecto

```
src/                      Código fuente
├── download.py           Descarga de Sleep-EDF desde PhysioNet
├── manifest.py           Generación de inventario de sesiones
├── preprocessing.py      Recorte alrededor del período de sueño
├── features.py           Extracción de características
├── extract_features.py   CLI para extracción batch
├── models/               Modelos ML y DL
└── crossval.py           Cross-validation por sujeto

data/                 Datos (no versionados)
├── raw/              Espejo de PhysioNet
└── processed/        Manifests, PSG recortados y features

notebooks/            Exploración, análisis y entrenamiento DL
docs/                 Documentación técnica
```

## Dataset

[Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) de PhysioNet:
- 197 grabaciones de polisomnografía (PSG)
- Canales: 2 EEG (Fpz-Cz, Pz-Oz), 1 EOG, 1 EMG
- Hipnogramas anotados manualmente (30s epochs)

## Licencia

MIT
