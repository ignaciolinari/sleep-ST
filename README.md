# Sleep Classification Project

## Objetivo
Clasificar estadios de sueño a partir de EEG, EOG y EMG usando modelos de Machine Learning y Deep Learning.

## Estructura
- `data/`: contiene los datos (no versionados en GitHub).
- `notebooks/`: exploración y análisis preliminar.
- `src/`: código de descarga, preprocesamiento, extracción de características y modelos.

## Entorno
Entorno base:
```bash
conda env create -f environment.yml
conda activate sleep-st
```

Herramientas de desarrollo:
```bash
conda env update -n sleep-st -f environment.dev.yml
pre-commit install
```

## Descargar dataset
1. Crear cuenta en [PhysioNet](https://physionet.org).
2. Aceptar los términos de uso del dataset Sleep-EDF Expanded.
3. Descargar datos:
   ```bash
   wget -r -N -c -np https://physionet.org/files/sleep-edfx/1.0.0/
   ```
