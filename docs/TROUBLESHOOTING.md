# Troubleshooting

Guía de solución de problemas comunes y FAQ.

## Problemas de instalación

### Error: `ModuleNotFoundError: No module named 'yasa'`

YASA se instala via pip dentro del entorno conda:

```bash
conda activate sleep-st
pip install yasa
```

### Error: TensorFlow no disponible

El proyecto funciona sin TensorFlow para modelos ML (Random Forest, XGBoost). Para modelos DL:

```bash
pip install tensorflow==2.16.2
```

### TensorFlow en macOS Apple Silicon

Los modelos DL están configurados para usar CPU en macOS para evitar errores con Metal:

```python
# Esto se hace automáticamente en src/models/base.py
os.environ["TF_METAL_PLUGIN_LIBRARY_PATH"] = ""
tf.config.set_visible_devices([], "GPU")
```

Si experimentás crashes (bus error, segfault), verificá que TensorFlow use CPU.

## Problemas de descarga

### Error: PhysioNet requiere autenticación

```bash
# Opción 1: Variables de entorno
export PHYSIONET_USERNAME=tu_usuario
export PHYSIONET_PASSWORD=tu_password

# Opción 2: El script lo preguntará interactivamente
python src/download.py --method wget --subset sleep-cassette --out data/raw
```

### Error: Hashes no coinciden

Los archivos pueden estar corruptos. Limpiá y reintentá:

```bash
python src/download.py --method wget --subset sleep-cassette --out data/raw --clean
```

### Error: Descarga incompleta

Si la descarga se interrumpió:

```bash
# Reintentar sin --clean (continúa donde quedó)
python src/download.py --method wget --subset sleep-cassette --out data/raw

# Validar
python -m src.check_data --raw-root data/raw --version 1.0.0 --subset sleep-cassette
```

## Problemas de preprocesamiento

### Error: "No se encontró ventana de sueño"

El sujeto puede no tener epochs de sueño anotados. Verificá el hipnograma:

```bash
python src/view_subject.py --subject-id SC4001E --list-channels
```

### Error: Archivos .fif muy grandes

Usá `--resample-sfreq` para reducir la frecuencia de muestreo:

```bash
python src/preprocessing.py \
  --manifest data/processed/manifest.csv \
  --resample-sfreq 100 \
  --out-root data/processed/sleep_trimmed
```

## Problemas de entrenamiento

### Error: "Dataset muy pequeño"

Con pocos sujetos, el split train/val/test puede fallar:

```python
ValueError: Dataset muy pequeño: 2 subject_core(s). Se necesitan al menos 3...
```

**Solución**: Usá cross-validation en lugar de split simple:

```bash
python -m src.models \
  --features-file data/processed/features.parquet \
  --model-type random_forest \
  --cross-validate \
  --cv-strategy loso
```

### Error: "No se pudo generar split con cobertura de clases"

Hay muy pocos sujetos con alguna clase. Opciones:

1. Usar más sujetos
2. Desactivar cobertura estricta (no recomendado)
3. Usar LOSO sin `--strict-class-coverage`

### Error: Modelo Keras sin estadísticas de normalización

Al cargar un modelo guardado incorrectamente:

```python
ValueError: El modelo Keras no tiene estadísticas de normalización guardadas...
```

**Solución**: Reentrenar el modelo. Los modelos nuevos guardan automáticamente `channel_means_`/`channel_stds_` (CNN1D) o `scaler_` (LSTM).

### Out of Memory (OOM) en modelos DL

Reducí el batch size:

```bash
python -m src.models \
  --model-type cnn1d \
  --batch-size 16 \
  --epochs 50
```

## Problemas de features

### Warning: "Error extrayendo features espectrales"

Puede ocurrir con epochs muy cortos o señales planas. El código asigna 0.0 y continúa. Verificá los datos:

```bash
python src/view_subject.py --subject-id SC4001E --duration 300
```

### Features con NaN

Las features entre canales pueden tener NaN si falta un canal. Por defecto se omiten si solo hay un EEG:

```bash
# Para incluirlas igualmente:
python -m src.extract_features \
  --manifest data/processed/manifest_trimmed.csv \
  --keep-cross-single-eeg
```

## FAQ

### ¿Cuánto tarda la extracción de features?

~20-30 minutos para ~150 sesiones en CPU.

### ¿Puedo usar GPU para modelos ML?

Random Forest y XGBoost usan CPU. Para GPU, usá modelos DL (CNN1D, LSTM) o entrenamiento en [Kaggle](KAGGLE_NOTEBOOKS.md).

### ¿Qué frecuencia de muestreo usar?

- 100 Hz: balance entre resolución y tamaño de archivo
- Original (~256 Hz): si necesitás frecuencias altas (gamma)

### ¿Qué epoch_length usar?

- 30 segundos: estándar en sleep staging (AASM)
- Otros valores pueden requerir ajustes en detección de spindles

### ¿Cómo sé si hay data leakage?

El pipeline divide por `subject_core` (no por epoch), garantizando que todas las noches de un sujeto van al mismo conjunto.

Verificá en los logs:

```
Division del dataset:
  Train: 45000 epochs (60.0%) de 90 subject_cores
  Test:  15000 epochs (20.0%) de 30 subject_cores
  Val:   15000 epochs (20.0%) de 30 subject_cores
```

Si ves "OVERLAP detectado", hay un bug.

### ¿Cómo interpreto Cohen's Kappa?

| Kappa | Interpretación |
|-------|----------------|
| < 0.20 | Pobre |
| 0.21 - 0.40 | Justo |
| 0.41 - 0.60 | Moderado |
| 0.61 - 0.80 | Sustancial |
| > 0.80 | Casi perfecto |

Para sleep staging, Kappa > 0.70 es considerado bueno.

## Calidad de código

### Pre-commit hooks

```bash
# Setup inicial
python -m pre_commit install

# Ejecutar manualmente
python -m pre_commit run --all-files
```

### Tests

```bash
# Ejecutar todos los tests
python -m pytest tests/ -v

# Solo tests de un módulo
python -m pytest tests/test_crossval.py -v
```

## Reportar bugs

Si encontrás un bug:

1. Verificá que estás usando la última versión
2. Revisá este documento
3. Incluí en el reporte:
   - Comando ejecutado
   - Error completo
   - Versión de Python y dependencias (`conda list`)
   - Sistema operativo
