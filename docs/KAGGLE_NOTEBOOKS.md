# Notebooks para Entrenamiento en Kaggle

Este documento describe los notebooks disponibles para entrenar modelos de clasificacion de etapas del sueno en Kaggle utilizando GPUs Tesla T4.

## Requisitos de Hardware

- **Plataforma**: Kaggle Notebooks
- **Acelerador**: 2x Tesla T4 (16GB VRAM cada una)
- **Estrategia**: TensorFlow MirroredStrategy para entrenamiento multi-GPU

## Datos a Subir a Kaggle

Antes de ejecutar los notebooks, debes subir los datos procesados a Kaggle como un dataset.

### Estructura del Dataset

Usa el dataset de Kaggle `sleep-edf-trimmed-f32` (https://www.kaggle.com/datasets/ignaciolinari/sleep-edf-trimmed-f32), que contiene los archivos preprocesados a **100 Hz** en float32:

```
sleep-edf-trimmed-f32/
├── manifest_trimmed_spt.csv
└── sleep_trimmed_spt/
    ├── psg/
    │   ├── SC4001E_sleep-cassette_1.0.0_trimmed_raw.fif
    │   ├── SC4002E_sleep-cassette_1.0.0_trimmed_raw.fif
    │   └── ... (todos los archivos .fif)
    └── hypnograms/
        ├── SC4001E_sleep-cassette_1.0.0_trimmed_annotations.csv
        ├── SC4002E_sleep-cassette_1.0.0_trimmed_annotations.csv
        └── ... (todos los archivos de anotaciones)
```

### Archivos Necesarios

| Carpeta | Contenido | Descripcion |
|---------|-----------|-------------|
| `manifest_trimmed_spt.csv` | Archivo raiz | Mapeo de sujetos a archivos PSG y hypnogramas |
| `sleep_trimmed_spt/psg/` | Archivos `.fif` | Datos de polisomnografia (senales EEG, EOG, EMG) |
| `sleep_trimmed_spt/hypnograms/` | Archivos `.csv` | Anotaciones de etapas del sueno por epoca |

> **Nota:** Los notebooks usan 100 Hz por defecto (`sfreq=100`) para evitar problemas de memoria en Kaggle. Si usas la versión 200 Hz, el manifest y carpetas se llaman `manifest_trimmed_resamp200.csv` y `sleep_trimmed_resamp200/`.

### Pasos para Subir

1. Ir a [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click en "New Dataset" (o "New Version" si ya existe)
3. Nombrar el dataset: `sleep-edf-trimmed-f32`
4. Subir el contenido de `data/processed/` correspondiente:
   - `manifest_trimmed_spt.csv`
   - Carpeta `sleep_trimmed_spt/` completa (`psg/` y `hypnograms/`)
5. Publicar el dataset (puede ser privado)

## Notebooks Disponibles

### A_CNN1D_KAGGLE.ipynb

**Modelo**: CNN1D con bloques residuales

**Arquitectura**:
- Bloques convolucionales 1D con conexiones residuales
- Kernel sizes: 3, 5, 7 (capturan patrones a diferentes escalas temporales)
- BatchNormalization + Dropout para regularizacion
- GaussianNoise para data augmentation
- Global Average Pooling antes de la clasificacion

**Hiperparametros Optimizables** (via Optuna):
- `n_blocks`: Numero de bloques residuales (2-5)
- `filters`: Filtros por bloque (32-256)
- `kernel_size`: Tamano del kernel (3, 5, 7)
- `dropout_rate`: Tasa de dropout (0.2-0.5)
- `learning_rate`: Tasa de aprendizaje (1e-5 a 1e-2)
- `noise_stddev`: Ruido gaussiano (0.01-0.1)

### B_LSTM_KAGGLE.ipynb

**Modelo**: BiLSTM con capa de Atencion

**Arquitectura**:
- Capas Bidirectional LSTM apiladas
- Capa de Atencion personalizada (aprende pesos sobre la secuencia temporal)
- Optimizado para cuDNN (sin `recurrent_dropout`)
- Dropout entre capas LSTM
- Dense con activacion softmax para 5 clases

**Hiperparametros Optimizables** (via Optuna):
- `n_layers`: Numero de capas LSTM (1-3)
- `units`: Unidades por capa LSTM (64-256)
- `dropout_rate`: Tasa de dropout (0.2-0.5)
- `learning_rate`: Tasa de aprendizaje (1e-5 a 1e-2)
- `use_attention`: Usar capa de atencion (True/False)

## Flujo de Trabajo Comun

Ambos notebooks siguen el mismo flujo:

```
1. Configuracion del Entorno
   └── Detectar GPUs y configurar MirroredStrategy

2. Carga de Datos
   └── Leer manifest_trimmed.csv
   └── Cargar archivos .fif con MNE
   └── Cargar anotaciones de hypnogramas

3. Preprocesamiento
   └── Segmentar en epocas de 30 segundos
   └── Normalizar senales
   └── Mapear etapas a indices (0-4)

4. Division de Datos
   └── Split por sujeto (no por epoca)
   └── Train: 70% | Val: 15% | Test: 15%

5. Entrenamiento Baseline
   └── Modelo con hiperparametros por defecto
   └── Early stopping + ReduceLROnPlateau

6. Evaluacion
   └── Matriz de confusion
   └── Metricas por clase (precision, recall, F1)
   └── Cohen's Kappa

7. Optimizacion con Optuna
   └── 20 trials de busqueda
   └── Pruning de trials no prometedores
   └── Guardar mejor modelo

8. Guardar Resultados
   └── Modelo en formato .keras
   └── Historial de Optuna
```

## Clases de Salida

El problema es clasificacion multiclase con 5 etapas del sueno:

| Indice | Etapa | Descripcion |
|--------|-------|-------------|
| 0 | W | Vigilia (Wake) |
| 1 | N1 | Sueno ligero etapa 1 |
| 2 | N2 | Sueno ligero etapa 2 |
| 3 | N3 | Sueno profundo (SWS) |
| 4 | REM | Sueno REM |

## Metricas de Evaluacion

- **Accuracy**: Precision global
- **Macro F1-Score**: Promedio de F1 por clase (maneja desbalance)
- **Cohen's Kappa**: Acuerdo mas alla del azar
- **Matriz de Confusion**: Visualizacion de errores por clase

## Tips para Kaggle

1. **Tiempo de GPU**: Kaggle limita a 30 horas semanales de GPU. Usa `n_trials` moderado en Optuna.

2. **Guardar Checkpoints**: Los notebooks guardan automaticamente el mejor modelo con `ModelCheckpoint`.

3. **Output**: Guarda los modelos finales en `/kaggle/working/` para descargarlos.

4. **Memoria**: Con 2x T4 tienes 32GB de VRAM total. Si usas el dataset 200 Hz, déjalo en `sfreq=100` (los notebooks ya re-muestrean) o baja `batch_size`; la versión 100 Hz (`manifest_trimmed_spt.csv`) suele evitar OOM.

5. **Persistencia**: Usa `kaggle datasets` para guardar modelos entrenados como datasets.

## Reanudacion desde Checkpoints

Si Kaggle se desconecta durante el entrenamiento, puedes reanudar facilmente:

### Pasos para reanudar:

1. **Antes de que se caiga**: El `ModelCheckpoint` callback guarda automaticamente el mejor modelo en cada epoch que mejora.

2. **Despues de reconectar**:
   - Abre el notebook
   - Ejecuta las celdas de configuracion, imports y carga de datos
   - En la celda de "Reanudar desde Checkpoint":
     ```python
     RESUME_FROM_CHECKPOINT = True
     CHECKPOINT_NAME = "cnn1d_20251125_143022_best.keras"  # Ajustar nombre
     ```
   - Ejecuta esa celda para cargar el modelo
   - Continua con la celda de `model.fit()` - el modelo seguira entrenando

### Notas importantes:

- El historial de entrenamiento previo se pierde (solo se guarda el modelo, no el historial)
- El learning rate scheduler (`ReduceLROnPlateau`) reinicia desde el valor inicial
- Para optimizacion con Optuna, cada trial guarda su mejor modelo individualmente

## Estructura de Archivos de Salida

Despues de ejecutar los notebooks, tendras:

```
/kaggle/working/
├── cnn1d_baseline.keras          # Modelo CNN1D baseline
├── cnn1d_best_optuna.keras       # Mejor modelo CNN1D optimizado
├── lstm_baseline.keras           # Modelo LSTM baseline
├── lstm_best_optuna.keras        # Mejor modelo LSTM optimizado
└── optuna_history_*.csv          # Historial de optimizacion
```

## Resultados Obtenidos

Los modelos entrenados en Kaggle lograron los siguientes resultados:

| Modelo | Cohen's Kappa | F1 Macro | Accuracy | Tiempo |
|--------|---------------|----------|----------|--------|
| **CNN1D** | **0.680** | 70.83% | 76.86% | ~105 min |
| LSTM Bi + Attention | 0.651 | 68.07% | 74.64% | ~200 min |
| LSTM Bidireccional | 0.521 | 58.18% | 65.41% | ~372 min |
| LSTM Unidireccional | 0.530 | 58.59% | 66.17% | ~202 min |

> **Hallazgo clave:** CNN1D supera a todas las variantes LSTM para clasificación single-epoch, y es ~4x más eficiente en tiempo de entrenamiento.

Ver [Análisis Comparativo](reports/COMPARATIVE_ANALYSIS.md) para detalles completos.

## Referencias

- Dataset original: [Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/)
- MNE-Python: [mne.tools](https://mne.tools/)
- Optuna: [optuna.org](https://optuna.org/)
- TensorFlow: [tensorflow.org](https://tensorflow.org/)
