# Informe de Extracción de Features

**Dataset:** Sleep-EDF (PhysioNet)
**Proyecto:** sleep-ST

---

## 1. Resumen Ejecutivo

Se extrajeron **133 features** de **186,499 epochs** (30s cada uno) correspondientes a **78 sujetos** (153 sesiones/noches) del dataset Sleep-EDF. El proceso incluyó corrección de un bug crítico en el preprocesamiento y validación científica de los resultados.

| Métrica | Valor |
|---------|-------|
| Archivo de salida | `features_resamp200.parquet` |
| Tamaño | 176 MB |
| Total epochs | 186,499 |
| Total sujetos | 78 |
| Total sesiones/noches | 153 |
| Total features | 133 |
| Valores NaN | 0.00% |

---

## 2. Pipeline de Datos

### 2.1 Datos de Entrada

```
Manifest: data/processed/manifest_trimmed_resamp200.csv
```

**Preprocesamiento aplicado:**
- Trimming: ±15 min alrededor del sueño anotado
- Filtro band-pass: 0.3–45 Hz (FIR, firwin)
- Filtro notch: 50 Hz
- Resample: 200 Hz
- Referencia promedio EEG (solo canales EEG reales)

### 2.2 Comando de Extracción

```bash
python -m src.extract_features \
  --manifest data/processed/manifest_trimmed_resamp200.csv \
  --output data/processed/features_resamp200.parquet \
  --no-prefilter \
  --epoch-length 30 \
  --psd-method welch \
  --movement-policy drop
```

**Parámetros:**
- `--no-prefilter`: Los datos ya están filtrados en preprocesamiento
- `--epoch-length 30`: Epochs de 30 segundos (estándar AASM)
- `--psd-method welch`: Método Welch para estimación espectral
- `--movement-policy drop`: Descartar epochs con movimiento

---

## 3. Bug Corregido

### 3.1 Problema Detectado

Durante el smoke test inicial, se detectó que el **99.9% de la potencia espectral** estaba concentrada en la banda delta, lo cual es fisiológicamente imposible.

**Causa raíz:** La función `set_eeg_reference('average')` en MNE incluía todos los canales (EMG, temperatura, markers) porque los archivos EDF de Sleep-EDF no especifican correctamente los tipos de canal.

### 3.2 Solución Implementada

Se agregó la función `_fix_channel_types()` en `src/preprocessing.py`:

```python
CHANNEL_TYPE_MAPPING = {
    "EEG Fpz-Cz": "eeg",
    "EEG Pz-Oz": "eeg",
    "EOG horizontal": "eog",
    "Resp oro-nasal": "misc",
    "EMG submental": "emg",
    "Temp rectal": "misc",
    "Event marker": "stim",
}
```

### 3.3 Verificación

| Métrica | Antes (corrupto) | Después (corregido) |
|---------|------------------|---------------------|
| EEG std | ~600 V | ~10 µV |
| Delta relativo | 99.9% | 66.2% |
| Spindles detectados | 0% | 20.2% |

---

## 4. Resultados

### 4.1 Distribución de Estadios de Sueño

| Estadio | Epochs | Porcentaje | Descripción |
|---------|--------|------------|-------------|
| N2 | 69,132 | 37.1% | Sueño ligero (spindles) |
| W | 56,971 | 30.5% | Vigilia |
| REM | 25,835 | 13.9% | Sueño REM |
| N1 | 21,522 | 11.5% | Transición sueño |
| N3 | 13,039 | 7.0% | Sueño profundo (SWS) |

### 4.2 Features Espectrales

**Potencia relativa por banda (EEG Fpz-Cz):**

| Banda | Frecuencia | Media ± STD | Rango esperado |
|-------|------------|-------------|----------------|
| Delta | 0.5–4 Hz | 0.662 ± 0.182 | 50–80% |
| Theta | 4–8 Hz | 0.135 ± 0.065 | 5–20% |
| Alpha | 8–13 Hz | 0.085 ± 0.076 | 5–15% |
| Sigma | 12–15 Hz | 0.029 ± 0.023 | 1–5% |
| Beta | 13–30 Hz | 0.080 ± 0.071 | 5–15% |
| Gamma | 30–45 Hz | 0.009 ± 0.015 | <5% |

**Suma de potencias relativas:** 1.029 ≈ 1.0

### 4.3 Detección de Spindles

| Métrica | Valor |
|---------|-------|
| Epochs con spindles | 37,605 (20.2%) |
| Densidad media | 3.46 spindles/min |
| Duración media | ~0.8 s |

**Parámetros de detección (AASM):**
- Banda: 12–15 Hz (sigma)
- Duración: 0.5–2.0 s
- Método: YASA `spindles_detect()`

---

## 5. Features Extraídas (133 total)

### 5.1 Por Canal

| Canal | Tipo | Features |
|-------|------|----------|
| EEG Fpz-Cz | EEG | Espectrales + Spindles + Estadísticas |
| EEG Pz-Oz | EEG | Espectrales + Spindles + Estadísticas |
| EOG horizontal | EOG | Espectrales + Estadísticas |
| EMG submental | EMG | Espectrales + Estadísticas |

### 5.2 Tipos de Features

**Por cada canal:**
- **Potencia absoluta (6):** delta, theta, alpha, sigma, beta, gamma
- **Potencia relativa (6):** rel_delta, rel_theta, rel_alpha, rel_sigma, rel_beta, rel_gamma
- **Estadísticas temporales (7):** mean, std, skewness, kurtosis, ptp, rms, zcr
- **Spindles (solo EEG, 4):** count, density, mean_duration, mean_amplitude

**Features globales:**
- `subject_id`: Identificador del sujeto
- `epoch_idx`: Índice del epoch
- `stage`: Estadio de sueño (etiqueta)

---

## 6. Validación Científica

### 6.1 Referencias

- **Bandas de frecuencia:** Estándar AASM (Berry et al., 2017)
- **Detección de spindles:** Warby et al. (2014), Purcell et al. (2017)
- **Librería YASA:** Vallat & Walker (2021) - validada contra scoring manual

### 6.2 Verificaciones Realizadas

1. Suma de potencias relativas ≈ 1.0
2. Valores fisiológicamente realistas por banda
3. Spindles detectados principalmente en N2
4. Sin valores NaN
5. Tests de reproducibilidad (seed fijo)

---

## 7. Resultados del Modelado

Los features extraídos fueron utilizados para entrenar modelos de **Machine Learning clásico**:

| Modelo | Cohen's Kappa | F1 Macro | Uso |
|--------|---------------|----------|-----|
| **XGBoost LOSO** | **0.641** | 70.02% | ML interpretable (elegido) |
| Random Forest | 0.635 | 69.50% | Baseline |

> **Nota:** Los modelos de Deep Learning (CNN1D, LSTM) trabajan directamente sobre la señal cruda, sin usar estas features.
>
> **Ver análisis completo:** [docs/reports/COMPARATIVE_ANALYSIS.md](../../docs/reports/COMPARATIVE_ANALYSIS.md)

---

## 8. Archivos Generados

```
data/processed/
├── features_resamp200.parquet      # Features finales (176 MB)
├── features_resamp200_test.parquet # Smoke test (3 sujetos)
├── manifest_trimmed_resamp200.csv  # Manifest de sesiones
└── sleep_trimmed_resamp200/        # Datos PSG procesados
    ├── psg/                        # Archivos .fif
    └── hypnograms/                 # Anotaciones .csv
```

---
