# Features Extraídas para Clasificación de Estadios de Sueño

Este documento describe todas las features que se extraen para cada epoch de 30 segundos.

## Resumen

Para cada epoch se extraen **133 features** distribuidas en:

- **Features espectrales** (por canal): Potencia en bandas de frecuencia
- **Features temporales** (por canal): Estadísticas, entropía y parámetros temporales
- **Features de spindles** (canales EEG): Detección y características de spindles
- **Features de ondas lentas** (canales EEG): Características de slow waves
- **Features entre canales**: Correlaciones, ratios y coherencia

> [!TIP]
> Para documentación técnica detallada sobre procesamiento de señales, FFT, y teoría detrás de cada feature, ver:
> - [Informe Resumido](../data/processed/FEATURE_EXTRACTION_REPORT.md)
> - [Informe Completo](../data/processed/FEATURE_EXTRACTION_REPORT_COMPLETE.md) (incluye fundamentos teóricos de Fourier, Welch, filtros FIR, etc.)

---

## 1. Features Espectrales (por canal)

Para cada canal (EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, EMG submental):

### Potencia en bandas de frecuencia

**Bandas definidas:**
- **Delta**: 0.5 - 4 Hz (sueño profundo N3)
- **Theta**: 4 - 8 Hz (N1, REM)
- **Alpha**: 8 - 13 Hz (vigilia relajada)
- **Sigma**: 12 - 15 Hz (spindles, N2)
- **Beta**: 13 - 30 Hz (vigilia activa)
- **Gamma**: 30 - 45 Hz

**Features por banda:**
- `{canal}_rel_{banda}`: Potencia relativa (proporción del total)
- `{canal}_abs_{banda}`: Potencia absoluta

**Ejemplos:**
- `EEG Fpz-Cz_rel_delta`
- `EEG Fpz-Cz_abs_theta`
- `EOG horizontal_rel_alpha`
- `EMG submental_abs_beta`

### Ratios espectrales importantes

- `{canal}_theta_delta_ratio`: Ratio theta/delta (útil para N1 vs N3)
- `{canal}_alpha_delta_ratio`: Ratio alpha/delta
- `{canal}_sigma_delta_ratio`: Ratio sigma/delta (spindles)

### Frecuencia dominante

- `{canal}_dominant_freq`: Frecuencia con mayor potencia espectral (0.5-45 Hz)

**Total por canal:** 16 features espectrales × 4 canales = **64 features**

---

## 2. Features Temporales (por canal)

Para cada canal:

### Estadísticas básicas

- `{canal}_mean`: Media
- `{canal}_std`: Desviación estándar
- `{canal}_var`: Varianza
- `{canal}_min`: Mínimo
- `{canal}_max`: Máximo
- `{canal}_range`: Rango (peak-to-peak)

### Parámetros de Hjorth

- `{canal}_hjorth_activity`: Actividad (varianza)
- `{canal}_hjorth_mobility`: Movilidad (complejidad espectral)
- `{canal}_hjorth_complexity`: Complejidad

### Entropía

- `{canal}_entropy`: Entropía de Shannon mejorada (100 bins, usando scipy.stats.entropy)
- `{canal}_spectral_entropy`: Entropía espectral (complejidad de la distribución espectral)

> **Nota:** La entropía de Shannon ahora usa 100 bins (antes 20) para mejor resolución, y se agregó entropía espectral para capturar la complejidad de la señal en el dominio frecuencial.

### Zero Crossing Rate

- `{canal}_zcr`: Tasa de cruces por cero (frecuencia aproximada)

**Total por canal:** 12 features temporales × 4 canales = **48 features**

> **Nota:** Los canales EEG también tienen features de spindles y slow waves (ver secciones siguientes), por lo que el total por canal EEG es mayor.

---

## 3. Features de Spindles (solo canales EEG)

Los spindles son ondas breves (0.5-2s) en la banda sigma (12-15 Hz), característicos del estadio N2.

Para cada canal EEG (EEG Fpz-Cz, EEG Pz-Oz):

- `{canal}_spindle_count`: Número de spindles detectados en el epoch
- `{canal}_spindle_density`: Spindles por minuto
- `{canal}_spindle_mean_duration`: Duración media de spindles (segundos)
- `{canal}_spindle_mean_amplitude`: Amplitud media de spindles (µV)

> **Nota:** La detección usa YASA (`yasa.spindles_detect`) con parámetros basados en criterios AASM y literatura:
> - **Banda**: 12-15 Hz (AASM, 2007)
> - **Duración**: 0.5-2s (Warby et al., 2014; Purcell et al., 2017)
> - **Umbrales**: rel_pow=0.2, corr=0.65, rms=1.5 (defaults YASA, validados vs. scoring manual en MASS)
>
> Referencias:
> - AASM (2007). The AASM Manual for the Scoring of Sleep.
> - Warby et al. (2014). Sleep spindle measurements. *Sleep*, 37(9), 1469-1479.
> - Vallat & Walker (2021). An open-source, high-performance tool for automated sleep staging. *eLife*, 10:e70092.

**Total:** 4 features × 2 canales EEG = **8 features**

---

## 4. Features de Ondas Lentas / Slow Waves (solo canales EEG)

Las ondas lentas son ondas de alta amplitud en la banda delta (0.5-4 Hz), características del estadio N3.

Para cada canal EEG:

- `{canal}_slow_wave_power`: Potencia absoluta en banda delta
- `{canal}_slow_wave_ratio`: Ratio de potencia delta vs. total
- `{canal}_slow_wave_peak_amplitude`: Amplitud pico-a-pico en banda delta

**Total:** 3 features × 2 canales EEG = **6 features**

---

## 5. Features Entre Canales (Cross-Channel)

### Correlaciones

- `eeg_eeg_correlation`: Correlación entre los dos canales EEG
- `eeg_eog_correlation`: Correlación entre EEG y EOG (importante para REM)

### Ratios de potencia

- `emg_eeg_ratio`: Ratio potencia EMG/EEG (distinguir REM de vigilia)
- `eog_emg_ratio`: Ratio potencia EOG/EMG (REM: EOG alto, EMG bajo)

### Coherencia

- `eeg_eog_theta_coherence`: Coherencia en banda theta (4-8 Hz) entre EEG y EOG
- `eeg_eog_delta_coherence`: Coherencia en banda delta (0.5-4 Hz) entre EEG y EOG
- `eeg_eog_sigma_coherence`: Coherencia en banda sigma (12-15 Hz) entre EEG y EOG

> **Nota:** La coherencia ahora se calcula correctamente usando `scipy.signal.coherence` (antes usaba un cálculo manual aproximado).

**Total:** **7 features entre canales**

---

## 6. Metadata (no son features para el modelo)

- `stage`: Etiqueta del estadio (W, N1, N2, N3, REM)
- `subject_id`: ID del sujeto
- `subject_core`: Primeros 5 caracteres del ID (para agrupar)
- `session_idx`: Índice de la sesión
- `epoch_time_start`: Tiempo de inicio del epoch
- `epoch_index`: Índice del epoch dentro de la sesión
- `episode_index` / `episodes_total`: Identificadores cuando se segmentan múltiples episodios de sueño

---

## Total de Features para el Modelo

**133 features** por epoch:

| Categoría | Features por canal | Canales | Total |
|-----------|-------------------|---------|-------|
| Espectrales | 16 | 4 | 64 |
| Temporales + Hjorth + Entropías | 12 | 4 | 48 (pero ver nota) |
| Spindles | 4 | 2 (EEG) | 8 |
| Ondas lentas | 3 | 2 (EEG) | 6 |
| Cross-channel | - | - | 7 |

**Desglose real verificado:**
- EEG Fpz-Cz: 35 features
- EEG Pz-Oz: 35 features
- EOG horizontal: 28 features
- EMG submental: 28 features
- Cross-channel: 7 features
- **Total: 133 features**

---

## Notas Técnicas

- Las features se calculan usando **YASA** (Yet Another Spindle Algorithm) para análisis espectral y detección de spindles
- El análisis espectral usa el método de **Welch** para calcular la densidad espectral de potencia (PSD)
- Opcional: PSD multi-taper (`--psd-method multitaper`) para mejorar estabilidad espectral en señales cortas
- La coherencia usa **scipy.signal.coherence** para un cálculo correcto
- La entropía usa **scipy.stats.entropy** con 100 bins para mejor resolución
- Las features están optimizadas para distinguir estadios de sueño con pocos canales (2 EEG + EOG + EMG)
- Cada canal se prefiltra (0.3–45 Hz) con detrend y notch 50/60 Hz antes de extraer PSD, spindles o slow waves; los flags de CLI permiten desactivar o ajustar banda/notch.
- Si sólo hay un EEG disponible, se omiten las features EEG-EEG/coherencia en vez de rellenar NaN
- La entropía espectral usa `fs=sfreq`; si no se provee `sfreq`, se deriva `fs = len(data) / epoch_length` para el Welch, de modo que cambiar `epoch_length` mantiene la escala correcta.
- Se valida que `overlap < epoch_length` y que el paso `(epoch_length - overlap) * sfreq` sea > 0; de lo contrario se lanza un `ValueError` con mensaje claro.
- Los epochs sin etiqueta se descartan y se loggea cuántos y qué porcentaje se pierden por sesión, antes de extraer features.
- La densidad de spindles se calcula con la duración real del epoch `len(data)/sfreq`, por lo que respeta `epoch_length` distintos de 30 s.
- El código maneja errores: si falla alguna feature, se asigna 0.0 en lugar de fallar
- Los ratios espectrales usan epsilon (1e-10) para evitar división por cero

### Splits y guardarraíles de modelado

- Los splits se hacen por `subject_core` para evitar leakage; se estratifican por estadio dominante del sujeto y se reintenta hasta lograr cobertura de todas las clases presentes en train/val/test (o se lanza un error si es imposible con los tamaños solicitados).
- Split temporal opcional por sesión/noches (`--temporal-split`): mantiene las sesiones más recientes en test/val y usa `GroupTimeSeriesSplit` en CV; requiere `epoch_time_start` o `epoch_index`.
- Modelos CNN1D/LSTM guardan estadísticas de normalización (`channel_means_`/`channel_stds_` o `scaler_`) y la evaluación falla si no están presentes para evitar normalizar con datos de test.

---

## Changelog

### v1.1.0 (Actual)
- Agregada entropía espectral
- Agregadas features de spindles (count, density, duration, amplitude)
- Agregadas features de ondas lentas (power, ratio, peak_amplitude)
- Agregada coherencia en bandas delta y sigma
- Mejorada entropía de Shannon (100 bins vs 20)
- Corregido cálculo de coherencia usando scipy.signal.coherence
- Corregida división por cero en ratios espectrales
- Documentado filtrado/notch configurable y nuevos guardarraíles de splitting (cobertura de clases, split temporal opcional) y normalización obligatoria en DL

### v1.0.0
- Versión inicial con features espectrales, temporales y entre canales
