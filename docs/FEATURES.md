# Features Extraídas para Clasificación de Estadios de Sueño

Este documento describe todas las features que se extraen para cada epoch de 30 segundos.

## Resumen

Para cada epoch se extraen aproximadamente **~130-150 features** distribuidas en:

- **Features espectrales** (por canal): Potencia en bandas de frecuencia
- **Features temporales** (por canal): Estadísticas, entropía y parámetros temporales
- **Features de spindles** (canales EEG): Detección y características de spindles
- **Features de ondas lentas** (canales EEG): Características de slow waves
- **Features entre canales**: Correlaciones, ratios y coherencia

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

**Total por canal:** ~15 features espectrales × 4 canales = **~60 features**

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

**Total por canal:** ~12 features temporales × 4 canales = **~48 features**

---

## 3. Features de Spindles (solo canales EEG)

Los spindles son ondas breves (0.5-2s) en la banda sigma (12-15 Hz), característicos del estadio N2.

Para cada canal EEG (EEG Fpz-Cz, EEG Pz-Oz):

- `{canal}_spindle_count`: Número de spindles detectados en el epoch
- `{canal}_spindle_density`: Spindles por minuto
- `{canal}_spindle_mean_duration`: Duración media de spindles (segundos)
- `{canal}_spindle_mean_amplitude`: Amplitud media de spindles (µV)

> **Nota:** La detección usa YASA (`yasa.spindles_detect`) con parámetros estándar: banda 12-15 Hz, duración 0.5-2s.

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

**Total:** **~7 features entre canales**

---

## 6. Metadata (no son features para el modelo)

- `stage`: Etiqueta del estadio (W, N1, N2, N3, REM)
- `subject_id`: ID del sujeto
- `subject_core`: Primeros 5 caracteres del ID (para agrupar)
- `session_idx`: Índice de la sesión
- `epoch_time_start`: Tiempo de inicio del epoch
- `epoch_index`: Índice del epoch dentro de la sesión

---

## Total de Features para el Modelo

**Aproximadamente 129 features** por epoch:
- ~60 features espectrales (15 × 4 canales)
- ~48 features temporales (12 × 4 canales, incluyendo entropía espectral)
- ~8 features de spindles (4 × 2 canales EEG)
- ~6 features de ondas lentas (3 × 2 canales EEG)
- ~7 features entre canales

---

## Notas Técnicas

- Las features se calculan usando **YASA** (Yet Another Spindle Algorithm) para análisis espectral y detección de spindles
- El análisis espectral usa el método de **Welch** para calcular la densidad espectral de potencia (PSD)
- La coherencia usa **scipy.signal.coherence** para un cálculo correcto
- La entropía usa **scipy.stats.entropy** con 100 bins para mejor resolución
- Las features están optimizadas para distinguir estadios de sueño con pocos canales (2 EEG + EOG + EMG)
- Cada canal se prefiltra (0.3–45 Hz) con detrend y notch 50/60 Hz antes de extraer PSD, spindles o slow waves
- Si sólo hay un EEG disponible, `eeg_eeg_correlation` se marca como NaN en lugar de duplicar el canal
- El código maneja errores: si falla alguna feature, se asigna 0.0 en lugar de fallar
- Los ratios espectrales usan epsilon (1e-10) para evitar división por cero

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

### v1.0.0
- Versión inicial con features espectrales, temporales y entre canales
