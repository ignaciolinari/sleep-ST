# Features Extraídas para Clasificación de Estadios de Sueño

Este documento describe todas las features que se extraen para cada epoch de 30 segundos.

## Resumen

Para cada epoch se extraen aproximadamente **~100-120 features** distribuidas en:

- **Features espectrales** (por canal): Potencia en bandas de frecuencia
- **Features temporales** (por canal): Estadísticas y parámetros temporales
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

- `{canal}_entropy`: Entropía de Shannon (medida de aleatoriedad/complejidad)

### Zero Crossing Rate

- `{canal}_zcr`: Tasa de cruces por cero (frecuencia aproximada)

**Total por canal:** ~11 features temporales × 4 canales = **~44 features**

---

## 3. Features Entre Canales (Cross-Channel)

### Correlaciones

- `eeg_eeg_correlation`: Correlación entre los dos canales EEG
- `eeg_eog_correlation`: Correlación entre EEG y EOG (importante para REM)

### Ratios de potencia

- `emg_eeg_ratio`: Ratio potencia EMG/EEG (distinguir REM de vigilia)
- `eog_emg_ratio`: Ratio potencia EOG/EMG (REM: EOG alto, EMG bajo)

### Coherencia

- `eeg_eog_theta_coherence`: Coherencia en banda theta entre EEG y EOG

**Total:** **~5 features entre canales**

---

## 4. Metadata (no son features para el modelo)

- `stage`: Etiqueta del estadio (W, N1, N2, N3, REM)
- `subject_id`: ID del sujeto
- `subject_core`: Primeros 5 caracteres del ID (para agrupar)
- `session_idx`: Índice de la sesión
- `epoch_time_start`: Tiempo de inicio del epoch
- `epoch_index`: Índice del epoch dentro de la sesión

---

## Total de Features para el Modelo

**Aproximadamente 109 features** por epoch:
- ~60 features espectrales (15 × 4 canales)
- ~44 features temporales (11 × 4 canales)
- ~5 features entre canales

---

## Notas

- Las features se calculan usando **YASA** (Yet Another Spindle Algorithm) para análisis espectral
- El análisis espectral usa el método de **Welch** para calcular la densidad espectral de potencia (PSD)
- Las features están optimizadas para distinguir estadios de sueño con pocos canales (2 EEG + EOG + EMG)
- El código maneja errores: si falla alguna feature, se asigna 0.0 en lugar de fallar
