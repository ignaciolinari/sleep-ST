# Informe Completo de Extracción de Features

**Dataset:** Sleep-EDF (PhysioNet)
**Proyecto:** sleep-ST

---

## 1. Resumen Ejecutivo

Se extrajeron **133 features** de **186,499 epochs** (30 segundos cada uno) correspondientes a **78 sujetos** (153 sesiones/noches) del dataset Sleep-EDF. Algunos sujetos tienen 2 noches de grabación. Este documento proporciona una explicación teórica completa del procesamiento de señales aplicado, incluyendo transformaciones del dominio temporal al frecuencial, filtros digitales, y métodos de estimación espectral.

---

## 2. Fundamentos Teóricos del Procesamiento de Señales

### 2.1 Representación de Señales: Dominio del Tiempo vs. Frecuencia

Las señales EEG capturadas representan actividad eléctrica cerebral en el **dominio del tiempo** - una secuencia de valores de voltaje muestreados a intervalos regulares (200 Hz en este proyecto).

```
Dominio del Tiempo: x[n], n = 0, 1, 2, ..., N-1
  → Representa la amplitud de la señal en cada instante

Dominio de la Frecuencia: X[k], k = 0, 1, 2, ..., N-1
  → Representa las componentes frecuenciales (cuánta "potencia" hay en cada frecuencia)
```

**¿Por qué transformar al dominio de la frecuencia?**

Los estadios de sueño se caracterizan por patrones de actividad oscilatoria específicos:

| Estadio | Banda Dominante | Características |
|---------|-----------------|-----------------|
| W (Vigilia) | Alpha (8-13 Hz) | Ojos cerrados, relajado |
| N1 | Theta (4-8 Hz) | Transición al sueño |
| N2 | Sigma/Spindles (12-15 Hz) | Sleep spindles, K-complexes |
| N3 | Delta (0.5-4 Hz) | Ondas lentas de alta amplitud |
| REM | Mixto (Theta + Beta) | Similar a vigilia, sin tono muscular |

### 2.2 Teorema de Nyquist-Shannon (Muestreo)

El **Teorema de Nyquist** establece que para capturar correctamente una señal, la frecuencia de muestreo debe ser al menos el doble de la frecuencia máxima de interés:

```
fs ≥ 2 · f_max

donde:
- fs = frecuencia de muestreo (200 Hz en nuestro caso)
- f_max = frecuencia máxima de interés (45 Hz)

Verificación: 200 Hz ≥ 2 · 45 Hz = 90 Hz  ✓
```

**Frecuencia de Nyquist:** f_nyq = fs / 2 = 100 Hz

Si intentáramos capturar frecuencias por encima de 100 Hz con muestreo a 200 Hz, ocurriría **aliasing** (las frecuencias altas "se disfrazan" de frecuencias bajas, corrompiendo la señal).

```
+----------------------------------------------------+
|              ALIASING (que evitamos)               |
|                                                    |
|  Senal real a 120 Hz  ->  Aparece como 80 Hz       |
|  (fs=200, f_nyq=100)      (120 se "refleja")       |
|                                                    |
|  Solucion: Filtro anti-aliasing a 45 Hz            |
|            (antes de muestrear o como band-pass)   |
+----------------------------------------------------+
```

**¿Por qué 200 Hz y no 100 Hz?**

- Sleep-EDF original viene a 100 Hz
- Resampleamos a 200 Hz para mejor resolución en gamma (30-45 Hz)
- Más muestras = mejor precisión temporal en detección de spindles

---

## 3. Pipeline de Preprocesamiento

### 3.1 Diagrama de Flujo

```
┌─────────────────┐
│ EDF Raw (100Hz) │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ 1. Corrección de Tipos de Canal          │
│    CHANNEL_TYPE_MAPPING:                 │
│    - EEG Fpz-Cz → eeg                    │
│    - EEG Pz-Oz  → eeg                    │
│    - EOG horizontal → eog                │
│    - EMG submental → emg                 │
│    - Temp rectal → misc                  │
│    - Event marker → stim                 │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ 2. Resampling: 100 Hz → 200 Hz           │
│    (Interpolación para mejor resolución  │
│    espectral en bandas altas)            │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ 3. Filtro Band-Pass: 0.3 - 45 Hz         │
│    Tipo: FIR (firwin)                    │
│    Diseño: Ventana de Hamming            │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ 4. Filtro Notch: 50 Hz                   │
│    (Elimina interferencia de línea AC)   │
│    Tipo: FIR (firwin)                    │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ 5. Referencia Promedio EEG               │
│    (Solo canales EEG reales)             │
│    ref = (Fpz-Cz + Pz-Oz) / 2            │
└────────┬─────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│ FIF Procesado   │
└─────────────────┘
```

### 3.2 Filtro Band-Pass (0.3 - 45 Hz)

**Tipo:** Filtro FIR (Finite Impulse Response)
**Diseño:** `firwin` (ventana de Hamming)
**Implementación:** MNE-Python `filter_data()`

#### Teoría del Filtro FIR

Un filtro FIR tiene la forma:

```
y[n] = Σ(k=0 to M) b[k] · x[n-k]
```

Donde:
- `b[k]` son los coeficientes del filtro
- `M` es el orden del filtro
- La respuesta es **siempre estable** (no tiene polos)
- Tiene **fase lineal** (no distorsiona la forma de onda)

**Diseño con ventana (firwin):**

1. Diseñar filtro ideal (rect en frecuencia)
2. Aplicar IFFT → sinc en tiempo
3. Truncar con ventana de Hamming → suavizar transición

#### Convolución: El Corazón de los Filtros FIR

Los filtros FIR se implementan mediante **convolución** en el dominio del tiempo:

```
y[n] = (x * h)[n] = Σ(k=0 to M) h[k] · x[n-k]
```

Donde:
- `x[n]` = señal de entrada
- `h[k]` = respuesta al impulso del filtro (kernel/coeficientes)
- `y[n]` = señal filtrada
- `*` = operador de convolución

**Interpretación gráfica:**

```
Señal x[n]:    ─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─►
               │ │ │ │ │ │ │ │ │ │

Kernel h[k]:       ┌─┬─┬─┐
                   │h₀│h₁│h₂│  (ventana deslizante)
                   └─┴─┴─┘
                      ↓
                   y[n] = Σ h[k]·x[n-k]
```

**Propiedad fundamental:** La convolución en tiempo equivale a multiplicación en frecuencia:

```
Dominio del Tiempo:    y[n] = x[n] * h[n]
                           ⟷  (FFT/IFFT)
Dominio de Frecuencia: Y[k] = X[k] · H[k]
```

Esto significa que cuando convolucionamos la señal EEG con el kernel del filtro band-pass, estamos **multiplicando** el espectro de la señal por la respuesta en frecuencia del filtro (que atenúa frecuencias fuera de 0.3-45 Hz).

**Implementación eficiente:** Para señales largas, MNE usa FFT para calcular la convolución:
1. `X = FFT(x)`
2. `H = FFT(h)` (respuesta en frecuencia)
3. `Y = X · H`
4. `y = IFFT(Y)`

Esto es más rápido que convolución directa: O(N log N) vs O(N·M)

**¿Por qué 0.3 - 45 Hz?**

```
┌─────────────────────────────────────────────────────────┐
│ Frecuencia (Hz)                                          │
│                                                          │
│ 0    0.3         4    8   13  15    30    45    50    100│
│ │     │          │    │    │   │     │     │     │      │
│ ├─────┼──────────┴────┴────┴───┴─────┴─────┼─────┼──────┤
│ │ DC  │         Bandas de Interés          │Notch│ Alias│
│ │drift│   δ    θ    α    σ    β    γ       │ AC  │      │
│ └─────┴────────────────────────────────────┴─────┴──────┘
       ↑                                      ↑
   Corte bajo                             Corte alto
   (elimina drift)                        (anti-aliasing)
```

### 3.3 Filtro Notch (50 Hz)

**Propósito:** Eliminar interferencia de la red eléctrica (50 Hz en Europa/Argentina)

**Tipo:** FIR notch
**Ancho de banda:** ±1 Hz (48-52 Hz atenuado)

```
Respuesta en Frecuencia:
           │
      1.0 ─┤─────────┐           ┌───────────
           │         │           │
           │         │    ▼      │
      0.5 ─┤         │   Notch   │
           │         │    50Hz   │
      0.0 ─┤─────────┴───────────┴───────────
           0   10   20   30   40  50   60   70   80  → Hz
```

### 3.4 Detrend Lineal

Antes de cada análisis por epoch, se aplica **detrend lineal** (`scipy.signal.detrend`):

```python
x_detrended = x - (a·t + b)
```

Donde `a` y `b` son los parámetros de la recta de mejor ajuste.

**Propósito:** Eliminar tendencias de baja frecuencia y offset DC residual.

---

## 4. Transformada de Fourier y Análisis Espectral

### 4.1 La Transformada de Fourier Discreta (DFT)

La DFT descompone una señal discreta en sus componentes frecuenciales:

```
           N-1
X[k] = Σ   x[n] · e^(-j·2π·k·n/N)
          n=0

donde:
- x[n] = señal en el tiempo (N muestras)
- X[k] = coeficientes de frecuencia (complejos)
- k = índice de frecuencia (0 a N-1)
- f_k = k · fs / N (frecuencia en Hz)
```

**Resolución frecuencial:** Δf = fs / N

Para epochs de 30s a 200 Hz:
- N = 6000 muestras
- Δf = 200/6000 = 0.033 Hz

### 4.2 FFT (Fast Fourier Transform)

La FFT es un algoritmo eficiente para calcular la DFT:

- **Complejidad DFT directa:** O(N²)
- **Complejidad FFT:** O(N log N)

Para N = 6000: ¡speedup de ~500x!

### 4.3 Densidad Espectral de Potencia (PSD)

La PSD estima cuánta "potencia" tiene la señal en cada frecuencia:

```
PSD[k] = |X[k]|² / N
```

#### Método Welch (usado en este proyecto)

El método Welch mejora la estimación del PSD reduciendo varianza:

```
┌───────────────────────────────────────────────────┐
│ Señal original (N = 6000 muestras, 30 segundos)   │
└─────────────────────────┬─────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
      ┌──────────┐  ┌──────────┐  ┌──────────┐
      │Segmento 1│  │Segmento 2│  │Segmento 3│  ... (con overlap 50%)
      │ (1024)   │  │ (1024)   │  │ (1024)   │
      └────┬─────┘  └────┬─────┘  └────┬─────┘
           │             │             │
           ▼             ▼             ▼
      ┌─────────┐   ┌─────────┐   ┌─────────┐
      │ Ventana │   │ Ventana │   │ Ventana │  (Hamming)
      │ Hamming │   │ Hamming │   │ Hamming │
      └────┬────┘   └────┬────┘   └────┬────┘
           │             │             │
           ▼             ▼             ▼
      ┌─────────┐   ┌─────────┐   ┌─────────┐
      │   FFT   │   │   FFT   │   │   FFT   │
      │  |X|²   │   │  |X|²   │   │  |X|²   │
      └────┬────┘   └────┬────┘   └────┬────┘
           │             │             │
           └──────────┬──┴──────────┬──┘
                      │ PROMEDIO    │
                      ▼             ▼
               ┌────────────────────────┐
               │   PSD Welch Estimado   │
               │   (varianza reducida)  │
               └────────────────────────┘
```

**Parámetros usados:**
- `nperseg = min(1024, len(data))` — longitud de segmento
- `overlap = 50%` (default de scipy)
- Ventana: Hamming

**Implementación:**
```python
from scipy.signal import welch
freqs, psd = welch(data, fs=200, nperseg=1024)
```

### 4.4 Ventanas Espectrales: ¿Por qué Hamming?

Cuando segmentamos la señal para aplicar FFT, los bordes del segmento crean **discontinuidades artificiales** que generan **spectral leakage** (fuga espectral).

**Problema sin ventana:**

```
Senal segmentada:   |~~~~~|   (la senal "cortada" en los bordes)
                        |
                        v
                    Discontinuidad en los bordes
                        |
                        v
                    FFT interpreta esto como
                    componentes de alta frecuencia
                    que NO estan en la senal real
```

**Solucion: Ventana de Hamming**

La ventana atenua suavemente los bordes a cero:

```
w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
```

```
Ventana Hamming:

       1.0  |      ____
            |     /    \
            |    /      \
       0.5  |   /        \
            |  /          \
       0.0  | /            \
            +----------------
              0     N/2     N
```

**Resultado:** La señal ventaneada tiene transiciones suaves, minimizando el leakage espectral.

| Ventana | Leakage | Resolución | Uso Típico |
|---------|---------|------------|------------|
| Rectangular | Alto | Máxima | Raro (mucho leakage) |
| Hamming | Bajo | Buena | **EEG (nuestro caso)** |
| Hann | Muy bajo | Buena | Audio, general |
| Blackman | Mínimo | Menor | Alta precisión espectral |

---

## 5. Bandas de Frecuencia

### 5.1 Definición de Bandas (Estándar AASM)

```python
FREQ_BANDS = {
    "delta": (0.5, 4),   # Ondas lentas, sueño profundo
    "theta": (4, 8),     # Transición, somnolencia
    "alpha": (8, 13),    # Vigilia relajada, ojos cerrados
    "sigma": (12, 15),   # Sleep spindles (N2)
    "beta":  (13, 30),   # Vigilia activa, REM
    "gamma": (30, 45),   # Procesamiento cognitivo
}
```

### 5.2 Cálculo de Potencia por Banda

**Potencia Absoluta:**
```
P_banda = ∫[f_low to f_high] PSD(f) df
```

**Potencia Relativa:**
```
P_rel_banda = P_banda / P_total

donde P_total = ∫[0.5 to 45] PSD(f) df
```

La suma de potencias relativas ≈ 1.0 (verificación de cordura).

---

## 6. Features Extraídas (133 total)

### 6.1 Features Espectrales (por canal)

Para cada canal (EEG Fpz-Cz, EEG Pz-Oz, EOG, EMG):

| Feature | Descripción | Fórmula/Método |
|---------|-------------|----------------|
| `{ch}_abs_delta` | Potencia absoluta delta | ∫PSD(0.5-4Hz) |
| `{ch}_abs_theta` | Potencia absoluta theta | ∫PSD(4-8Hz) |
| `{ch}_abs_alpha` | Potencia absoluta alpha | ∫PSD(8-13Hz) |
| `{ch}_abs_sigma` | Potencia absoluta sigma | ∫PSD(12-15Hz) |
| `{ch}_abs_beta` | Potencia absoluta beta | ∫PSD(13-30Hz) |
| `{ch}_abs_gamma` | Potencia absoluta gamma | ∫PSD(30-45Hz) |
| `{ch}_rel_delta` | Potencia relativa delta | P_delta / P_total |
| `{ch}_rel_theta` | Potencia relativa theta | P_theta / P_total |
| `{ch}_rel_alpha` | Potencia relativa alpha | P_alpha / P_total |
| `{ch}_rel_sigma` | Potencia relativa sigma | P_sigma / P_total |
| `{ch}_rel_beta` | Potencia relativa beta | P_beta / P_total |
| `{ch}_rel_gamma` | Potencia relativa gamma | P_gamma / P_total |
| `{ch}_theta_delta_ratio` | Ratio θ/δ | P_theta / P_delta |
| `{ch}_alpha_delta_ratio` | Ratio α/δ | P_alpha / P_delta |
| `{ch}_sigma_delta_ratio` | Ratio σ/δ | P_sigma / P_delta |
| `{ch}_dominant_freq` | Frecuencia dominante (Hz) | argmax(PSD) |

**Total por canal:** 16 features espectrales × 4 canales = 64 features espectrales

### 6.2 Features Temporales (por canal)

| Feature | Descripción | Fórmula |
|---------|-------------|---------|
| `{ch}_mean` | Media | μ = (1/N) Σ x[n] |
| `{ch}_std` | Desviación estándar | σ = √Var(x) |
| `{ch}_var` | Varianza | Var(x) = E[(x - μ)²] |
| `{ch}_min` | Valor mínimo | min(x) |
| `{ch}_max` | Valor máximo | max(x) |
| `{ch}_range` | Rango (peak-to-peak) | max(x) - min(x) |
| `{ch}_zcr` | Zero crossing rate | Σ|sign(x[n]) ≠ sign(x[n-1])| / N |

### 6.3 Parámetros de Hjorth

Los parámetros de Hjorth caracterizan la señal en el dominio del tiempo:

| Parámetro | Descripción | Fórmula |
|-----------|-------------|---------|
| `{ch}_hjorth_activity` | Varianza de la señal | Var(x) |
| `{ch}_hjorth_mobility` | Frecuencia media | √(Var(x') / Var(x)) |
| `{ch}_hjorth_complexity` | Cambio de frecuencia | Mobility(x') / Mobility(x) |

Donde x' = dx/dt (primera derivada).

**Interpretación intuitiva de Hjorth:**

```
+-----------------------------------------------------------+
|                   PARAMETROS DE HJORTH                    |
+-----------------------------------------------------------+
| ACTIVITY (Actividad)                                      |
|   = Var(x) = "cuanto se mueve la senal"                   |
|   Alto en N3 (ondas lentas de gran amplitud)              |
|   Bajo en REM/Vigilia                                     |
+-----------------------------------------------------------+
| MOBILITY (Movilidad)                                      |
|   = sqrt(Var(x') / Var(x))                                |
|   ~ frecuencia media de la senal                          |
|   Alta en Vigilia (frecuencias rapidas)                   |
|   Baja en N3 (frecuencias lentas ~1 Hz)                   |
+-----------------------------------------------------------+
| COMPLEXITY (Complejidad)                                  |
|   = Mobility(x') / Mobility(x)                            |
|   ~ "cuan compleja/irregular es la forma de onda"         |
|   Baja si es sinusoidal pura (~1.0)                       |
|   Alta si tiene muchas frecuencias mezcladas              |
+-----------------------------------------------------------+
```

**Ventaja de Hjorth:** Calculan “frecuencia” sin usar FFT — son puramente temporales y computacionalmente eficientes.

### 6.4 Entropías

| Feature | Descripción | Método |
|---------|-------------|--------|
| `{ch}_entropy` | Entropía de Shannon | H = -Σ p(x) log₂ p(x) |
| `{ch}_spectral_entropy` | Entropía espectral | -Σ PSD_norm · log₂(PSD_norm) |

**Teoría de la Información: Entropía**

La entropía mide la **incertidumbre** o **cantidad de información** en una distribución:

```
Entropía alta   = Distribución uniforme, impredecible
Entropía baja   = Distribución concentrada, predecible
```

**Entropía de Shannon (temporal):**

```
H = -Σ p(x) · log₂ p(x)

donde p(x) = probabilidad de cada valor de amplitud
             (calculada con histograma de 100 bins)
```

- Alta en señal ruidosa/caótica
- Baja en señal repetitiva/ordenada

**Entropía Espectral:**

Mide cuán "plana" es la distribución de potencia en frecuencia:

```
+----------------------------------------------------+
|       ENTROPIA ESPECTRAL                           |
+----------------------------------------------------+
|   ALTA                      BAJA                   |
|                                                    |
|   PSD: --------             PSD:  |                |
|        (plano)                    ||               |
|                                   |||   (picos)    |
|                                                    |
|   Ejemplo: Ruido blanco     Ejemplo: Spindle puro  |
|            REM, Vigilia              N2, N3        |
+----------------------------------------------------+
```

### 6.5 Features de Spindles (solo EEG)

Detectados con YASA (`spindles_detect`):

| Feature | Descripción |
|---------|-------------|
| `{ch}_spindle_count` | Número de spindles detectados |
| `{ch}_spindle_density` | Spindles por minuto |
| `{ch}_spindle_mean_duration` | Duración media (s) |
| `{ch}_spindle_mean_amplitude` | Amplitud media (µV) |

**Parámetros de detección (AASM):**
- Banda: 12-15 Hz (sigma)
- Duración: 0.5-2.0 s
- Umbrales YASA: `rel_pow=0.2`, `corr=0.65`, `rms=1.5`

### 6.6 Features de Ondas Lentas (solo EEG)

| Feature | Descripción |
|---------|-------------|
| `{ch}_slow_wave_power` | Potencia en banda delta |
| `{ch}_slow_wave_ratio` | Ratio delta/total |
| `{ch}_slow_wave_peak_amplitude` | Amplitud pico-a-pico en delta |

**Método:** Filtro Butterworth de orden 4 en banda delta (0.5-4 Hz), luego cálculo de métricas.

### 6.7 Features Cross-Channel

| Feature | Descripción | Relevancia |
|---------|-------------|------------|
| `eeg_eeg_correlation` | Correlación entre EEG Fpz-Cz y Pz-Oz | Sincronía cortical |
| `eeg_eog_correlation` | Correlación EEG-EOG | Artefactos oculares |
| `emg_eeg_ratio` | Ratio potencia EMG/EEG | REM vs Vigilia |
| `eog_emg_ratio` | Ratio potencia EOG/EMG | Detección REM |
| `eeg_eog_theta_coherence` | Coherencia en theta (4-8 Hz) | Modulación |
| `eeg_eog_delta_coherence` | Coherencia en delta (0.5-4 Hz) | N3 |
| `eeg_eog_sigma_coherence` | Coherencia en sigma (12-15 Hz) | N2 |

**Coherencia:** Mide sincronización entre dos señales por banda de frecuencia.

```
Coh(f) = |S_xy(f)|² / (S_xx(f) · S_yy(f))
```

Calculada con `scipy.signal.coherence(nperseg=256)`.

### 6.8 Interpretación Fisiológica de Features Clave

| Feature | Estadio donde es ALTO | Razón Fisiológica |
|---------|----------------------|-------------------|
| `rel_delta` | **N3** | Ondas lentas de alta amplitud (sincronización talamocortical) |
| `rel_theta` | **N1** | Transición al sueño, somnolencia |
| `rel_alpha` | **W** (ojos cerrados) | Vigilia relajada, ritmo α occipital |
| `rel_sigma` | **N2** | Sleep spindles (ráfagas 12-15 Hz del tálamo) |
| `rel_beta` | **W**, **REM** | Actividad cognitiva, sueño paradójico |
| `spindle_density` | **N2** | Marcador clásico de N2 según AASM |
| `emg_eeg_ratio` | **W** (alto), **REM** (bajo) | Atonía muscular en REM |
| `eog_emg_ratio` | **REM** | Movimientos oculares rápidos + atonía |
| `slow_wave_power` | **N3** | Potencia delta absoluta, sueño profundo |
| `hjorth_mobility` | **W**, **REM** | Frecuencias rápidas = alta movilidad |

**Clave para clasificación automática:**

```
N3:  Alto delta + Baja movilidad + Alta activity
N2:  Spindles + Sigma elevado + K-complexes
N1:  Theta dominante + Alpha decayendo
W:   Alpha/Beta alto + EMG alto
REM: Beta/Theta + EMG MUY bajo + EOG alto
```

---

## 7. Resumen de Canales y Features

| Canal | Tipo | Features Extraídas | Total |
|-------|------|-------------------|-------|
| EEG Fpz-Cz | EEG | Espectrales (16) + Temporales (7) + Hjorth (3) + Entropías (2) + Spindles (4) + Slow Waves (3) | 35 |
| EEG Pz-Oz | EEG | Espectrales (16) + Temporales (7) + Hjorth (3) + Entropías (2) + Spindles (4) + Slow Waves (3) | 35 |
| EOG horizontal | EOG | Espectrales (16) + Temporales (7) + Hjorth (3) + Entropías (2) | 28 |
| EMG submental | EMG | Espectrales (16) + Temporales (7) + Hjorth (3) + Entropías (2) | 28 |
| Cross-channel | — | Correlaciones (2) + Ratios (2) + Coherencias (3) | 7 |
| **Total** | | | **133** |

---

## 8. Diagrama Completo del Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DATOS CRUDOS (Sleep-EDF)                       │
│                      EDF @ 100 Hz, ~20 horas                          │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     PREPROCESAMIENTO (preprocessing.py)               │
│  ┌─────────────┐  ┌───────────────┐  ┌────────────┐  ┌────────────┐  │
│  │ Corrección  │→ │ Resampling    │→ │ Band-Pass  │→ │   Notch    │  │
│  │ Tipos Canal │  │ 100→200 Hz    │  │ 0.3-45 Hz  │  │   50 Hz    │  │
│  │             │  │               │  │ FIR/firwin │  │ FIR/firwin │  │
│  └─────────────┘  └───────────────┘  └────────────┘  └────────────┘  │
│                                                              │        │
│                                         ┌────────────────────┘        │
│                                         ▼                             │
│                                  ┌────────────┐                       │
│                                  │ Avg Ref    │                       │
│                                  │ (solo EEG) │                       │
│                                  └────────────┘                       │
│                                         │                             │
│                      ┌──────────────────┴──────────────────┐          │
│                      ▼                                     ▼          │
│               ┌────────────┐                       ┌────────────┐     │
│               │  Trimming  │                       │ Hipnograma │     │
│               │ ±15 min    │                       │    CSV     │     │
│               └────────────┘                       └────────────┘     │
│                      │                                     │          │
│                      └──────────────┬──────────────────────┘          │
│                                     │                                 │
│                                     ▼                                 │
│                              ┌────────────┐                           │
│                              │   .FIF     │                           │
│                              │  Archivo   │                           │
│                              └────────────┘                           │
└─────────────────────────────────────┬────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    EXTRACCIÓN DE FEATURES (features.py)               │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │                      Por cada Epoch (30s)                     │    │
│  │  ┌────────────────────────────────────────────────────────┐  │    │
│  │  │ 1. Detrend lineal (eliminar drift)                     │  │    │
│  │  └────────────────────────────────────────────────────────┘  │    │
│  │                              │                                │    │
│  │                              ▼                                │    │
│  │  ┌────────────────────────────────────────────────────────┐  │    │
│  │  │ 2. Welch PSD (nperseg=1024, overlap=50%)               │  │    │
│  │  │    - FFT por segmentos                                 │  │    │
│  │  │    - Ventana Hamming                                   │  │    │
│  │  │    - Promedio de periodogramas                         │  │    │
│  │  └────────────────────────────────────────────────────────┘  │    │
│  │                              │                                │    │
│  │            ┌─────────────────┼─────────────────┐              │    │
│  │            ▼                 ▼                 ▼              │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐     │    │
│  │  │   Features    │  │   Features    │  │   Features    │     │    │
│  │  │  Espectrales  │  │  Temporales   │  │   Spindles    │     │    │
│  │  │  (YASA/FFT)   │  │   (Hjorth)    │  │    (YASA)     │     │    │
│  │  └───────────────┘  └───────────────┘  └───────────────┘     │    │
│  │            │                 │                 │              │    │
│  │            └─────────────────┼─────────────────┘              │    │
│  │                              ▼                                │    │
│  │  ┌────────────────────────────────────────────────────────┐  │    │
│  │  │            Features Cross-Channel                      │  │    │
│  │  │  - Correlación (Pearson)                               │  │    │
│  │  │  - Coherencia (scipy.signal.coherence)                 │  │    │
│  │  └────────────────────────────────────────────────────────┘  │    │
│  └───────────────────────────────────────────────────────────────┘   │
│                                      │                                │
│                                      ▼                                │
│                        ┌──────────────────────────┐                   │
│                        │  DataFrame por Sesión   │                   │
│                        │  (epoch × 139 features) │                   │
│                        └──────────────────────────┘                   │
└─────────────────────────────────────┬────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          SALIDA FINAL                                 │
│                  features_resamp200.parquet (176 MB)                  │
│                     186,499 epochs × 133 features                     │
│                     78 sujetos (153 sesiones)                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 9. Justificación de Decisiones de Diseño

### 9.1 ¿Por qué FIR y no IIR?

| Criterio | FIR | IIR |
|----------|-----|-----|
| **Fase lineal** | ✓ Sí | ✗ No (distorsión de fase) |
| **Estabilidad** | ✓ Siempre estable | Puede ser inestable |
| **Orden requerido** | Mayor (más lento) | Menor (más rápido) |
| **Uso en EEG** | ✓ Preferido | Evitado |

Para EEG, la **fase lineal** es crítica para no distorsionar la forma de onda (spindles, K-complexes).

### 9.2 ¿Por qué Welch y no periodograma simple?

El método Welch reduce la **varianza** de la estimación del PSD a costa de resolución frecuencial. Para epochs de 30s (6000 muestras a 200 Hz), la resolución con `nperseg=1024` es:

```
Δf = 200 Hz / 1024 ≈ 0.2 Hz
```

Suficiente para las bandas de interés (más anchas que 0.2 Hz).

### 9.3 ¿Por qué YASA para spindles?

- Validado contra scoring manual (dataset MASS)
- Parámetros basados en criterios AASM
- Referenciado en literatura (Vallat & Walker, 2021)
- Implementación eficiente y mantenida

---

## 10. Referencias

1. **AASM (2007)**. The AASM Manual for the Scoring of Sleep and Associated Events.
2. **Berry et al. (2017)**. AASM Scoring Manual Updates.
3. **Warby et al. (2014)**. Sleep spindle measurements. *Sleep*, 37(9).
4. **Purcell et al. (2017)**. Characterizing sleep spindles. *Sleep*, 40(1).
5. **Vallat & Walker (2021)**. YASA: An open-source, high-performance tool for automated sleep staging. *eLife*, 10:e70092.
6. **Hjorth (1970)**. EEG analysis based on time domain properties. *Electroencephalography and Clinical Neurophysiology*, 29(3).
7. **Welch (1967)**. The use of fast Fourier transform for the estimation of power spectra. *IEEE Transactions on Audio and Electroacoustics*.

---

## 11. Comando de Extracción Utilizado

```bash
python -m src.extract_features \
  --manifest data/processed/manifest_trimmed_resamp200.csv \
  --output data/processed/features_resamp200.parquet \
  --no-prefilter \
  --epoch-length 30 \
  --psd-method welch \
  --movement-policy drop
```

**Nota:** `--no-prefilter` porque los datos ya fueron filtrados en el preprocesamiento. Si se usara `--prefilter`, se aplicaría un segundo filtrado innecesario.

---

## 12. Estadísticas Finales

| Métrica | Valor |
|---------|-------|
| Total epochs | 186,499 |
| Total sujetos | 78 |
| Total sesiones/noches | 153 |
| Total features | 133 |
| Frecuencia de muestreo | 200 Hz |
| Duración epoch | 30 s |
| Archivo de salida | `features_resamp200.parquet` |
| Tamaño archivo | 176 MB |
| Valores NaN | 0.00% |

---

## 13. Resultados del Modelado

Los features extraídos fueron utilizados para entrenar modelos de **Machine Learning clásico**:

| Modelo | Cohen's Kappa | F1 Macro | Accuracy | Uso |
|--------|---------------|----------|----------|-----|
| **XGBoost LOSO** | **0.641** | 70.02% | 73.08% | ML interpretable (elegido) |
| Random Forest | 0.635 | 69.50% | 72.82% | Baseline ML |

Adicionalmente, se entrenaron modelos de **Deep Learning** directamente sobre la señal cruda (sin usar estas features):

| Modelo | Cohen's Kappa | F1 Macro | Accuracy | Uso |
|--------|---------------|----------|----------|-----|
| **CNN1D** | **0.680** | 70.83% | 76.86% | Mejor rendimiento offline |
| LSTM Bi + Attention | 0.651 | 68.07% | 74.64% | DL secuencial |
| LSTM Unidireccional | 0.530 | 58.59% | 66.17% | Inferencia real-time |

> **Ver análisis completo:** [docs/reports/COMPARATIVE_ANALYSIS.md](../../docs/reports/COMPARATIVE_ANALYSIS.md)
