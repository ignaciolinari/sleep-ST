# Data Pipeline

Documentación del preprocesamiento de datos: desde los archivos EDF crudos hasta los datos listos para entrenar modelos.

## Flujo de datos

```
PhysioNet (EDF)
      ↓
   download.py
      ↓
data/raw/ (espejo PhysioNet)
      ↓
   manifest.py
      ↓
manifest.csv (inventario de sesiones)
      ↓
   preprocessing.py
      ↓
data/processed/sleep_trimmed/
├── psg/*.fif          (PSG recortados)
├── hypnograms/*.csv   (anotaciones recortadas)
└── manifest_trimmed.csv
      ↓
   extract_features.py
      ↓
features.parquet (~130 features por epoch)
```

## Preprocesamiento: Recorte alrededor del sueño

Las grabaciones originales incluyen horas de vigilia (ej: antes de dormir). El preprocesamiento recorta cada sesión al **Sleep Period Time (SPT)** con márgenes configurables.

### Uso básico

```bash
python src/preprocessing.py \
  --manifest data/processed/manifest.csv \
  --out-root data/processed/sleep_trimmed \
  --out-manifest data/processed/manifest_trimmed.csv \
  --pre-padding 900 --post-padding 900
```

> **Nota:** Usá `python src/preprocessing.py --help` para ver todas las opciones disponibles.

### Opciones principales

| Flag | Default | Descripción |
|------|---------|-------------|
| `--pre-padding` | 900 | Segundos de vigilia antes del sueño |
| `--post-padding` | 900 | Segundos de vigilia después del sueño |
| `--episode-strategy` | spt | Estrategia de segmentación (ver abajo) |
| `--wake-gap-min` | 60 | Minutos de vigilia para separar episodios |
| `--min-episode-min` | 20 | Duración mínima de episodio en minutos |
| `--limit` | - | Procesar solo N sesiones (para pruebas) |
| `--overwrite` | False | Reescribir archivos existentes |

### Opciones de filtrado

```bash
python src/preprocessing.py \
  --manifest data/processed/manifest.csv \
  --out-root data/processed/sleep_trimmed_filt \
  --out-manifest data/processed/manifest_trimmed_filt.csv \
  --filter-lowcut 0.3 --filter-highcut 45 \
  --notch-freqs 50 \
  --avg-ref
```

| Flag | Descripción |
|------|-------------|
| `--filter-lowcut` | Frecuencia de corte inferior (Hz) |
| `--filter-highcut` | Frecuencia de corte superior (Hz) |
| `--notch-freqs` | Frecuencias de notch (ej: 50 60) |
| `--resample-sfreq` | Re-muestrear a esta frecuencia |
| `--avg-ref` | Aplicar referencia promedio a EEG |

## Estrategias de episodios

Sleep-EDF no incluye anotaciones de "Lights Off/On", por lo que el SPT simple puede incluir siestas. Las estrategias permiten manejar esto:

### `spt` (default)

Usa todo el intervalo desde el primer al último epoch de sueño.

```
[====VIGILIA====][SUEÑO---------SUEÑO][====VIGILIA====]
                 ^                   ^
                SPT start          SPT end
```

**Cuándo usar**: Noches sin siestas ni despertares largos.

### `longest`

Detecta episodios separados por vigilia prolongada y conserva el más largo.

```
[SUEÑO--][===60min vigilia===][SUEÑO-----------]
   ↑                              ↑
 descartado                  conservado (más largo)
```

**Cuándo usar**: Cuando hay siestas que querés excluir.

```bash
python src/preprocessing.py \
  --episode-strategy longest \
  --wake-gap-min 60 \
  --min-episode-min 120
```

### `all`

Exporta todos los episodios detectados como archivos separados.

```
[SUEÑO--][===60min===][SUEÑO---][===60min===][SUEÑO----]
   ↑                      ↑                       ↑
_e1of3                 _e2of3                  _e3of3
```

**Cuándo usar**: Para inspección manual o análisis de fragmentación.

```bash
python src/preprocessing.py \
  --episode-strategy all \
  --wake-gap-min 60 \
  --min-episode-min 60
```

## Receta recomendada

Para evitar siestas y conservar el bloque principal de sueño:

```bash
python src/preprocessing.py \
  --manifest data/processed/manifest.csv \
  --out-root data/processed/sleep_trimmed \
  --out-manifest data/processed/manifest_trimmed.csv \
  --pre-padding 900 --post-padding 900 \
  --episode-strategy longest \
  --wake-gap-min 60 \
  --min-episode-min 120 \
  --overwrite
```

## Salida del preprocesamiento

```
data/processed/
├── manifest_trimmed.csv       # Inventario de sesiones procesadas
└── sleep_trimmed/
    ├── psg/                   # PSG recortados (.fif)
    │   ├── SC4001E_sleep-cassette_1.0.0_trimmed_raw.fif
    │   ├── SC4002E_sleep-cassette_1.0.0_trimmed_raw.fif
    │   └── ...
    └── hypnograms/            # Anotaciones recortadas (.csv)
        ├── SC4001E_sleep-cassette_1.0.0_trimmed_annotations.csv
        ├── SC4002E_sleep-cassette_1.0.0_trimmed_annotations.csv
        └── ...
```

### Columnas de manifest_trimmed.csv

| Columna | Descripción |
|---------|-------------|
| `subject_id` | ID del sujeto |
| `psg_trimmed_path` | Ruta al PSG recortado |
| `hypnogram_trimmed_path` | Ruta al hipnograma recortado |
| `trim_start_sec` | Segundo de inicio del recorte |
| `trim_end_sec` | Segundo de fin del recorte |
| `trim_duration_sec` | Duración del recorte |
| `sleep_duration_sec` | Duración total de sueño |
| `episode_index` | Índice del episodio (si strategy=all) |
| `episodes_total` | Total de episodios en la noche |

## Visualización rápida

Para explorar las señales de un sujeto:

```bash
# Listar canales disponibles
python src/view_subject.py --subject-id SC4001E --list-channels

# Graficar 2 minutos de señales
python src/view_subject.py --subject-id SC4001E --duration 120 --save out/SC4001E.png

# Con hipnograma y canales específicos
python src/view_subject.py --subject-id SC4001E \
  --channels "EEG Fpz-Cz,EEG Pz-Oz,EOG horizontal" \
  --resample 100 --with-hypnogram
```

> **Nota:** Usá `python src/view_subject.py --help` para ver todas las opciones.

> **Nota**: `view_subject.py` lee EDF crudos. Los `.fif` recortados requieren MNE directamente.

## Siguiente paso

Con los datos preprocesados, continuá con la [extracción de features](FEATURES.md) o directamente al [entrenamiento de modelos](MODELS.md).
