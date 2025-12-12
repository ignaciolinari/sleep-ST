# Getting Started

Guía completa para configurar el entorno, descargar datos y validar la instalación.

## Requisitos

- Python 3.10+
- Conda (recomendado) o pip
- ~60 GB de espacio para datos (20GB) (subset sleep-cassette) y su procesamiento (40GB)

## Instalación

### Opción A: Conda (recomendado)

```bash
conda env create -f environment.yml
conda activate sleep-st
```

> **Nota**: El archivo `environment.yml` incluye todas las dependencias necesarias, incluyendo TensorFlow para modelos de deep learning.

## Descargar el Dataset

El proyecto usa [Sleep-EDF Database Expanded](https://physionet.org/content/sleep-edfx/1.0.0/) de PhysioNet.

### Paso 1: Crear cuenta en PhysioNet

1. Ir a [physionet.org](https://physionet.org) y crear una cuenta
2. Aceptar los términos del dataset Sleep-EDF Expanded

### Paso 2: Descargar datos

**Opción A: Script Python (recomendado)**

```bash
# Activar entorno
conda activate sleep-st

# Dry-run (muestra lo que hará sin descargar)
python src/download.py --method wget --subset sleep-cassette --out data/raw --dry-run

# Descargar subset "sleep-cassette" (~153 sesiones)
python src/download.py --method wget --subset sleep-cassette --out data/raw --clean

# O descargar base completa con wfdb
python src/download.py --method wfdb --out data/raw --clean
```

> **Nota:** Usá `python src/download.py --help` para ver todas las opciones disponibles.

**Opción B: wget directo** *(requiere validación manual)*

```bash
wget -r -N -c -np -e robots=off -P data/raw \
  https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
```

### Autenticación

Si PhysioNet requiere login:

```bash
export PHYSIONET_USERNAME=tu_usuario
export PHYSIONET_PASSWORD=tu_password
```

O el script lo preguntará interactivamente.

## Flags del script de descarga

| Flag | Descripción |
|------|-------------|
| `--clean` | Borra descargas previas antes de empezar |
| `--processed-root` | Carpeta de salidas procesadas (default: `data/processed`) |
| `--skip-validation` | Salta validación post-descarga |
| `--qa-report PATH` | Guarda reporte de hashes en PATH |
| `--strict-validation` | Convierte warnings en errores |

## Validación de datos (QA)

El script de descarga ejecuta validación automática. También podés ejecutarla manualmente:

```bash
python -m src.check_data \
   --raw-root data/raw \
   --processed-root data/processed \
   --version 1.0.0 \
   --subset sleep-cassette \
   --report tmp/raw_sha256_report.csv
```

> **Nota:** Usá `python -m src.check_data --help` para ver todas las opciones.

### Qué valida

- Convierte `SC-subjects.xls` para confirmar sujetos presentes
- Calcula hashes SHA-256 y compara con `SHA256SUMS.txt`
- Verifica artefactos procesados (`manifest.csv`, recortes)

### Interpretación

| Resultado | Significado |
|-----------|-------------|
| `OK` | Todos los checks pasaron |
| `WARNING` | Archivos faltantes o regenerables |
| `ERROR` | Hashes incorrectos o archivos críticos faltantes |

> **Tip**: Si hay errores, ejecutá `python src/download.py ... --clean` y reintentá.

## Convención de nombres Sleep-EDF

Los archivos de *Sleep Cassette* siguen este esquema:

### PSG: `SC4ssNL0-PSG.edf`

| Parte | Significado |
|-------|-------------|
| `SC` | Sleep Cassette |
| `4` | Constante del estudio |
| `ss` | Número de sujeto (00-99) |
| `N` | Noche del registro (1 o 2) |
| `L` | Letra del lote (E, F, G) |
| `0` | Dígito fijo para PSG |

### Hipnogramas: `SC4ssNLX-Hypnogram.edf`

| Parte | Significado |
|-------|-------------|
| `SC4ssNL` | Mismo prefijo que el PSG |
| `X` | Letra del técnico anotador |

**Ejemplo**: `SC4001E0-PSG.edf` → Sujeto 00, noche 1, lote E.

## Generar Manifest

El manifest es un inventario de sesiones disponibles:

```bash
python src/manifest.py \
  --version 1.0.0 \
  --subset sleep-cassette \
  --raw-root data/raw \
  --out data/processed/manifest.csv
```

> **Nota:** Usá `python src/manifest.py --help` para ver todas las opciones.

### Salida

- Crea `manifest.csv` con columnas: `subject_id`, `subset`, `version`, `psg_path`, `hypnogram_path`, `status`
- Muestra resumen: sesiones totales, pares completos, faltantes

> **Nota**: El emparejamiento PSG/Hipnograma se realiza por prefijo canónico (primeros 7 caracteres).

## Siguiente paso

Una vez descargados y validados los datos, continuá con el [Data Pipeline](DATA_PIPELINE.md) para preprocesar las sesiones.
