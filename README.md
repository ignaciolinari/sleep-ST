# Clasificación de fases de sueño

## Objetivo
Clasificar estadios de sueño a partir de EEG, EOG y EMG usando modelos de Machine Learning y Deep Learning.

## Estructura
- `data/`: contiene los datos (no versionados en GitHub)
   - `raw/`: espejo de PhysioNet (p. ej. `physionet.org/files/sleep-edfx/1.0.0/...`)
   - `processed/`: salidas procesadas
      - `manifest.csv`: inventario PSG/Hypnogramas
      - `manifest_trimmed.csv`: inventario de sesiones recortadas
      - `sleep_trimmed/`: artefactos del recorte
         - `psg/`: PSG recortados en formato `.fif`
         - `hypnograms/`: anotaciones recortadas en `.csv`
- `notebooks/`: exploración y análisis
   - `01_raw_exploration.ipynb`
   - `02_trimmed_exploration.ipynb`
- `src/`: código fuente
   - `check_data.py`: automatiza QA (hashes, metadata y artefactos procesados)
   - `download.py`: descarga Sleep-EDF (wget / wfdb) y dispara QA post-descarga
   - `manifest.py`: genera `data/processed/manifest.csv`
   - `preprocessing.py`: recorte alrededor del periodo de sueño (genera `.fif` y CSV)
   - `view_subject.py`: visualización rápida de señales
   - `features.py`: extracción de características
   - `models.py`: modelos ML/DL
   - `__init__.py`
- `out/`: salidas generadas (gráficos/exports) por scripts de ejemplo (no versionados en GitHub)
- `tmp/`: archivos temporales y reportes de QA (p. ej. hashes calculados) (no versionados en GitHub)
- `environment.yml`: entorno base
- `README.md`, `LICENSE`

## Quickstart
1) Crear entorno
```bash
conda env create -f environment.yml
conda activate sleep-st
```
2) Descargar datos (ej. subset sleep-cassette)
```bash
python src/download.py --method wget --subset sleep-cassette --out data/raw --clean
```
3) Generar manifest
```bash
python src/manifest.py --version 1.0.0 --subset sleep-cassette --raw-root data/raw --out data/processed/manifest.csv
```
4) (Opcional) Recortar sesiones alrededor del sueño
```bash
python src/preprocessing.py \
  --manifest data/processed/manifest.csv \
  --out-root data/processed/sleep_trimmed \
  --out-manifest data/processed/manifest_trimmed.csv
# Padding por defecto: 900 s (15 min) antes y después del SPT-window.
```
5) Explorar en notebooks `notebooks/01_raw_exploration.ipynb` y `02_trimmed_exploration.ipynb`

## Entorno
Entorno base:
```bash
conda env create -f environment.yml
conda activate sleep-st
```

## Descargar dataset
Crear cuenta en [PhysioNet](https://physionet.org) y aceptar los términos del dataset Sleep-EDF Expanded (https://physionet.org/content/sleep-edfx/1.0.0/)

Tenes dos opciones: usando el script incluido o con `wget` directamente.

Opción A) Script Python
```bash
# Activar entorno
conda activate sleep-st

# Dry-run (muestra lo que hará)
python src/download.py --method wget --subset sleep-cassette --out data/raw --dry-run

# Descargar subset "sleep-cassette" con wget
python src/download.py --method wget --subset sleep-cassette --out data/raw --clean

# O descargar base completa con wfdb (sin filtrar subset)
python src/download.py --method wfdb --out data/raw --clean
```

Notas:
- Si PhysioNet pide login, exporta variables de entorno o usa flags:
   - `export PHYSIONET_USERNAME=tu_usuario`
   - `export PHYSIONET_PASSWORD=tu_password` (o deja que el script lo pregunte)
- Si la validación falla por archivos faltantes o hashes incorrectos, corré `python src/download.py ... --clean` para empezar de cero y volvé a ejecutar la descarga; luego revalidá con `python -m src.check_data`.

### Limpieza y validación automática

El script de descarga ahora puede preparar el entorno antes de bajar archivos y ejecutar checks de calidad al finalizar:

- `--clean`: borra el espejo previo de `data/raw`, los artefactos procesados (`data/processed`) y los `wget-log*` asociados para evitar mezclar descargas viejas con la corrientes. Úsalo cuando quieras partir de cero.
- `--processed-root`: ajusta la carpeta donde se esperan salidas procesadas (por defecto `data/processed`). Sirve si trabajás con otra ruta en equipos compartidos.
- `--skip-validation`: salta los chequeos automáticos post-descarga. Por defecto se ejecutan para confirmar hashes y que existan los artefactos principales.
- `--qa-report`: guarda el reporte de hashes calculados en la ruta que indiques (ej. `--qa-report tmp/raw_sha256_report.csv`).
- `--strict-validation`: marca como error cualquier hallazgo del QA (por defecto solo se registran las advertencias).

Los checks post-descarga reutilizan el módulo `src.check_data`. Si ocurre un corte en la conexión y faltan archivos, la validación lo reportará de inmediato.

### QA manual de datos

También podés ejecutar la verificación de manera independiente (por ejemplo, luego de reanudar una descarga manual):

```bash
python -m src.check_data \
   --raw-root data/raw \
   --processed-root data/processed \
   --version 1.0.0 \
   --subset sleep-cassette \
   --report tmp/raw_sha256_report.csv
```

Qué hace:

- Convierte la planilla `SC-subjects.xls` incluida en PhysioNet para confirmar que todos los sujetos están presentes.
- Calcula hashes SHA-256 de cada EDF y los compara con `SHA256SUMS.txt` (crea el CSV indicado en `--report`).
- Valida que existan los artefactos procesados claves (`manifest.csv`, recortes si corresponde).

Flags útiles:

- `--strict`: convierte cualquier advertencia en error (salida con código distinto de cero).
- `--qa-report`: opcional; permite guardar el informe de hashes en un sitio distinto o desactivarlo si no se especifica.

Interpretación:

- Salida `OK`: todos los checks pasaron y el código termina con `0`.
- Advertencias (`WARNING`): archivos faltantes o metadatos que se pueden regenerar; revisá según el mensaje.
- Errores (`ERROR`): hashes que no coinciden, archivos críticos faltantes o problemas al leer EDF. Rehacé la descarga con `--clean` y reintentá.

Opción B) `wget` directo *(requiere checks manuales y es proclive a errores)*
```bash
wget -r -N -c -np -e robots=off -P data/raw https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/
```

## Convención de nombres en Sleep-EDF
Los archivos de *Sleep Cassette* siguen un esquema estable que conviene conocer al trabajar con el `manifest` o al descargar manualmente desde PhysioNet:

- **PSG**: `SC4ssNL0-PSG.edf`
   - `SC` → Sleep Cassette.
   - `4` → constante del estudio.
   - `ss` → número de sujeto (dos dígitos, p. ej. `00` = primer sujeto).
   - `N` → noche del registro (`1` o `2`).
   - `L` → letra que aparece en el nombre del PSG: puede ser `E`, `F` o `G` según el lote/serie del estudio.
   - `0` → dígito cero fijo para PSG.
- **Hipnogramas**: `SC4ssNLX-Hypnogram.edf`
   - El prefijo `SC4ssNL` coincide exactamente con el PSG asociado (mismo sujeto y misma noche).
   - `X` → letra del técnico que puntuó el hipnograma (p. ej. `C`, `M`, `P`, `V`, `Y`, `J`, `A`, `W`, etc.).


## Manifest y validación de sesiones
Luego de descargar, podés generar un manifiesto con las sesiones disponibles y el estado de cada par PSG + Hypnograma.

Script: `src/manifest.py`

Uso:
```bash
# Generar manifest para sleep-cassette
python src/manifest.py --version 1.0.0 --subset sleep-cassette --raw-root data/raw --out data/processed/manifest.csv

# También soporta sleep-telemetry
python src/manifest.py --version 1.0.0 --subset sleep-telemetry --raw-root data/raw --out data/processed/manifest.csv
```

Salida:
- Crea `data/processed/manifest.csv` con columnas: `subject_id, subset, version, psg_path, hypnogram_path, status`.
- Muestra un resumen por consola (sesiones totales, pares completos y faltantes).

Notas:
- El emparejamiento PSG/Hypnograma se realiza por prefijo canónico (primeros 7 caracteres, por ejemplo `SC4001E`).
- Si agregás más archivos después, re-ejecutá el script para actualizar el CSV.

## Preprocesamiento: recorte alrededor del sueño
Para acelerar análisis, se puede recortar cada sesión al periodo de sueño, conservando un margen configurable de vigilia antes y después. Esto genera PSG en `.fif` y anotaciones en `.csv` bajo `data/processed/sleep_trimmed/` y un `manifest_trimmed.csv` con metadatos.

Script: `src/preprocessing.py`

Uso típico:
```bash
python src/preprocessing.py \
   --manifest data/processed/manifest.csv \
   --out-root data/processed/sleep_trimmed \
   --out-manifest data/processed/manifest_trimmed.csv \
   --pre-padding 900 --post-padding 900

# Limitar a N sesiones para pruebas
python src/preprocessing.py --limit 5

# Reescribir si ya existen salidas
python src/preprocessing.py --overwrite

# Segmentar episodios de sueño fragmentado
python src/preprocessing.py \
   --episode-strategy all \
   --wake-gap-min 60 \
   --min-episode-min 20
```

Notas:
- El recorte detecta el *Sleep Period Time* (SPT): desde el primer epoch etiquetado como sueño hasta el último, lo recorta y añade un margen configurable (`--pre/post-padding`, por defecto 900 s = 15 min) de vigilia antes y después.
- Las estrategias `--episode-strategy` (`spt`, `longest`, `all`) permiten dividir noches con vigilia prolongada interna. Con `all` cada bloque de sueño separado por al menos `--wake-gap-min` minutos de vigilia se exporta como un episodio distinto, opcionalmente descartando los muy cortos (`--min-episode-min`).
- La visualización con `view_subject.py` lee EDF crudos; los `.fif` recortados no se visualizan con ese script por ahora.

### Elegir estrategia de episodios (sin Lights On/Off)

Sleep-EDF no incluye anotaciones estandarizadas de "Lights Off/On" en los hipnogramas, por lo que usar SPT (desde el primer al último epoch de sueño) puede incluir siestas o segmentos muy alejados del periodo nocturno principal.

Recomendaciones:

- `spt`: usa todo el intervalo entre el primer y el último episodio de sueño. Útil para noches sin siestas largas; puede fallar si hay siestas tempranas.
- `longest`: detecta episodios de sueño separados por vigilia de al menos `--wake-gap-min` minutos y conserva el episodio con mayor tiempo de sueño acumulado. Es la opción recomendada cuando no hay Lights On/Off.
  - Elegí `--wake-gap-min` según tolerancia a despertares: 60 min une cortes breves; 90–120 min une despertares largos.
  - Usá `--min-episode-min` (p.ej., 120) para descartar siestas cortas.
- `all`: exporta todos los episodios detectados para inspección manual.

Ejemplo recomendado (evita siestas y conserva el bloque principal):

```bash
python src/preprocessing.py \
  --manifest data/processed/manifest.csv \
  --out-root data/processed/sleep_trimmed \
  --out-manifest data/processed/manifest_trimmed.csv \
  --pre-padding 900 --post-padding 900 \
  --episode-strategy longest --wake-gap-min 60 --min-episode-min 120 \
  --overwrite
```

Notas prácticas:

- Si hay un despertar largo en medio de la noche y supera `--wake-gap-min`, la noche se parte en dos episodios y se elige el que acumula más sueño. Si preferís mantenerlo como uno, aumentá `--wake-gap-min`.
- Si una siesta supera `--min-episode-min` y se vuelve candidata, subí ese umbral o inspeccioná con `--episode-strategy all`.

**Receta: separar episodios (exportar todos)**

```bash
python src/preprocessing.py \
  --manifest data/processed/manifest.csv \
  --out-root data/processed/sleep_trimmed \
  --out-manifest data/processed/manifest_trimmed.csv \
  --pre-padding 900 --post-padding 900 \
  --episode-strategy all --wake-gap-min 60 --min-episode-min 60 \
  --overwrite
```

Salida esperada:
- Varios archivos por noche con sufijos `_e{i}of{n}` en `sleep_trimmed/psg` y `sleep_trimmed/hypnograms` (uno por episodio).
- `manifest_trimmed.csv` incluye una fila por episodio con `episode_index` y `episodes_total`.

## Visualización rápida de señales
Para explorar las señales de un sujeto puntual, usá `src/view_subject.py`. Requiere haber generado el `manifest.csv` previamente.

```bash
# Listar canales disponibles para el sujeto SC4001E
python src/view_subject.py --subject-id SC4001E --list-channels

# Graficar 2 minutos de EEG/EOG/EMG y guardar resultado
python src/view_subject.py --subject-id SC4001E --duration 120 --save out/SC4001E.png

# Opcional: habilitar hipnograma, seleccionar canales y resamplear a 100 Hz
python src/view_subject.py --subject-id SC4001E \
   --channels "EEG Fpz-Cz,EEG Pz-Oz,EOG horizontal" \
   --resample 100 --with-hypnogram
```

El script también acepta rutas directas a archivos `.edf` (`--psg-path` / `--hypnogram-path`), filtros por subset/versión dentro del manifest y parámetros de ventana (`--start`, `--duration`).


## Estado de features y modelos
- `src/features.py` y `src/models.py` están en construcción. La extracción de features clásicas (espectrales/tiempo-frecuencia) y pipelines de entrenamiento se documentarán aquí cuando estén listos.

## Calidad de código (pre-commit)
Para mantener formato y checks automáticos antes de cada commit se usa pre-commit con Ruff/Black y validaciones básicas.

Setup inicial (una vez por repo):

```bash
/opt/miniconda3/envs/sleep-st/bin/python -m pre_commit install
```

Ejecutar manualmente sobre todos los archivos:

```bash
/opt/miniconda3/envs/sleep-st/bin/python -m pre_commit run --all-files
```

Si un hook modifica archivos, el commit se cancela. Volvé a hacer `git add` y `git commit`. Para saltar los hooks en un commit puntual:

```bash
git commit -m "mensaje" --no-verify
```
