# Datos procesados (Sleep-ST)

Esta carpeta contiene derivados del dataset Sleep-EDF tras distintas etapas de preparación. Se mantienen varias versiones en paralelo para comparar impacto de filtrado y re-muestreo.

## Conjuntos

- `manifest.csv` (base):
  - Inventario de PSG/Hypnogram crudos descargados.
  - Columnas: `subject_id, subset, version, psg_path, hypnogram_path, status`.
  - Se genera con `python src/manifest.py --raw-root data/raw --version 1.0.0 --subset sleep-cassette --out data/processed/manifest.csv`.

- `sleep_trimmed/` + `manifest_trimmed.csv`:
  - Recorte alrededor del sueño (SPT) con padding 15 min, sin filtrado adicional.
  - PSG en `.fif`, hipnogramas en `.csv`.
  - Se genera con `python src/preprocessing.py --manifest data/processed/manifest.csv --out-root data/processed/sleep_trimmed --out-manifest data/processed/manifest_trimmed.csv --pre-padding 900 --post-padding 900`.

- `sleep_trimmed_filt/` + `manifest_trimmed_filt.csv`:
  - Recorte SPT con padding 15 min.
  - Filtrado band-pass 0.3–45 Hz, sin re-muestreo (fs original 100 Hz), referencia promedio EEG, notch omitido por Nyquist=50.
  - Útil para ver impacto de filtrado suave sin cambiar fs.
  - Generado con `python src/preprocessing.py --manifest data/processed/manifest.csv --out-root data/processed/sleep_trimmed_filt --out-manifest data/processed/manifest_trimmed_filt.csv --pre-padding 900 --post-padding 900 --filter-lowcut 0.3 --filter-highcut 45 --avg-ref`.

- `sleep_trimmed_resamp200/` + `manifest_trimmed_resamp200.csv`:
  - Recorte SPT con padding 15 min.
  - Re-muestreo a 200 Hz, band-pass 0.3–45 Hz, notch 50 Hz, referencia promedio EEG.
  - Recomendado para entrenamiento/feature extraction por limpieza de línea y mejor resolución espectral.
  - Generado con `python src/preprocessing.py --manifest data/processed/manifest.csv --out-root data/processed/sleep_trimmed_resamp200 --out-manifest data/processed/manifest_trimmed_resamp200.csv --pre-padding 900 --post-padding 900 --resample-sfreq 200 --filter-lowcut 0.3 --filter-highcut 45 --notch-freqs 50 --avg-ref`.

## Notas y buenas prácticas

- No se sobrescriben conjuntos previos: cada raíz (`sleep_trimmed*`) y su manifest correspondiente son independientes.
- El campo `notes` en los manifests recortados indica si hubo filtrado, notch, re-muestreo y referencia usada.
- Para comparaciones, extrae features por separado desde cada manifest (`manifest_trimmed*.csv`) y guarda sus outputs con nombres distintos (ej. `features_resamp200.parquet`).
- Si necesitas espacio, puedes eliminar versiones que no uses (carpetas `sleep_trimmed_*` y su manifest asociado), manteniendo siempre `manifest.csv` y los crudos.
