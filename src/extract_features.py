"""Script para extraer y guardar features de todas las sesiones.

Este script extrae features una vez y las guarda en un archivo para uso posterior
en entrenamiento de modelos, evitando re-extraer features cada vez.

Uso:
    python -m src.extract_features \
        --manifest data/processed/manifest_trimmed.csv \
        --output data/processed/features.parquet \
        --format parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.features import extract_features_from_session

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_and_save_features(
    manifest_path: Path | str,
    output_path: Path | str,
    limit: int | None = None,
    epoch_length: float = 30.0,
    sfreq: float | None = None,
    format: str = "parquet",
) -> None:
    """Extrae features y las guarda en un archivo.

    Parameters
    ----------
    manifest_path : Path | str
        Ruta al manifest CSV con sesiones procesadas
    output_path : Path | str
        Ruta donde guardar las features extraídas
    limit : int, optional
        Limitar número de sesiones a procesar (para pruebas)
    epoch_length : float
        Duración de cada epoch en segundos
    sfreq : float, optional
        Frecuencia de muestreo objetivo
    format : str
        Formato de salida: 'parquet' (default, más eficiente) o 'csv'
    """
    logging.info("Iniciando extracción de features...")
    logging.info(f"Manifest: {manifest_path}")
    logging.info(f"Output: {output_path}")

    # Cargar manifest
    manifest = pd.read_csv(manifest_path)

    # Filtrar solo sesiones OK
    manifest_ok = manifest[manifest["status"] == "ok"].copy()

    if limit:
        manifest_ok = manifest_ok.head(limit)
        logging.info(f"Procesando solo {len(manifest_ok)} sesiones (modo limit)")

    all_features = []

    for idx, row in manifest_ok.iterrows():
        # Intentar usar las rutas del manifest primero (si existen)
        psg_path_str = row.get("psg_trimmed_path", "")
        hyp_path_str = row.get("hypnogram_trimmed_path", "")

        # Construir Paths solo si las rutas no están vacías
        psg_path = Path(psg_path_str) if psg_path_str else None
        hyp_path = Path(hyp_path_str) if hyp_path_str else None

        # Si las rutas del manifest no existen o están vacías, construir rutas relativas
        if (
            not psg_path
            or not hyp_path
            or not psg_path.exists()
            or not hyp_path.exists()
        ):
            # Construir rutas relativas basadas en subject_id, subset, version
            manifest_dir = Path(manifest_path).parent
            subject_id = row["subject_id"]
            subset = row.get("subset", "sleep-cassette")
            version = row.get("version", "1.0.0")

            psg_path = (
                manifest_dir
                / "sleep_trimmed"
                / "psg"
                / f"{subject_id}_{subset}_{version}_trimmed_raw.fif"
            )
            hyp_path = (
                manifest_dir
                / "sleep_trimmed"
                / "hypnograms"
                / f"{subject_id}_{subset}_{version}_trimmed_annotations.csv"
            )

        if not psg_path.exists() or not hyp_path.exists():
            logging.warning(
                f"Archivos faltantes para {row['subject_id']} (buscado en {psg_path}), saltando"
            )
            continue

        try:
            features_df = extract_features_from_session(
                psg_path,
                hyp_path,
                epoch_length=epoch_length,
                sfreq=sfreq,
            )

            if not features_df.empty:
                features_df["subject_id"] = row["subject_id"]
                # Extraer subject_core (primeros 5 caracteres) para agrupar noches del mismo sujeto
                # Si subject_id tiene menos de 5 caracteres, usar el ID completo
                subject_id_str = str(row["subject_id"])
                features_df["subject_core"] = (
                    subject_id_str[:5] if len(subject_id_str) >= 5 else subject_id_str
                )
                features_df["session_idx"] = idx
                # Asegurar que los epochs mantengan orden temporal dentro de la sesión
                # (ya están ordenados por epoch_time_start en extract_features_from_session)
                all_features.append(features_df)
                logging.info(
                    f"Extraídas {len(features_df)} epochs de {row['subject_id']}"
                )
        except Exception as e:
            logging.exception(f"Error procesando {row['subject_id']}: {e}")
            continue

    if not all_features:
        raise ValueError("No se pudieron extraer features de ninguna sesión")

    combined = pd.concat(all_features, ignore_index=True)
    # Ordenar por sujeto y tiempo para mantener orden temporal
    combined = combined.sort_values(
        ["subject_core", "subject_id", "epoch_time_start"]
    ).reset_index(drop=True)
    logging.info(
        f"Dataset completo: {len(combined)} epochs de {combined['subject_id'].nunique()} sujetos"
    )

    # Guardar features
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "parquet":
        combined.to_parquet(output_path, index=False, engine="pyarrow")
        logging.info(f"Features guardadas en formato Parquet: {output_path}")
    elif format.lower() == "csv":
        combined.to_csv(output_path, index=False)
        logging.info(f"Features guardadas en formato CSV: {output_path}")
    else:
        raise ValueError(f"Formato no soportado: {format}. Use 'parquet' o 'csv'")

    logging.info(f"Total epochs: {len(combined)}")
    logging.info(f"Total subjects: {combined['subject_id'].nunique()}")
    logging.info(f"Features extraídas: {len(combined.columns)} columnas")


def main() -> int:
    """Función principal para ejecutar desde CLI."""
    parser = argparse.ArgumentParser(
        description="Extraer y guardar features de sesiones de sueño"
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/manifest_trimmed.csv",
        help="Ruta al manifest CSV con sesiones procesadas",
    )
    parser.add_argument(
        "--output",
        default="data/processed/features.parquet",
        help="Ruta donde guardar las features extraídas",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limitar número de sesiones a procesar (para pruebas)",
    )
    parser.add_argument(
        "--epoch-length",
        type=float,
        default=30.0,
        help="Duración de cada epoch en segundos",
    )
    parser.add_argument(
        "--sfreq",
        type=float,
        default=None,
        help="Frecuencia de muestreo objetivo (None = mantener original)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Formato de salida (parquet es más eficiente)",
    )

    args = parser.parse_args()

    try:
        extract_and_save_features(
            manifest_path=args.manifest,
            output_path=args.output,
            limit=args.limit,
            epoch_length=args.epoch_length,
            sfreq=args.sfreq,
            format=args.format,
        )
        return 0
    except Exception as e:
        logging.exception(f"Error extrayendo features: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
