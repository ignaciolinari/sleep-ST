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

from src.features import (
    DEFAULT_FILTER_BAND,
    DEFAULT_NOTCH_FREQS,
    extract_features_from_session,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_and_save_features(
    manifest_path: Path | str,
    output_path: Path | str,
    limit: int | None = None,
    epoch_length: float = 30.0,
    sfreq: float | None = None,
    format: str = "parquet",
    movement_policy: str = "drop",
    overlap: float = 0.0,
    prefilter: bool = True,
    bandpass_low: float | None = None,
    bandpass_high: float | None = None,
    notch_freqs: list[float] | None = None,
    psd_method: str = "welch",
    skip_cross_if_single_eeg: bool = True,
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

    if epoch_length <= 0:
        raise ValueError("epoch_length debe ser > 0")
    if overlap < 0:
        raise ValueError("overlap debe ser >= 0")
    if overlap >= epoch_length:
        raise ValueError("overlap debe ser menor que epoch_length")

    bp_low = DEFAULT_FILTER_BAND[0] if bandpass_low is None else float(bandpass_low)
    bp_high = DEFAULT_FILTER_BAND[1] if bandpass_high is None else float(bandpass_high)
    if bp_low >= bp_high:
        raise ValueError("bandpass_low debe ser menor que bandpass_high")
    bandpass = (bp_low, bp_high)
    notch = tuple(notch_freqs) if notch_freqs else DEFAULT_NOTCH_FREQS

    psd_method = (psd_method or "welch").lower()
    if psd_method not in {"welch", "multitaper"}:
        raise ValueError("psd_method debe ser 'welch' o 'multitaper'")

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

            base_name = f"{subject_id}_{subset}_{version}_trimmed"
            psg_dir = manifest_dir / "sleep_trimmed" / "psg"
            hyp_dir = manifest_dir / "sleep_trimmed" / "hypnograms"

            # Buscar cualquier episodio (e1ofN) si existe; si no, usar nombre base
            psg_candidates = sorted(psg_dir.glob(f"{base_name}*.fif"))
            hyp_candidates = sorted(hyp_dir.glob(f"{base_name}*_annotations.csv"))

            if psg_candidates:
                psg_path = psg_candidates[0]
            else:
                psg_path = psg_dir / f"{base_name}_raw.fif"

            if hyp_candidates:
                hyp_path = hyp_candidates[0]
            else:
                hyp_path = hyp_dir / f"{base_name}_annotations.csv"

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
                movement_policy=movement_policy,
                overlap=overlap,
                apply_prefilter=prefilter,
                bandpass=bandpass,
                notch_freqs=notch,
                psd_method=psd_method,
                skip_cross_if_single_eeg=skip_cross_if_single_eeg,
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
        "--overlap",
        type=float,
        default=0.0,
        help="Solapamiento entre epochs en segundos (debe ser menor que epoch-length).",
    )
    parser.add_argument(
        "--prefilter",
        dest="prefilter",
        action="store_true",
        help="Aplicar detrend + band-pass + notch antes de extraer features (default: on).",
    )
    parser.add_argument(
        "--no-prefilter",
        dest="prefilter",
        action="store_false",
        help="Desactivar filtrado/notch previo a la extracción de features.",
    )
    parser.set_defaults(prefilter=True)
    parser.add_argument(
        "--bandpass-low",
        type=float,
        default=None,
        help="Corte inferior del band-pass para features (Hz). Default: 0.3 Hz.",
    )
    parser.add_argument(
        "--bandpass-high",
        type=float,
        default=None,
        help="Corte superior del band-pass para features (Hz). Default: 45 Hz.",
    )
    parser.add_argument(
        "--notch-freqs",
        type=float,
        nargs="+",
        default=None,
        help="Frecuencias de notch (Hz), ej. 50 60. Default: 50 y 60 Hz.",
    )
    parser.add_argument(
        "--psd-method",
        choices=["welch", "multitaper"],
        default="welch",
        help="Método para PSD en features espectrales (welch o multitaper).",
    )
    parser.add_argument(
        "--keep-cross-single-eeg",
        dest="skip_cross_if_single_eeg",
        action="store_false",
        help="Mantener features cross-channel aunque solo haya un EEG (por defecto se omiten).",
    )
    parser.set_defaults(skip_cross_if_single_eeg=True)
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
    parser.add_argument(
        "--movement-policy",
        choices=["drop", "map_to_w", "keep_unknown"],
        default="drop",
        help="Tratamiento de anotaciones 'Movement time' o '?' (drop por defecto)",
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
            movement_policy=args.movement_policy,
            overlap=args.overlap,
            prefilter=args.prefilter,
            bandpass_low=args.bandpass_low,
            bandpass_high=args.bandpass_high,
            notch_freqs=args.notch_freqs,
            psd_method=args.psd_method,
            skip_cross_if_single_eeg=args.skip_cross_if_single_eeg,
        )
        return 0
    except Exception as e:
        logging.exception(f"Error extrayendo features: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
