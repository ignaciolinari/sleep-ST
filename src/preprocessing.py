"""Herramientas de preprocesamiento para el dataset Sleep-EDFx.

Este módulo expone un script que recorta las sesiones al periodo de sueño
añadiendo un margen configurable de vigilia antes y después. El objetivo es
generar una versión procesada de los datos que facilite los análisis sin las
horas de vigilia extendida de los cassettes.

Uso típico:

	python src/preprocessing.py \
		--manifest data/processed/manifest.csv \
		--out-root data/processed/sleep_trimmed \
		--out-manifest data/processed/manifest_trimmed.csv \
		--pre-padding 3600 --post-padding 3600

Esto producirá:
  • Archivos PSG recortados en ``<out-root>/psg`` en formato FIF.
  • Anotaciones de hipnograma recortadas en ``<out-root>/hypnograms`` como CSV.
  • Un nuevo manifest con metadatos sobre el recorte.

Se puede limitar la cantidad de sesiones a procesar con ``--limit`` para
pruebas rápidas.
"""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import mne
import pandas as pd


# Mapeo de etiquetas crudas a estadios canónicos
STAGE_CANONICAL = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}

SLEEP_STAGES = {"N1", "N2", "N3", "REM"}


@dataclass
class TrimResult:
    subject_id: str
    subset: str
    version: str
    status: str
    psg_trimmed_path: Optional[Path]
    hyp_trimmed_path: Optional[Path]
    trim_start_sec: Optional[float]
    trim_end_sec: Optional[float]
    trim_duration_sec: Optional[float]
    padding_pre_sec: float
    padding_post_sec: float
    notes: Optional[str] = None


def _canonical_stage(description: str) -> Optional[str]:
    return STAGE_CANONICAL.get(description)


def _compute_trim_bounds(
    annotations: mne.Annotations, padding_pre: float, padding_post: float
) -> Optional[tuple[float, float]]:
    timeline = []
    for onset, duration, desc in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        canonical = _canonical_stage(desc)
        timeline.append((float(onset), float(duration), canonical))

    sleep_indices = [
        idx
        for idx, (_, _, canonical) in enumerate(timeline)
        if canonical in SLEEP_STAGES
    ]
    if not sleep_indices:
        return None

    first_idx = sleep_indices[0]
    last_idx = sleep_indices[-1]
    sleep_start = timeline[first_idx][0]
    sleep_end = timeline[last_idx][0] + timeline[last_idx][1]

    total_duration = timeline[-1][0] + timeline[-1][1]
    trim_start = max(0.0, sleep_start - padding_pre)
    trim_end = min(total_duration, sleep_end + padding_post)
    if trim_end <= trim_start:
        return None
    return trim_start, trim_end


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(f"No se encontró el manifest en {manifest_path}")
    df = pd.read_csv(manifest_path)
    expected_cols = {
        "subject_id",
        "subset",
        "version",
        "psg_path",
        "hypnogram_path",
        "status",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"El manifest no contiene las columnas esperadas: faltan {sorted(missing)}"
        )
    return df


def _write_manifest(results: Iterable[TrimResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "subject_id",
                "subset",
                "version",
                "status",
                "psg_trimmed_path",
                "hypnogram_trimmed_path",
                "trim_start_sec",
                "trim_end_sec",
                "trim_duration_sec",
                "padding_pre_sec",
                "padding_post_sec",
                "notes",
            ]
        )
        for res in results:
            writer.writerow(
                [
                    res.subject_id,
                    res.subset,
                    res.version,
                    res.status,
                    str(res.psg_trimmed_path) if res.psg_trimmed_path else "",
                    str(res.hyp_trimmed_path) if res.hyp_trimmed_path else "",
                    (
                        f"{res.trim_start_sec:.3f}"
                        if res.trim_start_sec is not None
                        else ""
                    ),
                    f"{res.trim_end_sec:.3f}" if res.trim_end_sec is not None else "",
                    (
                        f"{res.trim_duration_sec:.3f}"
                        if res.trim_duration_sec is not None
                        else ""
                    ),
                    f"{res.padding_pre_sec:.1f}",
                    f"{res.padding_post_sec:.1f}",
                    res.notes or "",
                ]
            )


def _process_session(
    row: pd.Series,
    out_psg_dir: Path,
    out_hyp_dir: Path,
    padding_pre: float,
    padding_post: float,
    overwrite: bool,
) -> TrimResult:
    subject_id = row["subject_id"]
    subset = row["subset"]
    version = row["version"]
    status = row.get("status", "")
    psg_path = Path(row["psg_path"])
    hyp_path = Path(row["hypnogram_path"])

    if status != "ok":
        return TrimResult(
            subject_id,
            subset,
            version,
            status,
            None,
            None,
            None,
            None,
            None,
            padding_pre,
            padding_post,
            notes="Estado != ok",
        )

    if not psg_path.exists() or not hyp_path.exists():
        return TrimResult(
            subject_id,
            subset,
            version,
            status,
            None,
            None,
            None,
            None,
            None,
            padding_pre,
            padding_post,
            notes="Archivos faltantes",
        )

    try:
        annotations = mne.read_annotations(hyp_path)
    except Exception as exc:  # pragma: no cover - dependencias externas
        logging.exception("No se pudo leer el hipnograma %s", hyp_path)
        return TrimResult(
            subject_id,
            subset,
            version,
            status,
            None,
            None,
            None,
            None,
            None,
            padding_pre,
            padding_post,
            notes=f"Error leyendo hipnograma: {exc}",
        )

    bounds = _compute_trim_bounds(annotations, padding_pre, padding_post)
    if bounds is None:
        return TrimResult(
            subject_id,
            subset,
            version,
            status,
            None,
            None,
            None,
            None,
            None,
            padding_pre,
            padding_post,
            notes="No se encontró ventana de sueño",
        )

    trim_start, trim_end = bounds
    trim_duration = trim_end - trim_start

    out_psg_dir.mkdir(parents=True, exist_ok=True)
    out_hyp_dir.mkdir(parents=True, exist_ok=True)

    psg_out = out_psg_dir / f"{subject_id}_{subset}_{version}_trimmed_raw.fif"
    hyp_out = out_hyp_dir / f"{subject_id}_{subset}_{version}_trimmed_annotations.csv"

    if not overwrite and psg_out.exists() and hyp_out.exists():
        logging.info(
            "Archivos ya existen para %s, se omite (usar --overwrite para regenerar)",
            subject_id,
        )
        return TrimResult(
            subject_id,
            subset,
            version,
            status,
            psg_out,
            hyp_out,
            trim_start,
            trim_end,
            trim_duration,
            padding_pre,
            padding_post,
            notes="Reuse",
        )

    try:
        trimmed_annotations = annotations.copy()
        trimmed_annotations.crop(tmin=trim_start, tmax=trim_end)
        trimmed_annotations.onset = trimmed_annotations.onset - trim_start

        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
        raw.crop(tmin=trim_start, tmax=trim_end)
        raw.set_annotations(trimmed_annotations)
        raw.save(psg_out, overwrite=True)

        ann_df = pd.DataFrame(
            {
                "onset": trimmed_annotations.onset,
                "duration": trimmed_annotations.duration,
                "description": trimmed_annotations.description,
            }
        )
        ann_df.to_csv(hyp_out, index=False)
    except Exception as exc:  # pragma: no cover - dependencias externas
        logging.exception("Fallo al recortar %s", subject_id)
        return TrimResult(
            subject_id,
            subset,
            version,
            status,
            None,
            None,
            None,
            None,
            None,
            padding_pre,
            padding_post,
            notes=f"Error procesando sesión: {exc}",
        )

    return TrimResult(
        subject_id,
        subset,
        version,
        status,
        psg_out,
        hyp_out,
        trim_start,
        trim_end,
        trim_duration,
        padding_pre,
        padding_post,
    )


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    manifest_path = Path(args.manifest)
    out_root = Path(args.out_root)
    out_manifest = Path(args.out_manifest)
    out_psg_dir = out_root / "psg"
    out_hyp_dir = out_root / "hypnograms"

    df = _load_manifest(manifest_path)
    to_process = df[df["status"] == "ok"]
    if args.limit:
        to_process = to_process.head(args.limit)
        logging.info("Procesando sólo %s sesiones (modo límite)", len(to_process))
    else:
        logging.info("Procesando %s sesiones ok", len(to_process))

    results: list[TrimResult] = []
    for _, row in to_process.iterrows():
        res = _process_session(
            row,
            out_psg_dir,
            out_hyp_dir,
            args.pre_padding,
            args.post_padding,
            args.overwrite,
        )
        results.append(res)

    _write_manifest(results, out_manifest)
    logging.info("Manifest recortado guardado en %s", out_manifest)
    ok = sum(1 for r in results if r.psg_trimmed_path)
    skipped = len(results) - ok
    logging.info("Sesiones exitosas: %s | saltadas: %s", ok, skipped)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recortar sesiones Sleep-EDFx alrededor del sueño"
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/manifest.csv",
        help="CSV con las sesiones originales",
    )
    parser.add_argument(
        "--out-root",
        default="data/processed/sleep_trimmed",
        help="Directorio raíz para los archivos recortados",
    )
    parser.add_argument(
        "--out-manifest",
        default="data/processed/manifest_trimmed.csv",
        help="Ruta del manifest resultante",
    )
    parser.add_argument(
        "--pre-padding",
        type=float,
        default=3600.0,
        help="Segundos de vigilia a conservar antes del inicio del sueño",
    )
    parser.add_argument(
        "--post-padding",
        type=float,
        default=3600.0,
        help="Segundos de vigilia a conservar tras el despertar final",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limitar la cantidad de sesiones para pruebas (0 = todas)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Reescribir archivos existentes"
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":  # pragma: no cover - punto de entrada script
    raise SystemExit(main())
