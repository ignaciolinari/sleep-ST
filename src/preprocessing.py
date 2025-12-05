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
		--pre-padding 900 --post-padding 900 \
		--episode-strategy all --wake-gap-min 60 --min-episode-min 20

Esto producirá:
  • Archivos PSG recortados en ``<out-root>/psg`` en formato FIF.
  • Anotaciones de hipnograma recortadas en ``<out-root>/hypnograms`` como CSV.
  • Un nuevo manifest con metadatos sobre el recorte.

Se puede limitar la cantidad de sesiones a procesar con ``--limit`` para
pruebas rápidas. Las opciones ``--episode-strategy``, ``--wake-gap-min`` y
``--min-episode-min`` permiten controlar cómo segmentar episodios cuando no hay
anotaciones de ``Lights Off``/``Lights On``.
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

# Mapeo de nombres de canal a tipos correctos para Sleep-EDF.
# Los archivos EDF originales no especifican tipos correctamente,
# causando que MNE clasifique todo como 'eeg'.
CHANNEL_TYPE_MAPPING = {
    "EEG Fpz-Cz": "eeg",
    "EEG Pz-Oz": "eeg",
    "EOG horizontal": "eog",
    "Resp oro-nasal": "misc",
    "EMG submental": "emg",
    "Temp rectal": "misc",
    "Event marker": "stim",
}


def _fix_channel_types(raw: mne.io.Raw) -> mne.io.Raw:
    """Corrige los tipos de canal para archivos Sleep-EDF.

    Los archivos EDF de Sleep-EDF no especifican correctamente los tipos
    de canal, lo que causa que MNE clasifique todos los canales como 'eeg'.
    Esto es problemático cuando se aplica referencia promedio, ya que
    incluiría canales no-EEG (temperatura, EMG, etc.) en el cálculo.

    Parameters
    ----------
    raw : mne.io.Raw
        Objeto Raw con canales posiblemente mal clasificados.

    Returns
    -------
    mne.io.Raw
        El mismo objeto Raw con tipos de canal corregidos.
    """
    types_to_set = {}
    for ch_name in raw.ch_names:
        if ch_name in CHANNEL_TYPE_MAPPING:
            current_type = raw.get_channel_types([ch_name])[0]
            expected_type = CHANNEL_TYPE_MAPPING[ch_name]
            if current_type != expected_type:
                types_to_set[ch_name] = expected_type

    if types_to_set:
        raw.set_channel_types(types_to_set, verbose="ERROR")
        logging.debug("Tipos de canal corregidos: %s", types_to_set)

    return raw


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
    sleep_duration_sec: Optional[float] = None
    episode_index: Optional[int] = None
    episodes_total: Optional[int] = None
    episode_strategy: Optional[str] = None
    notes: Optional[str] = None


def _canonical_stage(description: str) -> Optional[str]:
    return STAGE_CANONICAL.get(description)


def _build_timeline(
    annotations: mne.Annotations,
) -> list[tuple[float, float, Optional[str]]]:
    timeline: list[tuple[float, float, Optional[str]]] = []
    for onset, duration, desc in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        canonical = _canonical_stage(desc)
        timeline.append((float(onset), float(duration), canonical))
    return timeline


def _total_recording_duration(
    timeline: list[tuple[float, float, Optional[str]]],
) -> float:
    if not timeline:
        return 0.0
    onset, duration, _ = timeline[-1]
    return onset + duration


def _find_sleep_indices(
    timeline: list[tuple[float, float, Optional[str]]],
) -> list[int]:
    return [
        idx
        for idx, (_, _, canonical) in enumerate(timeline)
        if canonical in SLEEP_STAGES
    ]


def _compute_spt_bounds(
    timeline: list[tuple[float, float, Optional[str]]],
) -> Optional[tuple[float, float]]:
    sleep_indices = _find_sleep_indices(timeline)
    if not sleep_indices:
        return None

    first_idx = sleep_indices[0]
    last_idx = sleep_indices[-1]
    sleep_start = timeline[first_idx][0]
    sleep_end = timeline[last_idx][0] + timeline[last_idx][1]
    return sleep_start, sleep_end


def _compute_trim_bounds_from_spt(
    timeline: list[tuple[float, float, Optional[str]]],
    spt_bounds: tuple[float, float],
    padding_pre: float,
    padding_post: float,
) -> tuple[float, float]:
    total_duration = _total_recording_duration(timeline)
    sleep_start, sleep_end = spt_bounds
    trim_start = max(0.0, sleep_start - padding_pre)
    trim_end = min(total_duration, sleep_end + padding_post)
    return trim_start, trim_end


# Sleep episode segmentation -------------------------------------------------


def _generate_sleep_segments(
    timeline: list[tuple[float, float, Optional[str]]],
) -> list[tuple[float, float, float]]:
    """Return sleep segments as (start, end, duration)."""

    segments: list[tuple[float, float, float]] = []
    for onset, duration, canonical in timeline:
        if canonical in SLEEP_STAGES:
            segments.append((onset, onset + duration, duration))
    return segments


def _merge_segments_with_gap(
    segments: list[tuple[float, float, float]],
    max_gap_sec: float,
) -> list[tuple[float, float, float]]:
    if not segments:
        return []

    merged: list[tuple[float, float, float]] = []
    current_start, current_end, current_sleep = segments[0]

    for onset, offset, sleep_duration in segments[1:]:
        gap = onset - current_end
        if gap <= max_gap_sec:
            current_end = max(current_end, offset)
            current_sleep += sleep_duration
        else:
            merged.append((current_start, current_end, current_sleep))
            current_start, current_end, current_sleep = onset, offset, sleep_duration

    merged.append((current_start, current_end, current_sleep))
    return merged


def _filter_segments_by_sleep_duration(
    segments: list[tuple[float, float, float]],
    min_sleep_duration_sec: float,
) -> list[tuple[float, float, float]]:
    if min_sleep_duration_sec <= 0:
        return segments
    return [seg for seg in segments if seg[2] >= min_sleep_duration_sec]


def _choose_segments_by_strategy(
    segments: list[tuple[float, float, float]],
    strategy: str,
) -> list[tuple[float, float, float]]:
    if not segments:
        return []

    if strategy == "longest":
        return [max(segments, key=lambda seg: seg[2])]
    if strategy == "spt":
        return [segments[0]]
    if strategy == "all":
        return segments
    logging.warning("Estrategia de episodio desconocida: %s", strategy)
    return [segments[0]]


def _expand_segments_with_padding(
    segments: list[tuple[float, float, float]],
    timeline: list[tuple[float, float, Optional[str]]],
    padding_pre: float,
    padding_post: float,
) -> list[dict[str, float]]:
    total_duration = _total_recording_duration(timeline)
    expanded: list[dict[str, float]] = []
    for start, end, sleep_duration in segments:
        trim_start = max(0.0, start - padding_pre)
        trim_end = min(total_duration, end + padding_post)
        expanded.append(
            {
                "episode_start": start,
                "episode_end": end,
                "trim_start": trim_start,
                "trim_end": trim_end,
                "sleep_duration": sleep_duration,
            }
        )
    return expanded


def _find_sleep_episodes(
    annotations: mne.Annotations,
    padding_pre: float,
    padding_post: float,
    wake_gap_sec: float,
    min_episode_sleep_sec: float,
    strategy: str,
) -> list[dict[str, float]]:
    timeline = _build_timeline(annotations)
    spt_bounds = _compute_spt_bounds(timeline)
    if spt_bounds is None:
        return []

    if strategy == "spt":
        sleep_start, sleep_end = spt_bounds
        segments = [(sleep_start, sleep_end, sleep_end - sleep_start)]
    else:
        sleep_segments = _generate_sleep_segments(timeline)
        merged_segments = _merge_segments_with_gap(
            sleep_segments, max(0.0, wake_gap_sec)
        )
        filtered_segments = _filter_segments_by_sleep_duration(
            merged_segments, max(0.0, min_episode_sleep_sec)
        )
        segments = _choose_segments_by_strategy(filtered_segments, strategy)
        if not segments:
            return []

    segments_with_padding = _expand_segments_with_padding(
        segments, timeline, padding_pre, padding_post
    )
    spt_start, spt_end = spt_bounds
    spt_duration = spt_end - spt_start
    for item in segments_with_padding:
        item["spt_start"] = spt_start
        item["spt_end"] = spt_end
        item["spt_duration"] = spt_duration
    return segments_with_padding


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
                "sleep_duration_sec",
                "padding_pre_sec",
                "padding_post_sec",
                "episode_index",
                "episodes_total",
                "episode_strategy",
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
                    (
                        f"{res.sleep_duration_sec:.3f}"
                        if res.sleep_duration_sec is not None
                        else ""
                    ),
                    f"{res.padding_pre_sec:.1f}",
                    f"{res.padding_post_sec:.1f}",
                    res.episode_index if res.episode_index is not None else "",
                    res.episodes_total if res.episodes_total is not None else "",
                    res.episode_strategy or "",
                    res.notes or "",
                ]
            )


def _process_session(
    row: pd.Series,
    out_psg_dir: Path,
    out_hyp_dir: Path,
    padding_pre: float,
    padding_post: float,
    wake_gap_min: float,
    min_episode_sleep_min: float,
    episode_strategy: str,
    overwrite: bool,
    resample_sfreq: Optional[float] = None,
    l_freq: Optional[float] = None,
    h_freq: Optional[float] = None,
    notch_freqs: Optional[list[float]] = None,
    avg_ref: bool = False,
) -> list[TrimResult]:
    subject_id = row["subject_id"]
    subset = row["subset"]
    version = row["version"]
    status = row.get("status", "")
    psg_path = Path(row["psg_path"])
    hyp_path = Path(row["hypnogram_path"])

    if status != "ok":
        return [
            TrimResult(
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
        ]

    if not psg_path.exists() or not hyp_path.exists():
        return [
            TrimResult(
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
        ]

    try:
        annotations = mne.read_annotations(hyp_path)
    except Exception as exc:  # pragma: no cover - dependencias externas
        logging.exception("No se pudo leer el hipnograma %s", hyp_path)
        return [
            TrimResult(
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
        ]

    if episode_strategy not in {"spt", "longest", "all"}:
        logging.warning("Estrategia %s no válida, se forza a 'spt'", episode_strategy)
        episode_strategy = "spt"

    episodes = _find_sleep_episodes(
        annotations,
        padding_pre,
        padding_post,
        wake_gap_min * 60.0,
        min_episode_sleep_min * 60.0,
        episode_strategy,
    )

    if not episodes:
        return [
            TrimResult(
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
        ]

    results: list[TrimResult] = []
    episodes_total = len(episodes)
    out_psg_dir.mkdir(parents=True, exist_ok=True)
    out_hyp_dir.mkdir(parents=True, exist_ok=True)

    for idx, episode in enumerate(episodes, start=1):
        trim_start = episode["trim_start"]
        trim_end = episode["trim_end"]
        trim_duration = trim_end - trim_start
        sleep_duration = episode["sleep_duration"]

        suffix = "" if episodes_total == 1 else f"_e{idx}of{episodes_total}"
        psg_out = (
            out_psg_dir / f"{subject_id}_{subset}_{version}_trimmed{suffix}_raw.fif"
        )
        hyp_out = (
            out_hyp_dir
            / f"{subject_id}_{subset}_{version}_trimmed{suffix}_annotations.csv"
        )

        if not overwrite and psg_out.exists() and hyp_out.exists():
            logging.info(
                "Archivos ya existen para %s episodio %s, se omite (usar --overwrite para regenerar)",
                subject_id,
                idx,
            )
            filter_note = []
            if l_freq is not None or h_freq is not None:
                filter_note.append(
                    f"BP {l_freq if l_freq is not None else 0.0}-{h_freq if h_freq is not None else 'nyq'} Hz"
                )
            if resample_sfreq:
                filter_note.append(f"Resample {resample_sfreq} Hz")
            if notch_freqs:
                filter_note.append("Notch " + ",".join(str(f) for f in notch_freqs))
            if avg_ref:
                filter_note.append("AvgRef EEG")
            extra_note = "; ".join(filter_note)
            results.append(
                TrimResult(
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
                    sleep_duration_sec=sleep_duration,
                    episode_index=idx,
                    episodes_total=episodes_total,
                    episode_strategy=episode_strategy,
                    notes="Reuse" + (f" | {extra_note}" if extra_note else ""),
                )
            )
            continue

        try:
            raw = mne.io.read_raw_edf(psg_path, preload=True, verbose="ERROR")
            # Corregir tipos de canal antes de cualquier procesamiento
            # (necesario para que avg_ref solo use canales EEG reales)
            _fix_channel_types(raw)
            if (
                resample_sfreq
                and abs(raw.info.get("sfreq", 0.0) - resample_sfreq) > 1e-3
            ):
                raw.resample(resample_sfreq, npad="auto", verbose="ERROR")
            recording_tmax = raw.times[-1]
            effective_trim_end = min(trim_end, recording_tmax)
            if effective_trim_end <= trim_start:
                logging.warning(
                    "Ventana sin longitud útil tras ajustar al máximo del PSG (%s) episodio %s",
                    subject_id,
                    idx,
                )
                results.append(
                    TrimResult(
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
                        episode_index=idx,
                        episodes_total=episodes_total,
                        episode_strategy=episode_strategy,
                        notes="Ventana vacía tras ajustar al final del PSG",
                    )
                )
                continue

            trim_end = effective_trim_end
            trim_duration = trim_end - trim_start

            trimmed_annotations = annotations.copy()
            trimmed_annotations.crop(tmin=trim_start, tmax=trim_end)
            trimmed_annotations.onset = trimmed_annotations.onset - trim_start

            raw.crop(tmin=trim_start, tmax=trim_end)
            raw.set_annotations(trimmed_annotations)

            # Opcional: filtrado band-pass y notch, y re-referenciado
            if l_freq is not None or h_freq is not None:
                raw.filter(
                    l_freq=l_freq,
                    h_freq=h_freq,
                    method="fir",
                    fir_design="firwin",
                    verbose="ERROR",
                )
            if notch_freqs:
                sfreq = raw.info.get("sfreq", 0.0) or 0.0
                nyq = sfreq / 2.0 if sfreq else 0.0
                safe_notch: list[float] = []
                for f in notch_freqs:
                    if not nyq or f >= nyq:
                        continue
                    # Alejarse del Nyquist para evitar ValueError interno
                    if nyq - f < 1.0:
                        safe_notch.append(nyq - 1.0)
                    else:
                        safe_notch.append(f)
                if not safe_notch:
                    logging.warning(
                        "Notch omitido para %s: freqs %s no son válidas con sfreq=%.2f",
                        subject_id,
                        notch_freqs,
                        sfreq,
                    )
                else:
                    raw.notch_filter(
                        freqs=safe_notch,
                        method="fir",
                        fir_design="firwin",
                        notch_widths=1.0,
                        verbose="ERROR",
                    )
            if avg_ref:
                raw.set_eeg_reference("average", verbose="ERROR")
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
            logging.exception("Fallo al recortar %s episodio %s", subject_id, idx)
            results.append(
                TrimResult(
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
                    episode_index=idx,
                    episodes_total=episodes_total,
                    episode_strategy=episode_strategy,
                    notes=f"Error procesando episodio: {exc}",
                )
            )
            continue

        filter_note = []
        if l_freq is not None or h_freq is not None:
            filter_note.append(
                f"BP {l_freq if l_freq is not None else 0.0}-{h_freq if h_freq is not None else 'nyq'} Hz"
            )
        if resample_sfreq:
            filter_note.append(f"Resample {resample_sfreq} Hz")
        if notch_freqs:
            filter_note.append("Notch " + ",".join(str(f) for f in notch_freqs))
        if avg_ref:
            filter_note.append("AvgRef EEG")
        filter_note_str = "; ".join(filter_note)

        base_note = (
            "Episodio recortado"
            if episodes_total == 1
            else f"Episodio {idx}/{episodes_total}"
        ) + (
            " (ajustado al límite del PSG)"
            if abs(trim_end - episode["trim_end"]) > 1e-6
            else ""
        )
        final_note = base_note + (f" | {filter_note_str}" if filter_note_str else "")

        results.append(
            TrimResult(
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
                sleep_duration_sec=sleep_duration,
                episode_index=idx,
                episodes_total=episodes_total,
                episode_strategy=episode_strategy,
                notes=final_note,
            )
        )

    return results


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
        res_items = _process_session(
            row,
            out_psg_dir,
            out_hyp_dir,
            args.pre_padding,
            args.post_padding,
            args.wake_gap_min,
            args.min_episode_min,
            args.episode_strategy,
            args.overwrite,
            args.resample_sfreq,
            args.filter_lowcut,
            args.filter_highcut,
            args.notch_freqs,
            args.avg_ref,
        )
        results.extend(res_items)

    _write_manifest(results, out_manifest)
    logging.info("Manifest recortado guardado en %s", out_manifest)
    ok = sum(1 for r in results if r.psg_trimmed_path)
    skipped = len(results) - ok
    logging.info("Episodios exitosos: %s | saltados: %s", ok, skipped)
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
        default=900.0,
        help="Segundos de vigilia a conservar antes del inicio del sueño",
    )
    parser.add_argument(
        "--post-padding",
        type=float,
        default=900.0,
        help="Segundos de vigilia a conservar tras el despertar final",
    )
    parser.add_argument(
        "--episode-strategy",
        choices=["spt", "longest", "all"],
        default="spt",
        help=(
            "Cómo seleccionar episodios de sueño detectados: 'spt' usa toda la "
            "ventana sueño-sueño; 'longest' recorta sólo el episodio de sueño "
            "más largo; 'all' exporta todos los episodios"
        ),
    )
    parser.add_argument(
        "--wake-gap-min",
        type=float,
        default=60.0,
        help=(
            "Minutos de vigilia continua necesarios para separar episodios "
            "cuando se usa 'longest' o 'all'"
        ),
    )
    parser.add_argument(
        "--min-episode-min",
        type=float,
        default=20.0,
        help=(
            "Duración mínima (min) de sueño acumulado para conservar un episodio "
            "cuando se usa 'longest' o 'all'"
        ),
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
    parser.add_argument(
        "--filter-lowcut",
        type=float,
        default=None,
        help="Corte inferior del band-pass (Hz). None = sin filtro pasa-altas.",
    )
    parser.add_argument(
        "--filter-highcut",
        type=float,
        default=None,
        help="Corte superior del band-pass (Hz). None = sin filtro pasa-bajas.",
    )
    parser.add_argument(
        "--resample-sfreq",
        type=float,
        default=None,
        help="Re-muestrear PSG a esta frecuencia (Hz) antes de filtrar. None = sin re-muestreo.",
    )
    parser.add_argument(
        "--notch-freqs",
        type=float,
        nargs="+",
        default=None,
        help="Frecuencias de notch (Hz), p.ej. 50 o 50 60. None = sin notch.",
    )
    parser.add_argument(
        "--avg-ref",
        action="store_true",
        help="Aplicar referencia promedio a canales EEG tras filtrar.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":  # pragma: no cover - punto de entrada script
    raise SystemExit(main())
