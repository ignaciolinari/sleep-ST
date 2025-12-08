"""Herramientas para visualizar señales PSG de un sujeto del dataset Sleep-EDFx.

Uso típico
---------

    # Ver 2 minutos de señales EEG/EOG/EMG del sujeto SC4001E
    python src/view_subject.py --subject-id SC4001E --duration 120

    # Listar canales disponibles antes de graficar
    python src/view_subject.py --subject-id SC4001E --list-channels

Requiere haber generado previamente `data/processed/manifest.csv` con `src/manifest.py`
para resolver las rutas de PSG/Hypnograma.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

DEFAULT_CHANNELS: Sequence[str] = (
    "EEG Fpz-Cz",
    "EEG Pz-Oz",
    "EOG horizontal",
    "EMG submental",
)

STAGE_TO_VALUE = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,  # Combinar N3/N4
    "Sleep stage R": 4,
}

VALUE_TO_STAGE = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}


@dataclass
class SessionInfo:
    subject_id: str
    psg_path: str
    hypnogram_path: str | None


def _parse_channels(value: str | None) -> list[str] | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value.lower() == "all":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_manifest(manifest_path: str) -> pd.DataFrame:
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"No se encontró el manifest en {manifest_path}. Generalo con src/manifest.py"
        )
    df = pd.read_csv(manifest_path)
    expected_cols = {
        "subject_id",
        "psg_path",
        "hypnogram_path",
        "status",
        "subset",
        "version",
    }
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"El manifest no tiene las columnas esperadas. Faltan: {', '.join(sorted(missing))}"
        )
    return df


def _resolve_session(
    subject_id: str,
    manifest_path: str,
    subset: str | None,
    version: str | None,
    require_ok_status: bool = True,
) -> SessionInfo:
    df = _load_manifest(manifest_path)
    mask = df["subject_id"].astype(str) == subject_id
    if subset:
        mask &= df["subset"].astype(str) == subset
    if version:
        mask &= df["version"].astype(str) == version

    matches = df.loc[mask]
    if matches.empty:
        raise ValueError(
            f"No se encontró una sesión para subject_id={subject_id!r} en el manifest {manifest_path}"
        )
    if len(matches) > 1:
        print(
            f"Aviso: se encontraron {len(matches)} sesiones para {subject_id}, se utilizará la primera.",
            file=sys.stderr,
        )

    row = matches.iloc[0]
    if require_ok_status and row["status"] != "ok":
        raise ValueError(
            "La sesión seleccionada no está completa (status != 'ok'). "
            "Revisa que existan los archivos PSG e Hypnograma."
        )

    psg_path = row["psg_path"]
    hypnogram_path = (
        row["hypnogram_path"]
        if isinstance(row["hypnogram_path"], str) and row["hypnogram_path"]
        else None
    )

    if not isinstance(psg_path, str) or not psg_path:
        raise ValueError(
            f"La ruta a PSG no es válida en el manifest (valor: {psg_path!r})."
        )

    return SessionInfo(
        subject_id=subject_id, psg_path=psg_path, hypnogram_path=hypnogram_path
    )


def _print_channels(psg_path: str) -> None:
    raw = mne.io.read_raw_edf(psg_path, preload=False, verbose=False)
    print(f"Canales en {psg_path}:")
    width = len(str(len(raw.ch_names)))
    for idx, name in enumerate(raw.ch_names, start=1):
        print(f"  {idx:>{width}}. {name}")


def _load_raw(
    psg_path: str,
    channels: Sequence[str] | None,
    resample: float | None,
    verbose: bool,
) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=verbose)
    available_channels = list(raw.ch_names)

    if channels is not None:
        if len(channels) == 0:  # "all"
            picks = available_channels
        else:
            missing = [ch for ch in channels if ch not in available_channels]
            if missing:
                raise ValueError(
                    "Los siguientes canales no existen en el archivo PSG: "
                    + ", ".join(missing)
                )
            picks = list(channels)
        raw.pick(picks)
    else:
        # Canal por defecto: EEG/EOG/EMG si existen
        picks = [ch for ch in DEFAULT_CHANNELS if ch in available_channels]
        if not picks:
            picks = available_channels
        raw.pick(picks)

    if resample is not None:
        if resample <= 0:
            raise ValueError("resample debe ser > 0")
        raw.resample(resample)

    raw.load_data()
    return raw


def _extract_data(
    raw: mne.io.BaseRaw,
    start: float,
    duration: float | None,
) -> tuple[mne.io.BaseRaw, list[float], list[list[float]]]:
    start = max(0.0, float(start))
    sfreq = float(raw.info["sfreq"])

    start_sample = int(round(start * sfreq))
    if duration is not None:
        if duration <= 0:
            raise ValueError("duration debe ser > 0")
        stop_sample = start_sample + int(round(duration * sfreq))
    else:
        stop_sample = None

    data_array = raw.get_data(start=start_sample, stop=stop_sample)
    if isinstance(data_array, tuple):
        data_array = data_array[0]
    data_array = np.asarray(data_array)
    if data_array.ndim == 1:
        data_array = data_array[np.newaxis, :]
    n_samples = data_array.shape[1]
    sample_indices = start_sample + np.arange(n_samples)
    times = sample_indices / sfreq
    return raw, times.tolist(), data_array.tolist()


def _plot_signals(
    raw: mne.io.BaseRaw,
    times: Sequence[float],
    data: Sequence[Sequence[float]],
    subject_id: str,
    start: float,
    duration: float | None,
    stage_series: tuple[Sequence[float], Sequence[int]] | None,
    save_path: str | None,
    dpi: int,
) -> None:
    channel_names = raw.ch_names
    n_channels = len(channel_names)
    total_plots = n_channels + (1 if stage_series else 0)
    height = max(3, total_plots * 1.8)

    fig, axes = plt.subplots(total_plots, 1, figsize=(14, height), sharex=True)
    if total_plots == 1:
        axes = [axes]

    for idx, (ax, ch_name) in enumerate(zip(axes, channel_names)):
        ax.plot(times, data[idx], linewidth=0.8)
        ax.set_ylabel(ch_name)
        ax.grid(True, linestyle="--", alpha=0.3)

    x_label = "Tiempo (s)"

    if stage_series:
        stage_times, stage_values = stage_series
        ax = axes[-1]
        ax.step(stage_times, stage_values, where="post", color="black")
        ax.set_ylabel("Estadio")
        yticks = sorted({value for value in stage_values if value in VALUE_TO_STAGE})
        ax.set_yticks(yticks)
        ax.set_yticklabels([VALUE_TO_STAGE[val] for val in yticks])
        ax.grid(True, linestyle="--", alpha=0.3)

    axes[-1].set_xlabel(x_label)
    duration_txt = f" | Δt = {duration:.1f}s" if duration else ""
    fig.suptitle(
        f"Subject {subject_id} | inicio = {start:.1f}s{duration_txt}", fontsize=14
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi)
        print(f"Figura guardada en {save_path}")
    else:
        plt.show()


def _maybe_build_stage_series(
    hypnogram_path: str | None,
    start: float,
    duration: float | None,
    verbose: bool,
) -> tuple[list[float], list[int]] | None:
    if not hypnogram_path or not os.path.exists(hypnogram_path):
        return None

    annotations = mne.read_annotations(hypnogram_path)
    stage_times: list[float] = []
    stage_values: list[int] = []
    end = start + duration if duration is not None else None

    for onset, dur, desc in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        if desc not in STAGE_TO_VALUE:
            # Ignorar anotaciones desconocidas (por ejemplo, movimientos)
            continue
        seg_start = max(onset, start)
        seg_end = onset + dur
        if end is not None:
            seg_end = min(seg_end, end)
        if seg_end <= seg_start:
            continue
        value = STAGE_TO_VALUE[desc]
        if stage_times:
            # Garantizar continuidad para step plot
            if stage_times[-1] != seg_start:
                stage_times.append(seg_start)
                stage_values.append(stage_values[-1])
        stage_times.extend([seg_start, seg_end])
        stage_values.extend([value, value])
        if end is not None and seg_end >= end:
            break
    if not stage_times:
        return None
    return stage_times, stage_values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualiza señales PSG de un sujeto Sleep-EDFx"
    )
    parser.add_argument(
        "--subject-id", required=False, help="Identificador del sujeto (ej.: SC4001E)"
    )
    parser.add_argument(
        "--psg-path",
        help="Ruta directa al archivo PSG (.edf). Si se proporciona, no se usa el manifest.",
    )
    parser.add_argument(
        "--hypnogram-path",
        help="Ruta directa al archivo Hypnograma (.edf). Opcional si se usa manifest.",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/manifest.csv",
        help="Ruta al CSV generado por src/manifest.py",
    )
    parser.add_argument(
        "--subset", help="Filtrar subset dentro del manifest (ej.: sleep-cassette)"
    )
    parser.add_argument(
        "--version", help="Filtrar versión dentro del manifest (ej.: 1.0.0)"
    )
    parser.add_argument(
        "--channels",
        help="Lista de canales separada por comas o 'all' para usar todos",
    )
    parser.add_argument(
        "--start",
        type=float,
        default=0.0,
        help="Tiempo inicial en segundos desde el comienzo de la grabación",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Duración en segundos a visualizar (usar 0 o omitir para ver todo)",
    )
    parser.add_argument(
        "--resample",
        type=float,
        help="Frecuencia (Hz) para resamplear antes de graficar (útil para acelerar).",
    )
    parser.add_argument(
        "--with-hypnogram",
        dest="with_hypnogram",
        action="store_true",
        help="Forzar la inclusión del hipnograma si está disponible",
    )
    parser.add_argument(
        "--no-hypnogram",
        dest="with_hypnogram",
        action="store_false",
        help="Deshabilitar la inclusión del hipnograma aunque esté disponible",
    )
    parser.add_argument(
        "--save",
        help="Ruta de salida para guardar la figura en vez de mostrarla en pantalla",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI para guardar la figura",
    )
    parser.add_argument(
        "--list-channels",
        action="store_true",
        help="Sólo listar canales del PSG y salir (no grafica)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Habilitar mensajes verbosos de MNE",
    )
    parser.set_defaults(with_hypnogram=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.psg_path:
        if not args.subject_id:
            parser.error("Debe especificar --subject-id o --psg-path")
        session = _resolve_session(
            subject_id=args.subject_id,
            manifest_path=args.manifest,
            subset=args.subset,
            version=args.version,
        )
        subject_id = session.subject_id
        psg_path = session.psg_path
        hypnogram_path = args.hypnogram_path or session.hypnogram_path
    else:
        psg_path = args.psg_path
        hypnogram_path = args.hypnogram_path
        subject_id = args.subject_id or os.path.splitext(os.path.basename(psg_path))[0]

    if not os.path.exists(psg_path):
        raise FileNotFoundError(f"No se encontró el archivo PSG en {psg_path}")

    if args.list_channels:
        _print_channels(psg_path)
        return 0

    channels = _parse_channels(args.channels)
    raw = _load_raw(
        psg_path=psg_path,
        channels=channels,
        resample=args.resample,
        verbose=args.verbose,
    )

    duration = None if args.duration in (None, 0) else args.duration
    raw, times, data = _extract_data(raw, start=args.start, duration=duration)

    if args.with_hypnogram is None:
        stage_series = _maybe_build_stage_series(
            hypnogram_path, start=args.start, duration=duration, verbose=args.verbose
        )
    elif args.with_hypnogram:
        stage_series = _maybe_build_stage_series(
            hypnogram_path, start=args.start, duration=duration, verbose=args.verbose
        )
        if stage_series is None:
            print(
                "Aviso: no se pudo generar el hipnograma (no disponible o sin anotaciones válidas).",
                file=sys.stderr,
            )
    else:
        stage_series = None

    _plot_signals(
        raw=raw,
        times=times,
        data=data,
        subject_id=subject_id,
        start=args.start,
        duration=duration,
        stage_series=stage_series,
        save_path=args.save,
        dpi=args.dpi,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
