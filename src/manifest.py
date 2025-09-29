"""
Genera un manifiesto de sesiones del dataset Sleep-EDFx:
- Busca archivos PSG (.edf) y Hypnograma (.edf) en data/raw/physionet.org/files/sleep-edfx/<version>/<subset>
- Empareja por prefijo (ej.: SC4001E0-PSG.edf con SC4001EC-Hypnogram.edf)
- Emite CSV con columnas: subject_id, subset, version, psg_path, hypnogram_path, status

Uso:
  python src/manifest.py --version 1.0.0 --subset sleep-cassette --raw-root data/raw --out data/processed/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Session:
    subject_id: str
    subset: str
    version: str
    psg_path: Optional[str] = None
    hypnogram_path: Optional[str] = None

    @property
    def status(self) -> str:
        if self.psg_path and self.hypnogram_path:
            return "ok"
        if self.psg_path and not self.hypnogram_path:
            return "missing_hypnogram"
        if self.hypnogram_path and not self.psg_path:
            return "missing_psg"
        return "empty"


def _canonical_key(basename: str) -> str:
    """Devuelve una clave canónica para emparejar PSG e Hypnograma.

    Para Sleep-EDFx, los pares se distinguen por un sufijo (E0 vs EC, J0 vs JC),
    mientras que los primeros 7 caracteres suelen ser comunes, por ejemplo:
        - SC4001E0-PSG.edf  y  SC4001EC-Hypnogram.edf  -> prefijo 'SC4001E'
        - ST7011J0-PSG.edf  y  ST7011JC-Hypnogram.edf  -> prefijo 'ST7011J'
    """
    if len(basename) >= 7 and (basename.startswith("SC") or basename.startswith("ST")):
        return basename[:7]
    # Fallback: usar el nombre completo antes del guión
    return basename.split("-")[0]


def scan_sessions(raw_root: str, version: str, subset: str) -> List[Session]:
    base = os.path.join(
        raw_root, "physionet.org", "files", "sleep-edfx", version, subset
    )
    sessions: Dict[str, Session] = {}
    if not os.path.isdir(base):
        return []

    for entry in os.listdir(base):
        if not entry.endswith(".edf"):
            continue
        path = os.path.join(base, entry)
        base_name = entry.rsplit(".", 1)[0]
        key = _canonical_key(base_name)
        sess = sessions.setdefault(
            key, Session(subject_id=key, subset=subset, version=version)
        )
        if "-PSG" in entry:
            sess.psg_path = path
        elif "-Hypnogram" in entry:
            sess.hypnogram_path = path
    return list(sessions.values())


def write_manifest(sessions: List[Session], out_csv: str) -> None:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["subject_id", "subset", "version", "psg_path", "hypnogram_path", "status"]
        )
        for s in sessions:
            writer.writerow(
                [
                    s.subject_id,
                    s.subset,
                    s.version,
                    s.psg_path or "",
                    s.hypnogram_path or "",
                    s.status,
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generar manifest de sesiones Sleep-EDFx"
    )
    parser.add_argument(
        "--raw-root", default="data/raw", help="Directorio raíz de datos crudos"
    )
    parser.add_argument("--version", default="1.0.0", help="Versión del dataset")
    parser.add_argument(
        "--subset",
        choices=["sleep-cassette", "sleep-telemetry"],
        default="sleep-cassette",
        help="Subset a escanear",
    )
    parser.add_argument(
        "--out", default="data/processed/manifest.csv", help="Ruta de salida del CSV"
    )

    args = parser.parse_args()

    sessions = scan_sessions(args.raw_root, args.version, args.subset)
    write_manifest(sessions, args.out)

    ok = sum(1 for s in sessions if s.status == "ok")
    missing_h = sum(1 for s in sessions if s.status == "missing_hypnogram")
    missing_p = sum(1 for s in sessions if s.status == "missing_psg")
    print(
        f"Sesiones total: {len(sessions)} | ok: {ok} | missing_hypnogram: {missing_h} | missing_psg: {missing_p}"
    )
    print(f"CSV generado en: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
