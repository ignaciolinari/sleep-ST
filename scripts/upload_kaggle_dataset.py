#!/usr/bin/env python3
"""
Sube o versiona el dataset a Kaggle usando el CLI.

Requisitos previos:
- Tener instalado el CLI: `pip install kaggle`
- Tener el token en ~/.kaggle/kaggle.json con permisos 600
- Carpeta lista con manifest y datos, por defecto ~/kaggle_upload_sleep_edf/

Variables de entorno opcionales:
- DATASET_DIR: ruta al folder del dataset (default ~/kaggle_upload_sleep_edf)
- DATASET_SLUG: slug del dataset en Kaggle (default: nombre del folder con '-' en vez de '_')
- DATASET_TITLE: título al crear (default se deriva de DATASET_DATA_DIR)
- DATASET_LICENSE: licencia Kaggle (default "CC-BY-4.0")
- DATASET_MESSAGE: mensaje de versión (default "update")
- DATASET_MANIFEST: nombre del manifest (default "manifest_trimmed_resamp200.csv")
- DATASET_DATA_DIR: carpeta con psg/hypnograms (default "sleep_trimmed_resamp200")
"""

import json
import os
import pathlib
import stat
import subprocess
import sys
from typing import List


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def ensure_kaggle_token() -> str:
    kaggle_json = pathlib.Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        sys.exit(
            "ERROR: No se encuentra ~/.kaggle/kaggle.json. Descarga el token desde Kaggle."
        )
    mode = kaggle_json.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        kaggle_json.chmod(0o600)
    with kaggle_json.open() as f:
        data = json.load(f)
    username = data.get("username")
    if not username:
        sys.exit("ERROR: kaggle.json no contiene 'username'.")
    return username


def ensure_kaggle_cli():
    try:
        run(["kaggle", "--version"], check=True)
    except FileNotFoundError:
        sys.exit(
            "ERROR: No se encontró el comando 'kaggle'. Instala con: pip install kaggle"
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"ERROR: 'kaggle --version' falló: {e.stderr}")


def ensure_dataset_dir(path: pathlib.Path, manifest_name: str, data_dir_name: str):
    if not path.exists():
        sys.exit(f"ERROR: No existe la carpeta del dataset: {path}")
    required = [
        path / manifest_name,
        path / data_dir_name / "psg",
        path / data_dir_name / "hypnograms",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        msg = "\n".join(f" - {m}" for m in missing)
        sys.exit(f"ERROR: Faltan elementos requeridos:\n{msg}")


def ensure_metadata(path: pathlib.Path, title: str, full_id: str, license_name: str):
    meta = path / "dataset-metadata.json"
    payload = {
        "title": title,
        "id": full_id,
        "licenses": [{"name": license_name}],
    }
    meta.write_text(json.dumps(payload, indent=2))
    print(f"[INFO] Metadata escrita en {meta} (id={full_id})")


def dataset_exists(owner: str, slug: str) -> bool:
    try:
        run(["kaggle", "datasets", "status", f"{owner}/{slug}"], check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    ensure_kaggle_cli()
    username = ensure_kaggle_token()

    dataset_dir = pathlib.Path(
        os.environ.get("DATASET_DIR", "~/kaggle_upload_sleep_edf")
    ).expanduser()
    manifest_name = os.environ.get("DATASET_MANIFEST", "manifest_trimmed_resamp200.csv")
    data_dir_name = os.environ.get("DATASET_DATA_DIR", "sleep_trimmed_resamp200")
    dataset_slug = os.environ.get("DATASET_SLUG", dataset_dir.name.replace("_", "-"))
    dataset_title = os.environ.get(
        "DATASET_TITLE",
        "Sleep EDF trimmed 200Hz f32"
        if data_dir_name == "sleep_trimmed_resamp200"
        else f"Sleep EDF {data_dir_name.replace('_', ' ')}",
    )
    dataset_license = os.environ.get("DATASET_LICENSE", "CC-BY-4.0")
    version_message = os.environ.get("DATASET_MESSAGE", "update")
    dir_mode = os.environ.get("DATASET_DIR_MODE", "tar")  # zip | tar

    print(
        f"[INFO] Validando contenido: {manifest_name}, "
        f"{data_dir_name}/psg, {data_dir_name}/hypnograms"
    )
    ensure_dataset_dir(dataset_dir, manifest_name, data_dir_name)
    full_id = f"{username}/{dataset_slug}"
    ensure_metadata(dataset_dir, dataset_title, full_id, dataset_license)

    is_existing = dataset_exists(username, dataset_slug)
    if is_existing:
        print(f"[INFO] Dataset ya existe: {full_id}")
        cmd = [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(dataset_dir),
            "-m",
            version_message,
            "-u",
            "--dir-mode",
            dir_mode,
        ]
    else:
        print(f"[INFO] Creando dataset nuevo: {full_id}")
        # La metadata (dataset-metadata.json) ya contiene title/id/licencia.
        cmd = [
            "kaggle",
            "datasets",
            "create",
            "-p",
            str(dataset_dir),
            "-u",
            "--dir-mode",
            dir_mode,
        ]

    try:
        print(f"[INFO] Ejecutando: {' '.join(cmd)}")
        res = run(cmd, check=True)
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr:
            print(res.stderr.strip())
        print("[OK] Operación completada.")
    except subprocess.CalledProcessError as e:
        print(e.stdout)
        print(e.stderr)
        sys.exit(f"ERROR ejecutando {' '.join(cmd)}")


if __name__ == "__main__":
    main()
