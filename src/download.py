"""
Descargador del dataset Sleep-EDF (Sleep-EDFx) desde PhysioNet.

Dos métodos soportados:
  - wget: descarga recursiva por URL (permite elegir subset). Recomendado para controlar exactamente qué bajar.
  - wfdb: usa la función wfdb.dl_database para descargar la base completa.

Credenciales (si el dataset requiere login):
  - Variables de entorno: PHYSIONET_USERNAME y PHYSIONET_PASSWORD
  - Flags: --username y --password (se recomienda usar sólo --username y que el script pida el password)

Ejemplos:
  # Descarga sólo sleep-cassette con wget a data/raw
  python src/download.py --method wget --subset sleep-cassette --out data/raw

  # Descarga todo con wfdb
  python src/download.py --method wfdb --out data/raw

Más info del dataset:
  - https://physionet.org/content/sleep-edfx/1.0.0/
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
from pathlib import Path


def clean_previous_data(raw_root: str, processed_root: str, version: str) -> None:
    """Remove prior raw dataset, processed outputs and download logs."""

    raw_path = Path(raw_root)
    sleep_root = raw_path / "physionet.org" / "files" / "sleep-edfx"
    version_root = sleep_root / version
    if version_root.exists():
        shutil.rmtree(version_root)
        print(f"Eliminado directorio de versión raw: {version_root}")
    elif sleep_root.exists():
        shutil.rmtree(sleep_root)
        print(f"Eliminado árbol de datos raw: {sleep_root}")

    processed_path = Path(processed_root)
    if processed_path.exists():
        shutil.rmtree(processed_path)
        print(f"Eliminado directorio procesado: {processed_path}")

    qa_report = Path("tmp/raw_sha256_report.csv")
    if qa_report.exists():
        qa_report.unlink()
        print(f"Eliminado reporte previo: {qa_report}")

    for log_path in Path(".").glob("wget-log*"):
        if log_path.is_file():
            log_path.unlink()
            print(f"Eliminado log: {log_path}")


def _load_check_data_module():
    """Importar el módulo de QA sin importar el modo de ejecución."""

    try:  # Ejecutando como paquete (python -m src.download)
        from . import check_data as qa_module  # type: ignore
    except ImportError:  # Ejecutando como script (python src/download.py)
        import check_data as qa_module  # type: ignore
    return qa_module


def build_wget_command(
    base_url: str,
    out_dir: str,
    username: str | None = None,
    password: str | None = None,
) -> list[str]:
    """Construye el comando wget para descarga recursiva.

    -r: recursivo
    -N: respeta timestamps para reintentos idempotentes
    -c: continua descargas interrumpidas
    -np: no subir al directorio padre
    -e robots=off: desactiva restricciones de robots
    -P: directorio de salida base
    """
    cmd = [
        "wget",
        "-r",
        "-N",
        "-c",
        "-np",
        "-e",
        "robots=off",
        "-P",
        out_dir,
        base_url,
    ]

    # Autenticación básica si se provee (PhysioNet la soporta para descargas tras aceptar términos)
    if username:
        cmd.extend(["--user", username])
        # Sólo añadimos --password si viene explícito para evitar mostrarlo en history.
        if password:
            cmd.extend(["--password", password])
    return cmd


def run_wget(
    subset: str,
    version: str,
    out_dir: str,
    username: str | None,
    password: str | None,
    dry_run: bool,
) -> int:
    if subset not in {"sleep-cassette", "sleep-telemetry", "all"}:
        raise ValueError("subset debe ser 'sleep-cassette', 'sleep-telemetry' o 'all'")

    # Construimos URL base(s)
    base = f"https://physionet.org/files/sleep-edfx/{version}/"
    urls = [base]
    if subset != "all":
        urls = [base + f"{subset}/"]

    # Asegurar directorio de salida
    os.makedirs(out_dir, exist_ok=True)

    # Resolver ruta a wget: primero PATH, luego carpeta del intérprete (útil en conda envs)
    wget_path = shutil.which("wget")
    if not wget_path:
        candidate = os.path.join(os.path.dirname(sys.executable), "wget")
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            wget_path = candidate
    if not wget_path:
        print(
            "ERROR: 'wget' no está instalado o no está en PATH. Instálalo con conda (conda install -n sleep-st -c conda-forge wget) o Homebrew (brew install wget).",
            file=sys.stderr,
        )
        return 1

    exit_code = 0
    for url in urls:
        cmd = build_wget_command(url, out_dir, username=username, password=password)
        # Forzar uso de la ruta localizada a wget (primer elemento del comando)
        cmd[0] = wget_path
        print("Ejecutando:", " ".join(cmd))
        if dry_run:
            continue
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            exit_code = proc.returncode
            break
    return exit_code


def run_wfdb(out_dir: str, dry_run: bool) -> int:
    try:
        import wfdb
    except Exception as e:  # noqa: BLE001
        print(
            "ERROR: No se pudo importar wfdb. Instálalo con conda o pip.",
            file=sys.stderr,
        )
        print(f"Detalle: {e}", file=sys.stderr)
        return 1

    os.makedirs(out_dir, exist_ok=True)
    print("Descargando con wfdb.dl_database('sleep-edfx') al directorio:", out_dir)
    if dry_run:
        return 0
    try:
        # Nota: wfdb descarga la base completa. Si necesitas sólo un subset, usa --method wget.
        wfdb.dl_database("sleep-edfx", dl_dir=out_dir)
        return 0
    except Exception as e:  # noqa: BLE001
        print("ERROR durante la descarga con wfdb:", e, file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Descargar Sleep-EDFx desde PhysioNet")
    parser.add_argument(
        "--method",
        choices=["wget", "wfdb"],
        default="wget",
        help="Método de descarga (wget: control fino por subset; wfdb: descarga base completa)",
    )
    parser.add_argument(
        "--subset",
        choices=["sleep-cassette", "sleep-telemetry", "all"],
        default="sleep-cassette",
        help="Subset a descargar (sólo aplica a método wget)",
    )
    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Versión del dataset en PhysioNet (por ejemplo, 1.0.0)",
    )
    parser.add_argument(
        "--out",
        default="data/raw",
        help="Directorio de salida donde se guardarán los archivos",
    )
    parser.add_argument(
        "--processed-root",
        default="data/processed",
        help="Directorio donde se ubican los artefactos procesados (para limpieza/QA)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Eliminar datos previos (raw, procesados y logs) antes de descargar",
    )
    parser.add_argument(
        "--username",
        default=os.environ.get("PHYSIONET_USERNAME"),
        help="Usuario de PhysioNet (si es necesario). También puedes setear PHYSIONET_USERNAME",
    )
    parser.add_argument(
        "--password",
        default=None,
        help="Password de PhysioNet (no recomendado via flag). Si se omite y --username está definido, se pedirá interactivamente. También puedes setear PHYSIONET_PASSWORD",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No descarga, sólo muestra lo que haría",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Omitir chequeo de QA post-descarga",
    )
    parser.add_argument(
        "--qa-report",
        default="tmp/raw_sha256_report.csv",
        help="Ruta del CSV de reporte QA (se recrea en cada ejecución)",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Tratar warnings de QA como errores (exit code != 0)",
    )

    args = parser.parse_args(argv)

    if args.clean and not args.dry_run:
        print("Limpiando datos previos antes de la descarga...")
        clean_previous_data(args.out, args.processed_root, args.version)

    # Normalizar password
    password = args.password or os.environ.get("PHYSIONET_PASSWORD")
    if args.method == "wget" and args.username and not password and not args.dry_run:
        # Pedir password de forma segura si hace falta y no es dry-run
        try:
            password = getpass.getpass("Password de PhysioNet: ")
        except (EOFError, KeyboardInterrupt):
            print("No se ingresó password. Abortando.", file=sys.stderr)
            return 1

    if args.method == "wget":
        url_info = f"subset={args.subset}, version={args.version}"
        print(f"Método: wget | {url_info}")
        exit_code = run_wget(
            subset=args.subset,
            version=args.version,
            out_dir=args.out,
            username=args.username,
            password=password,
            dry_run=args.dry_run,
        )
    else:
        print("Método: wfdb (descargará la base completa)")
        if args.subset != "all":
            print(
                "Aviso: --subset no aplica con wfdb; usa --method wget si necesitas filtrar."
            )
        exit_code = run_wfdb(out_dir=args.out, dry_run=args.dry_run)

    if exit_code != 0 or args.dry_run:
        return exit_code

    if args.skip_validation:
        print("Chequeo de QA omitido (--skip-validation).")
        return exit_code

    qa_module = _load_check_data_module()
    qa_result = qa_module.run_checks(
        raw_root=Path(args.out),
        processed_root=Path(args.processed_root),
        version=args.version,
        report_path=Path(args.qa_report),
        strict=args.strict_validation,
    )

    for info in qa_result.infos:
        print(f"INFO: {info}")
    for warning in qa_result.warnings:
        print(f"WARNING: {warning}")
    for error in qa_result.errors:
        print(f"ERROR: {error}")

    qa_exit = qa_result.exit_code(strict=args.strict_validation)
    if qa_exit == 0:
        print("Validación post-descarga OK.")
    else:
        print("Validación post-descarga con errores.", file=sys.stderr)
    return qa_exit


if __name__ == "__main__":
    sys.exit(main())
