"""Utility to validate raw and processed Sleep-EDFx assets.

This module computes SHA-256 hashes for the downloaded raw dataset using the
reference `SHA256SUMS.txt` file distributed by PhysioNet, optionally converts
`SC-subjects.xls` into CSV for easier consumption, and verifies that processed
artifacts declared in the manifests exist on disk.

It is intended to be used both as a standalone CLI (``python -m src.check_data``)
and programmatically from other scripts such as ``src/download.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

BUFFER_SIZE = 1024 * 1024  # 1 MiB chunks for hashing


@dataclass
class QAResult:
    """Collect messages emitted during QA.

    Errors trigger a non-zero exit status. Warnings are informative and do not
    change the exit code unless ``--strict`` is requested by the caller.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_info(self, message: str) -> None:
        self.infos.append(message)

    def exit_code(self, strict: bool = False) -> int:
        if self.errors:
            return 1
        if strict and self.warnings:
            return 1
        return 0


@dataclass
class HashSummary:
    match: int = 0
    mismatch: int = 0
    missing: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "match": self.match,
            "mismatch": self.mismatch,
            "missing": self.missing,
        }


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_subject_metadata(base_dir: Path, result: QAResult) -> Path | None:
    """Ensure the CSV version of the subject metadata exists.

    Returns the path to the CSV file if it exists or could be generated.
    """

    xls_path = base_dir / "SC-subjects.xls"
    csv_path = base_dir / "SC-subjects.csv"
    if csv_path.exists():
        return csv_path
    if not xls_path.exists():
        result.add_warning(
            f"No metadata file found at {xls_path}. Skipping CSV conversion."
        )
        return None

    try:
        df = pd.read_excel(xls_path)
    except ImportError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "Reading SC-subjects.xls requires the 'xlrd' dependency. "
            "Install it via 'pip install xlrd'"
        ) from exc

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    result.add_info(f"Converted {xls_path} -> {csv_path}")
    return csv_path


def compute_hash_report(
    version_root: Path, report_path: Path, result: QAResult
) -> dict[str, list[str]]:
    """Compute hash report for the dataset version root.

    Parameters
    ----------
    version_root:
        Directory that contains ``SHA256SUMS.txt`` alongside the dataset
        structure (e.g. ``data/raw/.../sleep-edfx/1.0.0``).
    report_path:
        CSV file that will receive the detailed hash entries.
    result:
        QAResult instance to append informational messages.

    Returns
    -------
    A dictionary with the lists ``mismatches`` and ``missing`` that can be used
    by the caller to escalate errors.
    """

    sha_file = version_root / "SHA256SUMS.txt"
    if not sha_file.exists():
        raise FileNotFoundError(f"SHA256SUMS.txt not found under {version_root}")

    rows: list[dict[str, object]] = []
    summary = HashSummary()
    mismatches: list[str] = []
    missing: list[str] = []

    with sha_file.open() as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                expected, rel_path = line.split(maxsplit=1)
            except ValueError:
                result.add_warning(f"Skipping malformed line in SHA file: {raw_line!r}")
                continue
            rel_path = rel_path.strip()
            file_path = version_root / rel_path
            actual_hash: str | None = None
            size_bytes: int | None = None
            status: str
            if not file_path.exists():
                status = "missing"
                missing.append(rel_path)
            else:
                size_bytes = file_path.stat().st_size
                actual_hash = _compute_sha256(file_path)
                if actual_hash == expected:
                    status = "match"
                    summary.match += 1
                else:
                    status = "mismatch"
                    summary.mismatch += 1
                    mismatches.append(rel_path)
            if status == "missing":
                summary.missing += 1

            rows.append(
                {
                    "rel_path": rel_path,
                    "status": status,
                    "expected_sha256": expected,
                    "actual_sha256": actual_hash,
                    "size_bytes": size_bytes,
                }
            )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_path, index=False)
    result.add_info(
        "Hash report saved to "
        f"{report_path} | "
        + ", ".join(f"{k}={v}" for k, v in summary.as_dict().items())
    )
    return {"mismatches": mismatches, "missing": missing}


def check_manifest(paths: Iterable[str], label: str) -> list[str]:
    """Return the subset of paths that do not exist on disk."""

    missing: list[str] = []
    for raw_path in paths:
        if not isinstance(raw_path, str) or not raw_path:
            missing.append(str(raw_path))
            continue
        if not Path(raw_path).exists():
            missing.append(raw_path)
    return missing


def verify_processed_artifacts(
    processed_root: Path,
    result: QAResult,
    manifest_path: Path | None = None,
    manifest_trimmed_path: Path | None = None,
) -> None:
    """Validate manifest CSV files and referenced artifacts."""

    manifest_path = manifest_path or processed_root / "manifest.csv"
    manifest_trimmed_path = (
        manifest_trimmed_path or processed_root / "manifest_trimmed.csv"
    )

    if not manifest_path.exists():
        result.add_warning(
            f"Manifest not found at {manifest_path}. Processed checks skipped."
        )
        return

    manifest_df = pd.read_csv(manifest_path)
    if (manifest_df["status"] != "ok").any():
        bad_rows = manifest_df[manifest_df["status"] != "ok"]
        result.add_warning(
            "Some manifest rows are not marked as ok: "
            + ", ".join(bad_rows["subject_id"].tolist())
        )

    missing_psg = check_manifest(manifest_df["psg_path"], "PSG")
    missing_hyp = check_manifest(manifest_df["hypnogram_path"], "Hypnogram")
    if missing_psg or missing_hyp:
        sample = (missing_psg + missing_hyp)[:5]
        result.add_error(
            "Processed manifest references missing files: " + ", ".join(sample)
        )

    if not manifest_trimmed_path.exists():
        result.add_warning(f"Trimmed manifest not found at {manifest_trimmed_path}.")
        return

    trimmed_df = pd.read_csv(manifest_trimmed_path)
    missing_trim_psg = check_manifest(trimmed_df["psg_trimmed_path"], "Trimmed PSG")
    missing_trim_hyp = check_manifest(
        trimmed_df["hypnogram_trimmed_path"], "Trimmed Hypnogram"
    )
    if missing_trim_psg or missing_trim_hyp:
        sample = (missing_trim_psg + missing_trim_hyp)[:5]
        result.add_error("Trimmed artifacts missing on disk: " + ", ".join(sample))

    if (trimmed_df["trim_duration_sec"] <= 0).any():
        bad = trimmed_df[trimmed_df["trim_duration_sec"] <= 0]["subject_id"].tolist()
        result.add_error(
            "Trimmed manifest has non-positive durations for: " + ", ".join(bad)
        )


def run_checks(
    raw_root: Path,
    processed_root: Path,
    version: str,
    report_path: Path,
    result: QAResult | None = None,
    strict: bool = False,
) -> QAResult:
    """Execute the QA routine and return the aggregated result."""

    result = result or QAResult()
    version_root = raw_root / "physionet.org" / "files" / "sleep-edfx" / version

    if not version_root.exists():
        result.add_error(f"Dataset version directory not found: {version_root}")
        return result

    ensure_subject_metadata(version_root, result)
    try:
        hash_outcome = compute_hash_report(version_root, report_path, result)
    except FileNotFoundError as exc:
        result.add_error(str(exc))
        return result

    if hash_outcome["mismatches"]:
        sample = ", ".join(hash_outcome["mismatches"][:5])
        result.add_error(
            f"Detected {len(hash_outcome['mismatches'])} SHA mismatches. "
            f"Examples: {sample}"
        )
    if hash_outcome["missing"]:
        sample = ", ".join(hash_outcome["missing"][:5])
        result.add_error(
            f"Detected {len(hash_outcome['missing'])} missing files. "
            f"Examples: {sample}"
        )

    verify_processed_artifacts(processed_root, result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Sleep-EDFx assets")
    parser.add_argument(
        "--raw-root",
        default="data/raw",
        type=Path,
        help="Root directory where raw PhysioNet data lives.",
    )
    parser.add_argument(
        "--processed-root",
        default="data/processed",
        type=Path,
        help="Root directory where processed artifacts (manifests, trims) are stored.",
    )
    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Sleep-EDFx dataset version to validate.",
    )
    parser.add_argument(
        "--report",
        default=Path("tmp/raw_sha256_report.csv"),
        type=Path,
        help="CSV file to store the raw hash report.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (non-zero exit status).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_checks(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        version=args.version,
        report_path=Path(args.report),
        strict=args.strict,
    )

    for info in result.infos:
        print(f"INFO: {info}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    for error in result.errors:
        print(f"ERROR: {error}")

    exit_code = result.exit_code(strict=args.strict)
    if exit_code == 0:
        print("QA checks completed successfully.")
    else:
        print("QA checks completed with issues.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
