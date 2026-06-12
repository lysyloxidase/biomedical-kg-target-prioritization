"""Run-level provenance manifests for executable pipelines."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class RunManifestError(RuntimeError):
    """Raised when run provenance is missing or incompatible."""


REQUIRED_RUN_FIELDS = {
    "run_id",
    "git_commit",
    "dirty_worktree",
    "created_at",
    "python_version",
    "platform",
    "dependency_lock_sha256",
    "config_sha256",
    "source_versions",
    "source_licenses",
    "raw_file_hashes",
    "normalized_file_hashes",
    "graph_hash",
    "split_hash",
    "feature_hash",
    "checkpoint_hash",
    "seeds",
    "command",
    "status",
}


def start_run_manifest(
    path: Path,
    *,
    sample_manifest_path: Path,
    config_path: Path,
    seed: int,
    max_epochs: int,
    command: str,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Create a running manifest before any pipeline stage executes."""
    root = (repository_root or Path.cwd()).resolve()
    source_manifest = _load_json(sample_manifest_path)
    sources = source_manifest.get("sources", {})
    if not isinstance(sources, dict):
        raise RunManifestError("Sample source manifest has no source mapping")
    effective_config = {
        "config_file_sha256": sha256_file(config_path),
        "seed": seed,
        "max_epochs": max_epochs,
    }
    created_at = datetime.now(UTC).isoformat()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        ),
        "git_commit": _git_output(root, ["rev-parse", "HEAD"]) or "unknown",
        "dirty_worktree": bool(_git_output(root, ["status", "--porcelain"])),
        "created_at": created_at,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependency_lock_sha256": sha256_file(root / "uv.lock"),
        "config_sha256": canonical_sha256(effective_config),
        "effective_config": effective_config,
        "source_versions": {
            name: str(spec.get("release", "unknown"))
            for name, spec in sorted(sources.items())
            if isinstance(spec, dict)
        },
        "source_licenses": {
            name: str(spec.get("license", "unknown"))
            for name, spec in sorted(sources.items())
            if isinstance(spec, dict)
        },
        "raw_file_hashes": {
            file_path.name: sha256_file(file_path)
            for file_path in sorted(sample_manifest_path.parent.glob("*.parquet"))
        },
        "normalized_file_hashes": {},
        "graph_hash": "",
        "split_hash": "",
        "feature_hash": "",
        "checkpoint_hash": "",
        "checkpoint_hashes": {},
        "seeds": [seed],
        "command": command,
        "status": "running",
    }
    write_run_manifest(path, payload)
    return payload


def update_run_manifest(path: Path, **updates: Any) -> dict[str, Any]:
    """Update an existing run manifest without discarding provenance."""
    payload = load_run_manifest(path)
    payload.update(updates)
    write_run_manifest(path, payload)
    return payload


def complete_run_manifest(path: Path) -> dict[str, Any]:
    """Mark a fully populated run manifest completed."""
    payload = load_run_manifest(path)
    for field in ("graph_hash", "split_hash", "feature_hash", "checkpoint_hash"):
        if not payload.get(field):
            raise RunManifestError(f"Cannot complete run manifest without {field}")
    if not payload.get("normalized_file_hashes"):
        raise RunManifestError(
            "Cannot complete run manifest without normalized file hashes"
        )
    payload["status"] = "completed"
    payload["completed_at"] = datetime.now(UTC).isoformat()
    write_run_manifest(path, payload)
    return payload


def fail_run_manifest(path: Path, exc: BaseException) -> None:
    """Record a failed run without masking the original exception."""
    if not path.is_file():
        return
    payload = load_run_manifest(path)
    payload.update(
        {
            "status": "failed",
            "completed_at": datetime.now(UTC).isoformat(),
            "error": f"{type(exc).__name__}: {exc}",
        }
    )
    write_run_manifest(path, payload)


def load_run_manifest(path: Path) -> dict[str, Any]:
    """Load and validate the run-manifest schema."""
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunManifestError(f"Cannot read run manifest at {path}: {exc}") from exc
    missing = REQUIRED_RUN_FIELDS.difference(payload)
    if missing:
        raise RunManifestError(f"Run manifest is missing fields: {sorted(missing)}")
    if payload["status"] not in {"running", "completed", "failed"}:
        raise RunManifestError(f"Invalid run status: {payload['status']!r}")
    return payload


def validate_run_compatibility(
    path: Path,
    *,
    graph_hash: str,
    split_hash: str,
    feature_hash: str,
    checkpoint_hash: str | None = None,
    allowed_statuses: Sequence[str] = ("running", "completed"),
) -> dict[str, Any]:
    """Validate that a run manifest identifies the supplied artifacts."""
    payload = load_run_manifest(path)
    if payload["status"] not in allowed_statuses:
        raise RunManifestError(
            f"Run status {payload['status']!r} is not allowed for this operation"
        )
    expected = {
        "graph_hash": graph_hash,
        "split_hash": split_hash,
        "feature_hash": feature_hash,
    }
    if checkpoint_hash is not None:
        expected["checkpoint_hash"] = checkpoint_hash
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RunManifestError(
                f"Run manifest mismatch for {field}: "
                f"expected {value!r}, got {payload.get(field)!r}"
            )
    return payload


def hashes_for_files(paths: Sequence[Path]) -> dict[str, str]:
    """Return deterministic path-to-SHA mappings."""
    return {path.as_posix(): sha256_file(path) for path in sorted(paths)}


def sha256_file(path: Path) -> str:
    """Hash one file using SHA-256."""
    if not path.is_file():
        raise RunManifestError(f"Required provenance file is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping independent of key ordering."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_run_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    """Write the current manifest and preserve its run-id-specific history."""
    _write_json_atomic(path, payload)
    run_id = payload.get("run_id")
    if path.name == "run.json" and isinstance(run_id, str) and run_id:
        _write_json_atomic(path.parent / "runs" / f"{run_id}.json", payload)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def default_command(seed: int, max_epochs: int) -> str:
    """Return the canonical command recorded for direct Python callers."""
    if sys.argv and "pytest" not in Path(sys.argv[0]).name:
        return " ".join(sys.argv)
    return f"kgtp reproduce-small --seed {seed} --max-epochs {max_epochs}"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RunManifestError(f"Expected a JSON object in {path}")
    return payload


def _git_output(root: Path, arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip()
