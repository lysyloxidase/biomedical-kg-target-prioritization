"""Enforce risk-based per-module coverage gates from coverage.py JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

THRESHOLDS = {
    "src/kgtp/hetero/split_protocol.py": 70.0,
    "src/kgtp/hetero/feature_transformers.py": 80.0,
    "src/kgtp/eval/metrics.py": 90.0,
    "src/kgtp/artifacts.py": 80.0,
    "src/kgtp/api/app.py": 85.0,
    "src/kgtp/pipeline/sample.py": 80.0,
}


def main() -> int:
    report_path = Path(sys.argv[1] if len(sys.argv) > 1 else "coverage.json")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    files = payload.get("files", {})
    failures: list[str] = []
    for filename, threshold in THRESHOLDS.items():
        record = _find_record(files, filename)
        if record is None:
            failures.append(f"{filename}: absent from coverage report")
            continue
        coverage = float(record["summary"]["percent_covered"])
        print(f"{filename}: {coverage:.2f}% (required {threshold:.2f}%)")
        if coverage < threshold:
            failures.append(
                f"{filename}: {coverage:.2f}% is below required {threshold:.2f}%"
            )
    if failures:
        print("Per-module coverage gate failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    return 0


def _find_record(
    files: dict[str, Any],
    expected: str,
) -> dict[str, Any] | None:
    normalized = expected.replace("\\", "/")
    for filename, record in files.items():
        if filename.replace("\\", "/").endswith(normalized):
            return record
    return None


if __name__ == "__main__":
    raise SystemExit(main())
