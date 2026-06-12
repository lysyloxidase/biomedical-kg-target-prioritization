# Reproducibility

## Locked Execution

Use `uv sync --frozen --extra dev --no-editable` and
`make reproduce-small`. The production workflow is deliberately unavailable
through `make reproduce`.

## Run Manifest

Every complete sample run updates `artifacts/sample/manifests/run.json` and
preserves the same record at
`artifacts/sample/manifests/runs/<run_id>.json`. Each record includes:

- run ID, timestamps, command, Git commit, and dirty-worktree state;
- Python, platform, dependency-lock, and effective-config hashes;
- source versions and licenses;
- raw and normalized file hashes;
- graph, split, feature, and checkpoint hashes;
- all experiment seeds and final status.

The dirty-worktree flag records reality. A dirty local run is not represented
as a clean release run.

Stage manifests record declared inputs and outputs with byte size and SHA-256.
Training and evaluation validate upstream stage records and run compatibility.
Explanation and API serving additionally require a completed compatible run.

## Determinism

The sample uses deterministic ordering and fixed seeds. Tests verify identical
splits and sampled-unlabeled sets for the same seed and changes for different
seeds. Run IDs and timestamps are intentionally unique and are excluded from
scientific artifact equivalence.

## Remaining Sources Of Variation

Hardware, PyTorch kernels, future dependency releases, optional transformer
downloads, and upstream live-source refreshes can change outputs. Full
production reproducibility has not been demonstrated.
