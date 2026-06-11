# Contributing

Thank you for helping make this benchmark more reproducible.

## Development

1. Install dependencies with `uv sync`.
2. Run `make test` before opening a pull request.
3. Keep source versions pinned in `configs/sources.yaml`.
4. Document new data sources, licenses, and caveats before ingesting data.
5. Do not add DisGeNET-derived data unless the redistribution policy changes and
   the ADR is updated.

## Data Changes

Data pulls must record the source version, retrieval date, license, raw snapshot
path, normalization yield, and unmapped-ID log. Crosswalk coverage for seed genes
must remain above 90%.

## Code Style

The project uses Ruff formatting/linting and mypy. Keep functions small enough
to test directly with fixture-sized dataframes.
