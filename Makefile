UV ?= uv
UV_RUN ?= $(UV) run --no-editable --reinstall-package biomedical-kg-target-prioritization

.PHONY: setup data graph neo4j heterodata splits baselines train ablate explain api smoke-train reproduce reproduce-small clean-sample test lint typecheck format

setup:
	$(UV) sync --extra dev --no-editable
	$(UV_RUN) pre-commit install

data:
	$(UV_RUN) kgtp data

graph:
	$(UV_RUN) kgtp graph

neo4j:
	$(UV_RUN) kgtp neo4j

heterodata:
	$(UV_RUN) kgtp heterodata

splits:
	$(UV_RUN) kgtp splits

baselines:
	$(UV_RUN) kgtp baselines

train:
	$(UV_RUN) kgtp train

ablate:
	$(UV_RUN) kgtp ablate

explain:
	$(UV_RUN) kgtp explain \
		--checkpoint artifacts/sample/models/gnn/hgt/seed_13/best_checkpoint.pt \
		--model-config artifacts/sample/models/gnn/hgt/seed_13/config.json \
		--graph artifacts/sample/features/heterodata.pt \
		--dataset-manifest artifacts/sample/manifests/dataset.json \
		--graph-manifest artifacts/sample/manifests/assemble.json \
		--feature-manifest artifacts/sample/manifests/features.json \
		--node-index-map artifacts/sample/features/node_index_maps.json \
		--split-metadata artifacts/sample/splits/split_metadata.json \
		--validation-metrics artifacts/sample/models/gnn/hgt/seed_13/metrics.json \
		--run-manifest artifacts/sample/manifests/run.json \
		--output-dir artifacts/sample/report/explanations

api:
	$(UV_RUN) uvicorn kgtp.api.app:app --host 0.0.0.0 --port 8000

smoke-train:
	$(UV_RUN) python -m kgtp.cli smoke-train --tiny

reproduce:
	@echo "The full production workflow is not implemented. Use 'make reproduce-small'."
	@exit 2

reproduce-small:
	$(UV_RUN) kgtp reproduce-small

clean-sample:
	$(UV_RUN) kgtp clean-sample

test:
	$(UV_RUN) pytest --cov=kgtp --cov-report=term-missing --cov-report=json:coverage.json --cov-fail-under=80
	$(UV_RUN) python scripts/check_coverage.py coverage.json

lint:
	$(UV_RUN) ruff check .

typecheck:
	$(UV_RUN) mypy src
	$(UV_RUN) pyright src tests

format:
	$(UV_RUN) ruff format .
