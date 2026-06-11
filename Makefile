UV ?= python -m uv
UV_RUN ?= $(UV) run --no-editable --reinstall-package biomedical-kg-target-prioritization

.PHONY: setup data graph neo4j heterodata splits baselines train ablate explain api smoke-train reproduce test lint typecheck format

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
	$(UV_RUN) kgtp explain

api:
	$(UV_RUN) uvicorn kgtp.api.app:app --host 0.0.0.0 --port 8000

smoke-train:
	$(UV_RUN) python -m kgtp.cli smoke-train --tiny

reproduce: data graph neo4j heterodata splits baselines train ablate explain smoke-train test

test:
	$(UV_RUN) pytest --cov=kgtp --cov-report=term-missing --cov-fail-under=85

lint:
	$(UV_RUN) ruff check .

typecheck:
	$(UV_RUN) mypy src
	$(UV_RUN) pyright src tests

format:
	$(UV_RUN) ruff format .
