PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
UV ?= uv

.PHONY: setup download data eda baselines metadata-baseline validate test

setup:
	$(UV) sync --extra dev --extra imaging --extra training --locked

# Deliberately fetch only the five CSV metadata files.
download:
	PYTHONPATH=src $(PYTHON) -m rsnaknee.cli download

data:
	PYTHONPATH=src $(PYTHON) -m rsnaknee.cli data

eda:
	PYTHONPATH=src $(PYTHON) -m rsnaknee.eda

baselines:
	PYTHONPATH=src $(PYTHON) -m rsnaknee.cli baselines

metadata-baseline:
	PYTHONPATH=src $(PYTHON) -m rsnaknee.baseline

validate:
	@test -n "$(SUBMISSION)" || (echo "Usage: make validate SUBMISSION=path/to/submission.csv"; exit 2)
	PYTHONPATH=src $(PYTHON) -m rsnaknee.cli validate --submission "$(SUBMISSION)"

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q
