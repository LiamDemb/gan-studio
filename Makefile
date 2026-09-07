.PHONY: ingest test

export PYTHONPATH := src

ingest:
	python3 scripts/ingest.py --config $(CONFIG)

test:
	python3 -m pytest tests -q
