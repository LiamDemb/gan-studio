.PHONY: ingest train generate catalog interpolate test

export PYTHONPATH := src

ingest:
	python3 scripts/ingest.py --config $(CONFIG)

train:
	python3 scripts/train.py --config $(CONFIG)

generate:
	python3 scripts/generate.py --config $(CONFIG)

catalog:
	python3 scripts/catalog.py --config $(CONFIG)

interpolate:
	python3 scripts/interpolate.py --config $(CONFIG)

test:
	python3 -m pytest tests -q
