.PHONY: ingest train generate catalog interpolate video test check-stylegan2 benchmark

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

video:
	python3 scripts/video.py --config $(CONFIG)

test:
	python3 -m pytest tests -q

check-stylegan2:
	python3 -c 'import torch, triton; assert torch.cuda.is_available(), "CUDA is required: skipped tests are not validation"'
	python3 -m pytest tests/stylegan2/test_conv_grad.py tests/stylegan2/test_cuda.py -m cuda -q

benchmark:
	python3 scripts/benchmark_stylegan2.py --config $(CONFIG) --output $(OUTPUT)
