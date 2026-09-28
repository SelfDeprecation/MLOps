.PHONY: install generate bench inspect repro v1 v2 diff dag diversity contamination sample tokenize check check-hw3 check-hw2 clean

install:
	uv sync

generate:
	uv run python -m src.generate

bench:
	uv run python -m src.bench

inspect:
	uv run python -m src.inspect_model

repro:
	uv run dvc repro

v1:
	uv run python scripts/set_version.py v1
	uv run dvc repro

v2:
	uv run python scripts/set_version.py v2
	uv run dvc repro

diff:
	uv run dvc metrics diff

dag:
	uv run dvc dag

diversity:
	uv run python -m src.diversity

contamination:
	uv run python scripts/check_contamination.py

sample:
	uv run python -m scripts.make_sample

tokenize:
	uv run python -m src.tokenize_data

check:
	bash tests/check.sh

check-hw3:
	bash tests/check_hw3.sh

check-hw2:
	bash tests/check_hw2.sh

clean:
	rm -rf data/tokenized metrics/tokenize.json docs/tokenize_report.md docs/bench.json docs/report.json out1.txt out2.txt params.yaml.bak params.yaml.orig src/__pycache__ tests/__pycache__
