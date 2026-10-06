.PHONY: install generate bench inspect repro v1 v2 diff dag diversity contamination sample tokenize train train-all train-freeze plot compare check check-hw4 check-hw3 check-hw2 clean distclean

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

train: train-all train-freeze plot

train-all:
	uv run python -m src.train --variant all_layers

train-freeze:
	uv run python -m src.train --variant freeze14

plot:
	uv run python -m src.plot

compare:
	uv run python -m src.compare --variant all_layers

check:
	PYTHONUTF8=1 PYTHONIOENCODING=utf-8 bash tests/check.sh

check-hw4:
	PYTHONUTF8=1 PYTHONIOENCODING=utf-8 bash tests/check_hw4.sh

check-hw3:
	PYTHONUTF8=1 PYTHONIOENCODING=utf-8 bash tests/check_hw3.sh

check-hw2:
	PYTHONUTF8=1 PYTHONIOENCODING=utf-8 bash tests/check_hw2.sh

clean:
	rm -rf models metrics/train_*.json metrics/compare_*.json docs/curves.png docs/compare.md .check_*.log src/__pycache__ tests/__pycache__

distclean: clean
	rm -rf .venv
