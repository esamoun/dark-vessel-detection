.PHONY: env install lint test clean

env:
	conda env create -f environment.yml

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	ruff format --check src tests

test:
	pytest -q

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
