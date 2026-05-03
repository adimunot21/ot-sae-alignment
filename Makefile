.PHONY: smoke test lint format clean install hooks

install:
	conda env update -f environment.yml --prune

hooks:
	pre-commit install
	nbstripout --install

smoke:
	python scripts/smoke_test.py

test:
	pytest --cov=ot_primitives --cov-report=term-missing

lint:
	ruff check .
	mypy ot_primitives

format:
	ruff format .
	ruff check --fix .

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
