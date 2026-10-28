.PHONY: install uninstall lint docs format typecheck test test-dataloader FORCE

install: FORCE
	pip install -e .[dev]

uninstall: FORCE
	pip uninstall deltacells

lint: FORCE
	ruff check .
	ruff format --check .

docs: FORCE
	cd docs && make html

format: license FORCE
	ruff check --fix .
	ruff format .

typecheck: FORCE
	mypy src tests examples benchmarks

test: FORCE
	pytest -v tests

test-dataloader: FORCE
	python -m examples/dataloader_example.py

FORCE:
