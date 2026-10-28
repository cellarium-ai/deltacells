.PHONY: install uninstall lint docs format typecheck test test-dataloader benchmark FORCE

install: FORCE
	pip install -e .[dev]

uninstall: FORCE
	pip uninstall deltacells

lint: FORCE
	ruff check .
	ruff format --check .

docs: FORCE
	cd docs && make html

format: FORCE
	ruff check --fix .
	ruff format .

typecheck: FORCE
	mypy src tests examples benchmarks

test: FORCE
	pytest -v tests

test-dataloader: FORCE
	python examples/dataloader_example.py

# Run all the benchmarks that need no external data on a synthetic dataset (see DEVELOPING.md for what each one measures).
# Run it on an otherwise idle machine; the pipeline benchmark needs a few GB of free RAM per worker. Override the settings on
# the command line, for example:  make benchmark BENCH_CELLS=20000 BENCH_GENES=10000 BENCH_THREADS=1,2,4
#   BENCH_DIR      where the synthetic dataset is written (replaced on every run)
#   BENCH_CELLS, BENCH_GENES, BENCH_NNZ   size of the dataset: cells, genes, mean nonzeros per cell
#   BENCH_THREADS  decode threads to compare;  BENCH_WORKERS  DataLoader workers to compare
# Not included, because they need real data or other tools: benchmarks/compare_formats.py (one h5ad shard; optionally TileDB-SOMA
# and BPCells) and benchmarks/check_against_h5ad.py (a dataset converted from h5ad files, against those files).
BENCH_DIR ?= /tmp/deltacells_bench
BENCH_CELLS ?= 50000
BENCH_GENES ?= 30000
BENCH_NNZ ?= 3000
BENCH_THREADS ?= 1,2,4,8
BENCH_WORKERS ?= 0,1,2,4

benchmark: FORCE
	python benchmarks/make_synthetic.py $(BENCH_DIR) --n-cells $(BENCH_CELLS) --n-genes $(BENCH_GENES) --mean-nnz $(BENCH_NNZ) --obs-columns 20 --overwrite
	python benchmarks/bench_decode.py $(BENCH_DIR) --threads $(BENCH_THREADS)
	python benchmarks/bench_compression.py $(BENCH_DIR) --levels 3,12,19 --chunks 1,8
	python benchmarks/bench_obs.py $(BENCH_DIR)
	python benchmarks/bench_pipeline.py $(BENCH_DIR) --workers $(BENCH_WORKERS)
	python benchmarks/bench_pipeline.py $(BENCH_DIR) --workers 2 --bandwidth-mbps 150 --latency-ms 50  # emulate a remote store

FORCE:
