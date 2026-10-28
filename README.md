# deltacells

[![build](https://github.com/cellarium-ai/deltacells/actions/workflows/ci.yml/badge.svg)](https://github.com/cellarium-ai/deltacells/actions/workflows/ci.yml)

**Fast, compact tiles of single-cell count matrices, built for feeding a training `DataLoader`.**

**Documentation: https://cellarium-ai.github.io/deltacells/**

Training a model on tens of millions of cells means reading raw counts over and over, and the format they are stored in decides
how fast that goes and how much disk it takes. deltacells stores a count matrix as a directory (or bucket prefix) of *tiles* of
about 10,000 cells. Each tile is one self-contained object that a small C library decodes straight into CSR arrays.

## Why deltacells

* **Very fast data loading.** A 10,000-cell tile decodes in about 0.1 s on one core and 0.02 s on eight, with no intermediate
  AnnData or Arrow objects. A PyTorch `DataLoader` reading shuffled batches on a laptop sustains 45,000 to 95,000 cells per
  second, depending on the number of workers.
* **Small on disk.** A dataset is about 40% smaller than the same data in [BPCells](https://github.com/bnprks/BPCells) (3.8 vs
  6.7 kB per cell) and roughly a third of the size of a gzip-compressed h5ad file. The key idea is to store genes in order of
  decreasing total counts, which makes the values that remain to be compressed small and repetitive.
* **Fast enough to keep a GPU fed.** On a single thread, decoding is slower than BPCells' decoder (about 2.6 times as long per
  tile), in exchange for the smaller files. With a few `DataLoader` workers, each decoding on its own core, deltacells is
  designed to keep a GPU busy, as in the data loading of [cellarium-ml](https://github.com/cellarium-ai/cellarium-ml). Smaller
  files also mean less to download when the dataset lives in a bucket.

One real 10,000-cell shard (38,094 genes), counts only, read from a local file into memory on an M2 Pro laptop:

| format | size on disk | decode, 1 thread | decode, 8 threads |
|---|---|---|---|
| h5ad (gzip) | 12.5 kB per cell | 1170 ms | single-threaded reader |
| BPCells (C++ decoder) | 6.7 kB per cell | **48 ms** | **30 ms** |
| **deltacells** | **3.8 kB per cell** | 125 ms | 39 ms |

The [benchmarks page](https://cellarium-ai.github.io/deltacells/benchmarks.html) has the details, other hardware, the DataLoader
throughput, and how to reproduce all of it.

**Status: alpha.** CI runs on Linux x86-64 (Python 3.10 to 3.13) and macOS arm64 (Python 3.12). Reading tiles from Google Cloud
Storage and converting h5ad files from and to a bucket have been exercised against real buckets. See
[Limitations](#limitations).

## Install

```bash
pip install deltacells
```

Prebuilt wheels are available for Linux (x86-64 and aarch64) and macOS (arm64) on Python 3.10 to 3.13. On other platforms, pip
builds the package from source, which needs a C compiler and libzstd. Optional extras: `h5ad` (to convert h5ad files), `obs` (cell
metadata), `torch` and `gcs` (`gs://` paths). For example, `pip install "deltacells[h5ad,obs,gcs]"`. See the
[install page](https://cellarium-ai.github.io/deltacells/install.html).

## Quick start

Convert h5ad shards of any size into a dataset of exact-size tiles (this sorts the genes, re-chunks the cells and compresses):

```bash
deltacells convert --h5ad-glob "shards/*.h5ad" --output /data/my_dataset
deltacells info /data/my_dataset
deltacells verify /data/my_dataset          # checks the CRC of every tile and decodes it
```

The input and the output can also be `gs://` paths. All files must have the same genes, and `X` must hold raw integer counts.
The [conversion page](https://cellarium-ai.github.io/deltacells/convert.html) covers the options, Google Cloud Storage, disk
space and safety.

Read it:

```python
import numpy as np
from deltacells import open_dataset

ds = open_dataset("/data/my_dataset")  # or "gs://bucket/prefix"
ds.n_cells, ds.n_genes, ds.limits  # limits: cumulative number of cells per tile
batch = ds.get_batch(np.array([5, 99_000, 12]))  # any global cell indices, in any order; returns a CSR SparseBatch
batch.indptr, batch.indices, batch.values  # int64, int32, float32 numpy arrays
x = batch.to_torch_csr()  # torch.sparse_csr tensor sharing memory (needs torch)
batch.to_scipy()
ds.var_names, ds.gene_order  # gene names, and the original column of each stored gene
```

Write your own data:

```python
from deltacells import DatasetWriter

with DatasetWriter("/data/out", n_genes=n_genes, tile_size=10_000, gene_order=order, var_names=names) as w:
    for block in blocks:  # scipy CSR, or anything scipy can convert; integer counts in [0, 2**31 - 1]
        w.add_tile(block)  # every tile but the last must have exactly tile_size cells
```

For single tiles there is a lower-level interface: `encode_tile(matrix) -> bytes`, `write_tile(path, matrix)`,
`Tile(bytes).decode(threads=4)` and `read_tile(path)`.

## Documentation

* [Install](https://cellarium-ai.github.io/deltacells/install.html): wheels, extras, building from source.
* [Converting h5ad files](https://cellarium-ai.github.io/deltacells/convert.html): `deltacells convert` for local and
  `gs://` data.
* [Using a dataset in a PyTorch DataLoader](https://cellarium-ai.github.io/deltacells/dataloader.html): a worked example, and the
  pointer to [cellarium-ml](https://github.com/cellarium-ai/cellarium-ml) for a full PyTorch Lightning setup.
* [Cell metadata (obs) and genes (var)](https://cellarium-ai.github.io/deltacells/obs.html): what is stored, how to read it in
  training, and the local cache.
* [Benchmarks](https://cellarium-ai.github.io/deltacells/benchmarks.html): size, decode speed, comparison with other formats,
  DataLoader throughput, and how to reproduce them with `make benchmark`.
* [How the encoding works](https://cellarium-ai.github.io/deltacells/encoding.html): why sorting the genes makes the data compress
  so well, with figures from a real shard.
* [Format specification](https://cellarium-ai.github.io/deltacells/format.html): the exact layout of a tile and a dataset.

## Cell metadata (obs) and genes (var)

A dataset can also hold the cell metadata (`obs`) and the gene table (`var`), stored beside the tiles so that a loader that needs
no metadata pays nothing for it. The converter stores them from the h5ad files. In training, you choose the obs columns you need
and they are copied once into a local, memory-mapped cache, after which looking up a batch takes about 0.2 ms:

```python
ds = open_dataset("gs://bucket/prefix", obs_columns=["cell_type", "donor_id"])
codes = ds.obs.take(
    cell_indices, ["cell_type", "donor_id"]
)  # integer codes; ds.obs.categories("cell_type") is the vocabulary
ds.var  # the gene table (pandas), in the stored column order
```

See [Cell metadata (obs) and genes (var)](https://cellarium-ai.github.io/deltacells/obs.html) for what is stored, how it is read and
cached, and what it costs.

## Limitations

* **Integer counts only**: integers in [0, 2^31 - 1] (float32 batches are exact up to 2^24). Counts above 65535, and gaps above
  65535 between a cell's consecutive expressed genes (possible with more than 65,536 genes), are stored as exceptions and must be
  rare: the writer refuses a tile with more than `max(1024, nnz / 200)` of either kind, which is not what raw counts look like.
  Sparse feature spaces such as ATAC peaks are out of scope, and so are normalized or float data.
* **Conversion has no resume.** A failed conversion is started again with `--overwrite` (see
  [Converting h5ad files](https://cellarium-ai.github.io/deltacells/convert.html)).
* **Platforms.** Linux and macOS, little-endian hosts only. CI covers Linux x86-64 (Python 3.10 to 3.13) and macOS arm64
  (Python 3.12). Windows is not supported.
* **Obs needs `pyarrow` and `pandas`.** Categorical, numeric, boolean and string columns are supported; exclude others, such as
  datetimes. The ranged reads used to fetch obs from a bucket (`GCSBackend.read_range`) have not been tested against a real
  bucket. The local obs cache is uncompressed Arrow, several times the size of the stored parquet for small-integer columns.
* **Concurrent `get_batch` calls** on one dataset object, from several threads, are serialized by a lock (use one dataset object
  per worker process).
* **Gene order depends on the files sampled for it** (up to `--sort-genes-max-files`, spread evenly over the inputs). The same
  cells split into a different number of files can sort the genes slightly differently when there are more files than that. With
  `--no-sort-genes`, or when every file is used, the result depends only on the cell order and the tile size.
* **The converter does not shuffle.** A tile holds consecutive cells (a tile that spans the end of one file and the start of the
  next mixes just those two). If the input is sorted by donor, tissue or similar, shuffle the cells across files first (for example
  with `annslicer`).
* **Compression time.** Level 19 takes about 5 s per 10,000-cell tile with 8 threads (about 24 s with one chunk). Lower levels
  trade size for speed (level 12: about 1.3 s, level 3: about 0.7 s, at files 6% and 11% larger).
* **Reference loader.** `examples/reference_loader.py` shows how to call the dataset from a `DataLoader`. It is an example, not
  part of the package API, and it does not provide resume-from-checkpoint, DDP padding or `drop_last_indices`; for training, use
  [cellarium-ml](https://github.com/cellarium-ai/cellarium-ml).

## Development

```bash
make test         # the test suite: format validation, round trips and edge cases, gather, dataset (prefetch, caching, pickling,
                  # corruption), obs, backends, h5ad conversion, CLI, torch / DataLoader, benchmark smoke tests
make lint         # ruff
make typecheck    # mypy
make docs         # the documentation (needs the docs extra), built into docs/build/html
make benchmark    # all benchmarks that need no real data, on a synthetic dataset
```

[DEVELOPING.md](https://github.com/cellarium-ai/deltacells/blob/main/DEVELOPING.md) describes building from source, the code
layout, how to change the format, how the documentation is built, and how releases are made.

## License

BSD-3-Clause (see [LICENSE](https://github.com/cellarium-ai/deltacells/blob/main/LICENSE)). deltacells links libzstd, which is
dual-licensed BSD-3-Clause / GPL-2.0 and is used under the BSD license.
