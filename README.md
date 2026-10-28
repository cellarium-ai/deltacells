# deltacells

[![build](https://github.com/cellarium-ai/deltacells/actions/workflows/ci.yml/badge.svg)](https://github.com/cellarium-ai/deltacells/actions/workflows/ci.yml)

Fast, compact, cloud-friendly **tiles of single-cell count matrices**, designed for feeding a training `DataLoader`.

A deltacells dataset is a directory (or bucket prefix) of *tiles*. Each tile is a self-contained object holding about 10,000 cells
by all genes. Within a tile, the counts are stored as per-cell **gene gaps** (delta coding of the sorted gene indices) and
**counts**, as byte-planar uint16 streams compressed with **zstd**. The rare values that do not fit in a uint16 are stored
alongside them. A small C core decompresses a 10,000-cell tile straight into CSR arrays in about 0.11 s on one core (0.02 s on
8 cores; Xeon 2.3 GHz), without any intermediate AnnData or Arrow objects. Genes are sorted by decreasing total counts, which is
what makes the compression effective.

**Status: alpha.** CI runs on Linux x86-64 (Python 3.10 to 3.13) and macOS arm64 (Python 3.12). Reading tiles from Google Cloud
Storage and converting h5ad files from and to a bucket have been exercised against real buckets. See
[Limitations](#limitations).

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Performance](#performance)
- [Converting h5ad files](#converting-h5ad-files)
- [Using the dataset in a DataLoader](#using-the-dataset-in-a-dataloader)
- [Cell metadata (obs)](#cell-metadata-obs)
- [How it works](#how-it-works)
- [Limitations](#limitations)
- [Development](#development)

## Install

```bash
pip install deltacells
```

Prebuilt wheels are available for Linux (x86-64 and aarch64) and macOS (arm64) on Python 3.10 to 3.13. They include libzstd.
On other platforms, pip builds the package from source, which needs a C compiler and **libzstd** (headers and library):

```bash
conda install zstd                      # or: brew install zstd / apt install libzstd-dev
pip install deltacells
```

Optional extras:

| extra | adds | needed for |
|---|---|---|
| `torch` | PyTorch | `SparseBatch.to_torch_csr()`, the example data loaders |
| `h5ad` | anndata | `deltacells convert` |
| `obs` | pyarrow, pandas | cell metadata (obs) and the gene table (var) |
| `gcs` | google-cloud-storage, fsspec, gcsfs | reading, and converting from and to, `gs://` paths |

For example, `pip install "deltacells[h5ad,obs,gcs]"`. To build from a clone of the repository, see
[DEVELOPING.md](https://github.com/cellarium-ai/deltacells/blob/main/DEVELOPING.md).

## Quick start

Convert h5ad shards of any size into a dataset of exact-size tiles (this sorts the genes, re-chunks the cells and compresses):

```bash
deltacells convert --h5ad-glob "shards/*.h5ad" --output /data/my_dataset --tile-size 10000
                                            # h5ad files and output can be gs:// paths too: see "Converting h5ad files"
deltacells info /data/my_dataset
deltacells verify /data/my_dataset          # checks the CRC of every tile and decodes it
```

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

Cell metadata (obs) and the gene table (var), if the dataset has them (needs `pyarrow` and `pandas`; see
[Cell metadata](#cell-metadata-obs)):

```python
ds = open_dataset("gs://bucket/prefix", obs_columns=["cell_type", "donor_id"])  # copies just these columns to the local cache, once
ds.obs.columns  # every column and its kind; ds.obs.categories("cell_type") is the vocabulary
codes = ds.obs.take(batch_cell_indices, ["cell_type", "donor_id"])  # {"cell_type": int codes, ...}: local, ~0.2 ms per 5000 cells
ds.obs.localize(["age"])  # add more columns at any time (only their bytes are read)
df = ds.obs.to_pandas(["cell_type", "age"])  # whole-dataset DataFrame, for example for plotting
ds.var  # the gene table (pandas), in column order
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

## Performance

### Current numbers (Linux x86-64)

One real shard (10,000 cells x 61,497 genes, human, 2,309 nonzeros per cell), **X only**, on a Google Cloud VM (n1-standard-16:
Xeon 2.3 GHz, 8 cores / 16 vCPUs), local disk, warm page cache, no network. Reproduce with
`python benchmarks/compare_formats.py shard.h5ad --only h5ad,deltacells`:

| format | stored X | kB/cell | file -> RAM, 1 thread | 8 threads |
|---|---|---|---|---|
| h5ad (as is: original gene order, gzip) | 55.6 MB | 5.56 | 981 ms | n/a (single-threaded reader) |
| **deltacells** (zstd 19), reused buffers | **18.9 MB** | **1.89** | 118 ms | 26 ms |
| deltacells (zstd 19), fresh arrays per call | 18.9 MB | 1.89 | 164 ms | 35 ms |

Decoding alone (already in memory; `benchmarks/bench_decode.py`, int32 values, reused buffers). The results are the same for the
three tiles of this dataset, one of which has no counts above 65535 and two of which do, so the exception frames cost nothing
measurable:

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| decode a 10k-cell tile | 108 ms | 57 ms | 32 ms | 20 ms |
| cells / s | 93k | 175k | 312k | 497k |

Gathering 5,000 shuffled cells from a decoded tile takes 36 ms. The C core is built with `-fno-wrapv`; CPython's default
`-fwrapv` made the same code about 12% slower.

### Comparison with other formats (M2 Pro laptop)

One real shard (10,000 cells x 38,094 genes, snRNA-seq, 5,171 nonzeros per cell), **X only** (no obs or var), written in each
format and read from a local file back into RAM on an M2 Pro laptop (warm page cache, no network). TileDB-SOMA and BPCells are not
installed on the Linux machine above, so this comparison has not been repeated there, and its deltacells rows come from a build
that predates the exception format and the `-fno-wrapv` change. Reproduce with `python benchmarks/compare_formats.py shard.h5ad`
(the BPCells C++ row needs `benchmarks/bpcells_decode/build.sh` once):

| format | stored X | kB/cell | file -> RAM, 1 thread | 8 threads |
|---|---|---|---|---|
| h5ad (as is: original gene order, gzip) | 124.6 MB | 12.46 | 1170 ms | n/a (single-threaded reader) |
| TileDB-SOMA (tuned layout), -> CSR | 56.7 MB | 5.67 | 1361 ms | 394 ms |
| BPCells, Python bindings | 67.4 MB | 6.74 | 1234 ms | 275 ms |
| BPCells, C++ decoder (uint32 values, preallocated) | 67.4 MB | 6.74 | **48 ms** | **30 ms** |
| **deltacells** (zstd 19), reused buffers | **38.4 MB** | **3.84** | 125 ms | 39 ms |
| deltacells (zstd 19), fresh arrays per call | 38.4 MB | 3.84 | 148 ms | 56 ms |

Notes on reading the table:

* **BPCells' decoder is the fastest reader here, by a wide margin over its own bindings.** Calling BPCells' BP-128 decoders
  directly (the harness in `benchmarks/bpcells_decode/`, which downloads and builds against a pinned BPCells commit) takes 48 ms
  for a tile, including reading its files. The Python bindings take 1.2 s because they build the result through an Eigen sparse
  matrix. The harness is a measurement tool, not a supported reader: it leaves values as uint32 and decodes into preallocated
  arrays.
* **deltacells trades decode CPU for size.** Its files are 43% smaller than BPCells' (38.4 vs 67.4 MB) and it needs about 2.6x
  the decode time (and CPU time) per tile. Which one wins end to end depends on where the tiles live. When fetching a tile is
  slower than decoding it, the smaller file wins: with fetch and decode overlapped, deltacells' tile time is lower below roughly
  500 MB/s of per-worker bandwidth (one decode thread each), and the break-even is lower (about 400 MB/s) when they are not
  overlapped. These crossovers are estimates computed from the numbers above, not measurements. With fast local storage and few
  cores to spare, BPCells' decoder is the cheaper reader.
* The sorted formats (TileDB-SOMA, BPCells, deltacells) share the same gene order (genes sorted by decreasing total counts of this
  shard). The TileDB-SOMA layout is the best-performing one tried: tile extent equal to the shard, row-major, delta + zstd 3 on
  the gene coordinate, byte-shuffle + zstd 3 on the values. The h5ad file is left as users have it. Every row includes reading
  its file from disk.
* The times are for X alone; obs is a separate concern. The h5ad file as a whole is 148 MB, of which X is 124.6 MB.

### DataLoader throughput

Through a `DataLoader` (the reference loader in `examples/`: shuffled tiles and cells, 5000-cell batches in shared memory, CSR
tensor rebuilt in the main process), same laptop, 10 tiles:

| workers | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| cells / s | 50k | 45k | 78k | 95k |

With the object store emulated by `ThrottledBackend` (50 ms latency, per-connection bandwidth, 2 workers), where prefetching the
next tiles matters:

| emulated bandwidth per connection | no prefetch | `prefetch_tiles=2` |
|---|---|---|
| 150 MB/s | 35k cells/s | 62k cells/s |
| 60 MB/s | 21k cells/s | 48k cells/s |

In earlier design experiments, the same pipeline structure gave 6-11k cells/s over TileDB-SOMA tiles and 70-130k cells/s around
BPCells' C++ decoder (1.3-1.8x faster than deltacells on this laptop) at 1-3 workers. Those experiments were one-off scripts that
are not part of this repository. All of these are laptop numbers with a warm cache and no real network; `benchmarks/` reproduces
the deltacells measurements on your own data and hardware (`make benchmark` runs them on synthetic data).

## Converting h5ad files

`deltacells convert` turns any number of h5ad files (any sizes, local or in Google Cloud Storage) into one dataset of exact-size
tiles. It sorts the genes, re-chunks the cells, compresses, and stores obs and var. The cells keep the order of the files and of
the cells within them. The files are naturally sorted (`shard_2` comes before `shard_10`), and `--keep-order` keeps the order you
give instead. All files must have the same `var_names`. `X` should be CSR (dense input works but is much slower) with integer
counts.

```bash
deltacells convert --h5ad-glob "shards/*.h5ad" --output /data/my_dataset
deltacells convert --file-list files.txt --output /data/my_dataset      # one path per line
```

### From and to Google Cloud Storage

Install the extra: `pip install "deltacells[gcs]"`.

**Credentials.** deltacells uses Google *Application Default Credentials*, which are **not** the login that `gcloud` and `gsutil`
use, so `gsutil ls` can work while deltacells is refused. On a Compute Engine VM with no other credentials, the default is the
VM's service account, and the VM's access scopes are often read-only (`devstorage.read_only`): reading the inputs works and the
first upload fails with `Forbidden ... Provided scope(s) are not authorized`. To use your own account, run this once (add
`--no-launch-browser` on a machine without a browser):

```bash
gcloud auth application-default login
```

The account needs read access to the inputs and write access to the output prefix. Before it reads anything, `convert` uploads and
deletes a small test file in the output, so a missing permission is reported within seconds, with this advice, rather than after
the scan.

`gs://` paths are **not** globbed by the converter. To convert a bucket of h5ad files, list them into a file and point `--output`
at a bucket prefix:

```bash
gcloud storage ls 'gs://my-bucket/h5ad/*.h5ad' > files.txt          # or: gsutil ls 'gs://my-bucket/h5ad/*.h5ad'
deltacells convert --file-list files.txt \
    --output gs://my-bucket/deltacells/my_dataset \
    --staging-dir /mnt/scratch --workers 12
```

The list can also be piped (`--file-list -`), mixed with local files, and combined with `--h5ad-glob`. Blank lines and lines that
start with `#` are skipped. Without `--keep-order` the files are naturally sorted; with it, the cells follow the order of the list.

The conversion is designed so that nothing has to fit on the local disk:

* **Scan.** The metadata of every file (`var_names`, obs, number of cells) is read with three ranged reads per file (about 21 MB of
  a 63 MB file), in `--scan-workers` processes (about 1.5 per core, up to 24). Processes are used because h5py serializes the
  reads it makes through a Python file object, so threads do not overlap them. In one measurement from GCS, one thread (or 16)
  scanned 1.3 files/s, about 108 minutes for 9,447 files, while 24 processes scanned about 17 files/s (480 files in 28 s,
  including process start-up; about 10 minutes for 9,447). Fewer than 32 files, or a `filesystem=` argument, are scanned in the
  main process. The processes are started with `spawn`, so a Python **script** that calls `convert_h5ad` needs the usual
  `if __name__ == "__main__":` guard (the `deltacells` command does not).
* **Gene order.** The counts are summed over `--sort-genes-max-files` files (10 by default, spread evenly over the list). Those
  files are downloaded one after the other, then deleted.
* **Tiles.** Each file is downloaded whole into `--staging-dir` just before its first tile needs it, and deleted after its last
  tile is written. Each finished tile is uploaded at once and deleted locally, as are the obs shards and the small files, and
  **`manifest.json` is uploaded last**, so a prefix without a manifest is not a dataset. The main process does the downloads and
  uploads in threads and hands local files to the worker processes that write the tiles; those never touch the network.
* **Scratch space.** About `5 x workers` input files plus a few tiles (for example, 12 workers on 60-100 MB files need a few GB).
  The run prints its estimate. `--staging-dir` defaults to the system temp directory, so point it at a disk with room. With a
  local `--output` and local inputs nothing is staged.
* **Speed.** Compression is the bottleneck, not the network. At the default `--level 19`, a 10,000-cell tile of about 2,300
  nonzeros per cell takes about 25 core-seconds to encode on an idle core (4.9 s with 8 threads). In one large run (9,447 files,
  12 workers on a 16-vCPU VM, which has 8 physical cores with 2 hyperthreads each), it took about 55 s per tile per worker, or
  0.22 tiles/s, which is about 11 hours for 94M cells. Busy hyperthreads are much slower than the idle core of the benchmark, so
  vCPUs should not be counted as cores. All 12 workers were at 100% CPU with downloads and uploads keeping up (one process
  downloaded from GCS at about 340 MB/s in a quick test), so more workers than physical cores cannot help much. `--level 12`
  encodes about 4x faster and gives files 6% larger (`--level 3`: 7x faster, 11% larger), which is usually the better setting for
  a dataset of this size. `--workers` defaults to the number of cores, limited by memory (about 50 bytes per nonzero of a tile).

While it runs, `convert` prints a line when each phase starts and ends. Every `--progress-interval` seconds (60 by default, the
first after 10 s; `0` for none), it also prints one line about the running phase: how far along it is, its speed over the last few
minutes, the estimated time left and, for the tiles, how many input files are staged and how many uploads are pending (which shows
whether downloads, encoding or uploads limit the speed). The estimate improves as the run progresses, so the first lines of a
phase are the least reliable. An abridged example (the numbers are illustrative, not from a full run):

```
9,447 files: gs://.../extract_000000.h5ad ... gs://.../extract_009446.h5ad
the output gs://.../deltacells is empty
reading the metadata of 9,447 files (24 processes)
scan: 2,100/9,447 files (22%), 16 files/s, elapsed 0:02:11, ETA 0:07:40
scan: 9,447 files in 0:09:50
94,470,000 cells x 61,497 genes in 9,447 files, 590 GB (cells per file: min 10,000, max 10,000)
gene order: summing the counts of 10 of 9,447 files (100,000 cells)
writing 9,447 tiles
tiles: 1,200/9,447 (13%), 0.46 tiles/s, 8.7 MB/s written, elapsed 0:43:10, ETA 4:58:00 | 62 files staged (4.0 GB), 3 uploads pending
done in 5:41:12: 9,447 tiles, 180,000.0 MB = 1.91 kB per cell
```

Safety:

* `--overwrite` deletes the existing dataset at `--output` first, but **only files that a deltacells dataset has** (the manifest,
  tiles, obs shards, gene order and so on). If the output holds anything else, the conversion stops and deletes nothing, and you
  need to clean it up yourself. It also refuses to run if an input file lies inside the output. Without `--overwrite`, a
  non-empty output is an error.
* The input h5ad files are only read, never changed or deleted.
* A refusal of the credentials (Forbidden / not authorized) is reported at once with advice and is never retried. Other network
  errors are retried a few times with a growing pause.
* If a conversion fails (an unreadable file, a tile with too many large counts, a network error after several retries), the tiles
  already uploaded stay in the bucket but there is no manifest, and the error message says so. Start again with `--overwrite`;
  there is no resume. The output, the arguments and the metadata of every file are checked before anything is deleted or
  uploaded.

From Python, the same operation is `deltacells.convert.convert_h5ad(["gs://...", ...], "gs://.../out", staging_dir=...)`.

## Using the dataset in a DataLoader

`DeltaCellsDataset` is meant to sit behind a data-loading worker. It provides what a shuffling, distributed data loader needs from
its source, in the same terms as an AnnData collection (for example, `limits` follows `DistributedAnnDataCollection.limits`):

| a loader needs | deltacells |
|---|---|
| number of cells and genes | `n_cells`, `n_genes` |
| shard boundaries for shuffling and splitting | `limits`, `tile_bounds(i)`, `tile_of(indices)` |
| fetch a batch by global index | `get_batch(indices, values_dtype=..., out_indices=..., out_values=...)` |
| size the output buffers first (for example, shared memory) | `batch_nnz(indices)` |
| "these shards will be needed next" | `prefetch(tile_ids)`, `prefetch_cells(indices)` |
| gene names and the gene table | `var_names`, `var` |
| per-cell metadata for a batch | `obs.take(indices, columns)` after `obs.localize(columns)` (or `obs_columns=` when opening) |

Things to know:

* **One tile is one object read.** A batch that spans two tiles reads and decodes both, exactly like two h5ad shards. Iterate tile
  by tile (shuffle the tiles, then the cells within a tile) so that consecutive batches share a tile.
* **Prefetch.** Call `prefetch` with the tiles that will be needed soon (the reference loader asks for the next two). Fetching and
  parsing run on background threads, and the compressed tiles are cached (`max_prefetch_tiles`).
* **Memory.** A decoded tile takes about 8 bytes per nonzero (about 0.4 GB for 10,000 deep cells). `max_cached_tiles` (default 2)
  bounds that, and decode buffers are recycled instead of reallocated, because fresh allocations page-fault and cost tens of
  milliseconds per tile.
* **Threads.** `decode_threads=1` is right when each worker has about one core. The C core releases the GIL, so more threads
  decode the tile's chunks in parallel. The dataset object is picklable (caches and threads are rebuilt in each worker).
* **Obs and other metadata** are never fetched with the tiles; see the next section.
* **Gene order is part of the data.** Columns are in the stored (sorted) order. Models that need the original order can use
  `ds.gene_order`. A fixed order across all tiles is what lets batches from different tiles be concatenated.

`examples/reference_loader.py` is a complete `IterableDataset` that shows the pattern (tile shuffling, replica and worker splits,
prefetch hints, shared-memory output), and `examples/dataloader_example.py` runs it end to end on synthetic data. The reference
loader is an example of how to call the dataset, not part of the package API. It does not provide resume-from-checkpoint, DDP
padding or `drop_last_indices`.

## Cell metadata (obs)

Obs is stored *beside* the tiles, not in them, so a loader that needs no metadata pays nothing for it, and a loader that needs a
few columns pays only for those. In the dataset it is stored as parquet **shards** (`obs/obs_0000.parquet`, ...) of up to 1000
tiles each, with **one row group per tile**, so obs row group `i` is always tile `i`. Categorical columns are stored as integer
codes with one **global vocabulary** per column (shards from different files can have different category sets; the converter
unifies them), and columns with more than 20,000 distinct values (such as barcodes) are stored as strings.

Training never reads the shards directly. Columns are **localized** once, on request:

* Only the requested columns' bytes are read (ranged reads of the footer and of each row group's column chunk, in parallel) and
  written to a node-local cache (`$DELTACELLS_CACHE`, else `~/.cache/deltacells`) as one memory-mapped Arrow IPC file per column.
* The cache is **additive**: asking for another column later reads only that column, and the columns you already have are
  untouched.
* The cache is shared by every process on the node (a lock makes concurrent processes fetch each column once) and is keyed by the
  dataset's content fingerprint, so a changed dataset never reuses stale files.
* Afterwards, `obs.take(indices, columns)` is a zero-copy lookup in memory-mapped files: workers share the pages, opening a column
  costs milliseconds and about a megabyte, and a 5000-cell batch takes about 0.2 ms.

Select the columns your models use up front with `obs_columns=[...]` (or `deltacells obs localize DATASET --columns a,b,c`) so
that nothing stalls a training run. A column that was not localized is fetched on first use, with a warning. Localizing everything
is possible (`--all`) but usually unnecessary:

| 100,000 real cells x 50 obs columns (+ cell names) | stored size | per 100M cells |
|---|---|---|
| all 51 columns | 6.1 MB | 6.1 GB |
| `cell_type` (30 categories) | 0.1 MB | 55 MB |
| `donor_id` (178 categories) | 0.1 MB | 96 MB |
| `age` (float) | 0.1 MB | 75 MB |
| `frac_contamination` (float) | 0.6 MB | 556 MB |
| `cell_barcode` (string) | 0.7 MB | 745 MB |

`deltacells obs info DATASET` prints this table for any dataset. Localizing `cell_type`, `donor_id` and `age` read 0.29 MB in 31
ranged reads (the three columns plus the parquet footers; 4.8% of the obs bytes) and took 0.39 s against a store emulated at
150 MB/s with 50 ms latency (`benchmarks/bench_obs.py`). The requests are one per column per tile, so at 100M cells (10,000 tiles)
the time is dominated by request latency, roughly `columns x tiles x latency / threads`. This is an extrapolation, not a
measurement; raise `threads` for high-latency stores.

Missing values: categorical codes use `-1`. Numeric columns with missing values come back as float64 with NaN. Booleans and
strings with missing values come back as object arrays with `None`. `to_pandas` returns pandas categoricals for categorical
columns.

## How it works

[docs/FORMAT.md](https://github.com/cellarium-ai/deltacells/blob/main/docs/FORMAT.md) is the format specification. In short:

1. Genes are sorted by decreasing total counts (from up to 10 files spread evenly over the inputs, or from all of them with
   `--sort-genes-max-files 0`). Frequently detected genes get small, adjacent column indices, so the gaps between a cell's
   nonzero genes are small and repetitive.
2. Per cell, the gene indices are delta coded (`g0, g1-g0, g2-g1, ...`); together with the counts this gives two uint16 streams. A
   count or a gap above 65535 (a few counts of very highly expressed genes; a gap only with more than 65,536 genes) is stored as
   65535 plus an entry in a small per-chunk *exception frame*, so the common decoding path is not slower for them.
3. Each stream is split into byte planes (all low bytes, then all high bytes) and compressed with zstd, per chunk of about 1/8 of
   the tile's nonzeros. Chunks are independent, so a tile decodes in parallel, at a size cost of about 0.5% or less.
4. A tile is a header, a chunk table, `indptr` and the compressed streams: one object, decodable on its own and CRC-protected.

Measured trade-offs on one real tile (`benchmarks/bench_compression.py`; 10,000 cells x 61,497 genes, 23.1M nonzeros; Linux
x86-64, Xeon 2.3 GHz, Python 3.10):

| gene order | zstd level | chunks | size | decode (1 thread) | encode (8 threads) |
|---|---|---|---|---|---|
| sorted | 3 | 8 | 20.8 MB | 125 ms | 0.7 s |
| sorted | 12 | 8 | 20.0 MB | 125 ms | 1.3 s |
| sorted | **19** | 8 | **18.8 MB** | **110 ms** | 4.9 s |
| sorted | 19 | 1 | 18.8 MB | 130 ms | 23.5 s |
| random | 19 | 8 | 24.9 MB | 104 ms | 5.0 s |

Sorting the genes saves 22-24% at every level. Level 19 is the smallest and decodes fastest; it only costs compression time. The
single-thread decode times carry several percent of run-to-run noise (read the rows other than level 19 with 8 chunks as "about
the same"); the sizes are exact.

## Limitations

* **Integer counts only**: integers in [0, 2^31 - 1] (float32 batches are exact up to 2^24). Counts above 65535, and gaps above
  65535 between a cell's consecutive expressed genes (possible with more than 65,536 genes), are stored as exceptions and must be
  rare: the writer refuses a tile with more than `max(1024, nnz / 200)` of either kind, which is not what raw counts look like.
  Sparse feature spaces such as ATAC peaks are out of scope, and so are normalized or float data.
* **Conversion has no resume.** A failed conversion is started again with `--overwrite` (see
  [Converting h5ad files](#converting-h5ad-files)).
* **Platforms.** Linux and macOS, little-endian hosts only. CI covers Linux x86-64 (Python 3.10 to 3.13) and macOS arm64
  (Python 3.12). Windows is not supported.
* **Obs needs `pyarrow` and `pandas`.** Categorical, numeric, boolean and string columns are supported; exclude others, such as
  datetimes. The obs localization and shard layout are exercised against local files and the throttled backend only. The local
  cache is uncompressed Arrow, which is several times the size of the parquet for small-integer columns (4.8x for the three
  columns above); strings and floats cost their full size on disk.
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

## Development

```bash
make test         # the test suite: format validation, round trips and edge cases, gather, dataset (prefetch, caching, pickling,
                  # corruption), obs, backends, h5ad conversion, CLI, torch / DataLoader, benchmark smoke tests
make lint         # ruff
make typecheck    # mypy
make benchmark    # all benchmarks that need no real data, on a synthetic dataset
```

The individual benchmarks, which also run on real data:

```bash
python benchmarks/bench_decode.py DATASET
python benchmarks/bench_pipeline.py DATASET --workers 0,1,2,4 [--bandwidth-mbps 150 --latency-ms 50]
python benchmarks/bench_compression.py DATASET
python benchmarks/bench_obs.py DATASET --columns cell_type,donor_id
python benchmarks/check_against_h5ad.py DATASET "shards/*.h5ad"        # X and every obs column, cell by cell
python benchmarks/make_synthetic.py OUT_DIR       # realistic synthetic data if you have no real shards
```

[DEVELOPING.md](https://github.com/cellarium-ai/deltacells/blob/main/DEVELOPING.md) describes building from source, the code layout, how to change the format, and how releases are
made.

## License

BSD-3-Clause (see [LICENSE](https://github.com/cellarium-ai/deltacells/blob/main/LICENSE)). deltacells links libzstd, which is dual-licensed BSD-3-Clause / GPL-2.0 and is used under
the BSD license.
