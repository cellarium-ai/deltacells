# Benchmarks

This page gives the measurements behind the claims on the front page: how small the files are, how fast a tile decodes, how
deltacells compares with other formats, and how fast a `DataLoader` reads from it. Every table says which machine and data it
comes from, and every number can be reproduced with the scripts in
[`benchmarks/`](https://github.com/cellarium-ai/deltacells/tree/main/benchmarks) (see the [last section](#reproducing-the-benchmarks)).

All of these numbers are for the counts matrix `X` only. They were measured with a warm page cache and no network, unless a table
says otherwise, so they show the speed of the format and the decoder rather than of any particular disk or bucket.

## Size and decode speed on a server

One real shard of 10,000 cells by 61,497 genes (human, 2,309 nonzeros per cell), on a Google Cloud VM (n1-standard-16: Xeon 2.3
GHz, 8 cores and 16 vCPUs), local disk.

"File to RAM" includes reading the file from disk and producing CSR arrays.

| format | stored size | kB per cell | file to RAM, 1 thread | 8 threads |
|---|---|---|---|---|
| h5ad (as is: original gene order, gzip) | 55.6 MB | 5.56 | 981 ms | single-threaded reader |
| **deltacells** (zstd 19), reused buffers | **18.9 MB** | **1.89** | 118 ms | 26 ms |
| deltacells (zstd 19), fresh arrays per call | 18.9 MB | 1.89 | 164 ms | 35 ms |

"Reused buffers" is how `DeltaCellsDataset` runs: decoding writes into buffers that are kept between tiles. Fresh arrays pay for
page faults on every call.

Decoding alone, with the tile already in memory (`bench_decode.py`, int32 values, reused buffers):

| threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| decode a 10,000-cell tile | 108 ms | 57 ms | 32 ms | 20 ms |
| cells per second | 93k | 175k | 312k | 497k |

The three tiles of this dataset give the same results, although one has no counts above 65,535 and two do (see
[how counts above 65,535 are stored](encoding.md#values-that-do-not-fit)). Gathering 5,000 shuffled cells out of a decoded tile
takes 36 ms.

## Comparison with other formats

One real shard of 10,000 cells by 38,094 genes (snRNA-seq, 5,171 nonzeros per cell), written in each format and read from a local
file back into memory, on an M2 Pro laptop.

| format | stored size | kB per cell | file to RAM, 1 thread | 8 threads |
|---|---|---|---|---|
| h5ad (as is: original gene order, gzip) | 124.6 MB | 12.46 | 1170 ms | single-threaded reader |
| TileDB-SOMA (tuned layout), to CSR | 56.7 MB | 5.67 | 1361 ms | 394 ms |
| BPCells, Python bindings | 67.4 MB | 6.74 | 1234 ms | 275 ms |
| BPCells, C++ decoder (uint32 values, preallocated) | 67.4 MB | 6.74 | **48 ms** | **30 ms** |
| **deltacells** (zstd 19), reused buffers | **38.4 MB** | **3.84** | 125 ms | 39 ms |
| deltacells (zstd 19), fresh arrays per call | 38.4 MB | 3.84 | 148 ms | 56 ms |

```{note}
TileDB-SOMA and BPCells are not installed on the Linux machine above, so this comparison was not repeated there. The deltacells
rows come from a build that predates the exception format and the `-fno-wrapv` compiler flag, so they are slightly conservative
for the current code.
```

How to read this table:

* **BPCells' decoder is the fastest reader, by a wide margin over its own Python bindings.** Calling BPCells' BP-128 decoders
  directly takes 48 ms per tile, including reading its files. The Python bindings take 1.2 s because they build the result
  through an Eigen sparse matrix. The C++ harness in `benchmarks/bpcells_decode/` is a measurement tool and not a supported
  reader: it leaves the values as uint32 and decodes into preallocated arrays.
* **deltacells trades decode time for size.** Its files are 43% smaller than BPCells' (38.4 against 67.4 MB), and it needs about
  2.6 times the decode time and CPU time per tile. Which one is faster end to end depends on where the tiles live. When fetching a
  tile takes longer than decoding it, the smaller file wins. With fetching and decoding overlapped, deltacells' time per tile is
  lower below roughly 500 MB/s of bandwidth per worker, and the break-even is about 400 MB/s when they are not overlapped. These
  crossovers are estimates computed from the numbers above, not measurements. With fast local storage and few cores to spare,
  BPCells' decoder is the cheaper reader.
* The sorted formats (TileDB-SOMA, BPCells and deltacells) use the same gene order, sorted by decreasing total counts of this
  shard. The TileDB-SOMA layout is the best-performing one that was tried: tile extent equal to the shard, row-major, delta plus
  zstd 3 on the gene coordinate, and byte-shuffle plus zstd 3 on the values. The h5ad file is left as users have it.
* The times are for `X` alone. The h5ad file as a whole is 148 MB, of which `X` is 124.6 MB.

## Throughput through a DataLoader

This is the number that matters for training: cells per second delivered to the training process. It uses the reference loader in
[`examples/`](https://github.com/cellarium-ai/deltacells/blob/main/examples/reference_loader.py) (shuffled tiles and cells,
5,000-cell batches in shared memory, CSR tensor rebuilt in the main process), on the same M2 Pro laptop, over 10 tiles:

| DataLoader workers | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| cells per second | 50k | 45k | 78k | 95k |

To see what happens when tiles come from an object store, `ThrottledBackend` adds a fixed latency (50 ms) and a bandwidth limit
per connection. With 2 workers:

| emulated bandwidth per connection | no prefetch | `prefetch_tiles=2` |
|---|---|---|
| 150 MB/s | 35k cells/s | 62k cells/s |
| 60 MB/s | 21k cells/s | 48k cells/s |

Prefetching the next tiles roughly doubles the throughput when the store is slow, because fetching overlaps with decoding.

In earlier design experiments, the same pipeline structure gave 6k to 11k cells per second over TileDB-SOMA tiles, and 70k to
130k cells per second around BPCells' C++ decoder (1.3 to 1.8 times faster than deltacells on this laptop), at 1 to 3 workers.
Those experiments were one-off scripts that are not part of this repository. All of these are laptop numbers, with a warm cache
and no real network.

## Compression level and chunking

One real tile of 10,000 cells by 61,497 genes (23.1 million nonzeros), on the Linux server above (`bench_compression.py`):

| gene order | zstd level | chunks | size | decode, 1 thread | encode, 8 threads |
|---|---|---|---|---|---|
| sorted | 3 | 8 | 20.8 MB | 125 ms | 0.7 s |
| sorted | 12 | 8 | 20.0 MB | 125 ms | 1.3 s |
| sorted | **19** | 8 | **18.8 MB** | **110 ms** | 4.9 s |
| sorted | 19 | 1 | 18.8 MB | 130 ms | 23.5 s |
| random | 19 | 8 | 24.9 MB | 104 ms | 5.0 s |

* **Sorting the genes saves 22% to 24% at every level.** This is the main source of the compression; see
  [how the encoding works](encoding.md).
* **Level 19 is the smallest and decodes fastest.** It only costs compression time. Level 12 compresses about 4 times faster for
  files 6% larger, and level 3 about 7 times faster for files 11% larger.
* **Chunks cost almost nothing in size and make encoding parallel.** Splitting a tile into 8 independently compressed chunks
  lets 8 threads encode (and decode) it at once, at a size cost of about 0.5% or less. With one chunk, encoding takes 23.5 s
  instead of 4.9 s.
* The single-thread decode times carry several percent of run-to-run noise, so read the rows other than level 19 with 8 chunks as
  "about the same". The sizes are exact.

## Cell metadata

Cell metadata is stored beside the tiles, and a loader reads only the columns it asks for (see
[Cell metadata (obs) and genes (var)](obs.md)). For 100,000 real cells with 50 obs columns plus cell names:

| column | stored size | per 100M cells |
|---|---|---|
| all 51 columns | 6.1 MB | 6.1 GB |
| `cell_type` (30 categories) | 0.1 MB | 55 MB |
| `donor_id` (178 categories) | 0.1 MB | 96 MB |
| `age` (float) | 0.1 MB | 75 MB |
| `frac_contamination` (float) | 0.6 MB | 556 MB |
| `cell_barcode` (string) | 0.7 MB | 745 MB |

Copying `cell_type`, `donor_id` and `age` to the local cache read 0.29 MB in 31 ranged reads (4.8% of the metadata) and took 0.39 s
against a store emulated at 150 MB/s with 50 ms latency. Afterwards, looking up a 5,000-cell batch takes about 0.2 ms.
`deltacells obs info DATASET` prints a table like this one for any dataset.

## Is the loader fast enough for my model?

The throughput above is what the loader can deliver. Whether it keeps a GPU busy depends on how many cells per second your model
consumes. To check, measure two things on your hardware:

1. **The loader alone.** Run `bench_pipeline.py` on your dataset with the number of workers you plan to use. It reports cells per
   second and, for the main process, how long it waited for each batch. A wait near zero means the main process always had a
   batch ready.
2. **The model alone.** Time your training step on batches that are already in GPU memory, and convert to cells per second
   (batch size divided by step time).

If the loader's cells per second is above the model's, the loader is not the bottleneck. You can confirm this in a real training
run by watching GPU utilization, or by timing how long each step waits for its next batch. If it is lower, add workers (up to the
number of physical cores) or check where the time goes with `bench_decode.py`.

## Reproducing the benchmarks

The benchmark scripts live in `benchmarks/`. The simplest way to run them is on synthetic data, which needs no h5ad files:

```bash
make benchmark
```

This generates a synthetic dataset and runs the decode, compression, obs and DataLoader benchmarks on it. The size, thread counts
and worker counts can be changed on the command line, for example
`make benchmark BENCH_CELLS=100000 BENCH_THREADS=1,2,4 BENCH_WORKERS=0,2,4`. Run it on an otherwise idle machine; the pipeline
benchmark needs a few GB of free memory per worker.

To measure on your own data, convert it first (see [Converting h5ad files](convert.md)) and run the scripts directly:

```bash
python benchmarks/bench_decode.py DATASET                 # decode and gather speed of single tiles
python benchmarks/bench_pipeline.py DATASET --workers 0,1,2,4    # DataLoader throughput
python benchmarks/bench_pipeline.py DATASET --workers 2 --bandwidth-mbps 150 --latency-ms 50   # emulate a remote store
python benchmarks/bench_compression.py DATASET --levels 3,12,19 --chunks 1,8
python benchmarks/bench_obs.py DATASET --columns cell_type,donor_id
python benchmarks/compare_formats.py shard.h5ad           # h5ad against TileDB-SOMA, BPCells and deltacells, for one shard
```

`compare_formats.py` needs TileDB-SOMA and BPCells installed for their rows, and `benchmarks/bpcells_decode/build.sh` once for
the BPCells C++ decoder. Use `--only h5ad,deltacells` to compare just those two. Very small datasets give very short timed windows
in the pipeline benchmark, so use at least a few tens of thousands of cells.
