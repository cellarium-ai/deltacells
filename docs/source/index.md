# deltacells

**Fast, compact tiles of single-cell count matrices, built for feeding a training `DataLoader`.**

Training a model on tens to hundreds of millions of cells means reading raw counts over and over, and the format they are stored in determines
how fast that goes and how much disk it takes. deltacells stores a count matrix as a directory (or bucket prefix) of *tiles* of
10,000 cells (configurable). Each tile is one self-contained object that a small C library decodes straight into CSR arrays.

## Why deltacells

* **Very fast data loading.** A 10,000-cell tile decodes in about 0.1 s on one core and 0.02 s on eight, with no intermediate
  AnnData or Arrow objects. A PyTorch `DataLoader` reading shuffled batches on a laptop sustains 45,000 to 95,000 cells per
  second, depending on the number of workers ([benchmarks](benchmarks.md)).
* **Small on disk.** A dataset is about 40% smaller than the same data in [BPCells](https://github.com/bnprks/BPCells) (3.8 vs
  6.7 kB per cell) and roughly a third of the size of a gzip-compressed h5ad file. The key idea is to store genes in order of decreasing total counts, which makes the
  values that remain to be compressed small and repetitive ([how the encoding works](encoding.md)).
* **Fast enough to keep a GPU fed.** On a single thread, decoding is slower than BPCells' decoder (about 2.6 times as long per
  tile), in exchange for the smaller files. With a few `DataLoader` workers, each decoding on its own core, deltacells
  is designed to keep a GPU busy, as in the data loading of [cellarium-ml](https://github.com/cellarium-ai/cellarium-ml).
  Smaller files also mean less to download when the dataset lives in a bucket.

One real 10,000-cell shard (38,094 genes), counts only, read from a local file into memory on an M2 Pro laptop:

| format | size on disk | decode, 1 thread | decode, 8 threads |
|---|---|---|---|
| h5ad (gzip) | 12.5 kB per cell | 1170 ms | single-threaded reader |
| BPCells (C++ decoder) | 6.7 kB per cell | **48 ms** | **30 ms** |
| **deltacells** | **3.8 kB per cell** | 125 ms | 39 ms |

The [benchmarks page](benchmarks.md) has the details, other hardware, and how to reproduce them.

## Quick start

```bash
pip install "deltacells[h5ad,obs]"
deltacells convert --h5ad-glob "shards/*.h5ad" --output my_dataset
```

```python
import numpy as np
from deltacells import open_dataset

ds = open_dataset("my_dataset")
batch = ds.get_batch(np.array([5, 99_000, 12]))  # any cells, in any order
batch.to_scipy()  # or batch.to_torch_csr()
```

```{toctree}
:maxdepth: 1
:caption: Contents

install
convert
dataloader
obs
benchmarks
encoding
format
```
