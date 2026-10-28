# Using a dataset in a PyTorch DataLoader

A deltacells dataset is read through `DeltaCellsDataset`, which returns batches of any cells you ask for as CSR arrays. This page
shows how to put it behind a PyTorch `DataLoader`. For a complete training setup with PyTorch Lightning, see
[cellarium-ml](https://github.com/cellarium-ai/cellarium-ml), which provides the data loading, models and training loops built
for this kind of data.

## Reading cells

```python
import numpy as np
from deltacells import open_dataset

ds = open_dataset("my_dataset")  # a directory, or "gs://bucket/prefix"
ds.n_cells, ds.n_genes, ds.n_tiles

batch = ds.get_batch(np.array([5, 99_000, 12]))  # any global cell indices, in any order
batch.indptr, batch.indices, batch.values  # CSR arrays: int64, int32, float32
batch.to_scipy()  # a scipy.sparse.csr_matrix
batch.to_torch_csr()  # a torch.sparse_csr tensor sharing the memory
```

Columns are in the stored gene order (genes sorted by decreasing total counts). `ds.var_names` gives the gene names in that
order, and `ds.gene_order` the original column of each stored gene.

## What a loader needs

| a data loader needs | `DeltaCellsDataset` provides |
|---|---|
| number of cells and genes | `n_cells`, `n_genes` |
| shard boundaries for shuffling and splitting | `limits`, `tile_bounds(i)`, `tile_of(indices)` |
| a batch by global cell index | `get_batch(indices)` |
| the size of a batch, to allocate output buffers (for example in shared memory) | `batch_nnz(indices)`, then `get_batch(indices, out_indices=..., out_values=...)` |
| "these tiles will be needed next" | `prefetch(tile_ids)`, `prefetch_cells(indices)` |
| gene names and the gene table | `var_names`, `var` |
| cell metadata for a batch | `obs.take(indices, columns)` |

## A minimal DataLoader

The dataset stores cells in **tiles** of about 10,000 consecutive cells, and reading a tile is one object read. The efficient way
to shuffle is therefore in two levels: visit the tiles in random order, and shuffle the cells within each tile. Batches then
come from one tile at a time, and the next tiles can be fetched in the background while the current one is used.

```python
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from deltacells import open_dataset


class TileBatches(IterableDataset):
    """Shuffled batches of cells: tiles in random order, cells shuffled within each tile."""

    def __init__(self, path, batch_size=5000, seed=0):
        self.ds = open_dataset(path)
        self.batch_size, self.seed = batch_size, seed

    def __iter__(self):
        info = get_worker_info()
        worker, n_workers = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng([self.seed, worker])
        tiles = np.random.default_rng(self.seed).permutation(self.ds.n_tiles)[worker::n_workers]  # this worker's tiles
        self.ds.prefetch(tiles[:3])
        for k, tile in enumerate(tiles):
            self.ds.prefetch(tiles[k + 1 : k + 3])  # fetch the next tiles while this one is used
            start, stop = self.ds.tile_bounds(int(tile))
            cells = rng.permutation(np.arange(start, stop))
            for i in range(0, len(cells), self.batch_size):
                batch = self.ds.get_batch(cells[i : i + self.batch_size])
                yield {  # plain tensors cross the process boundary; the sparse tensor is built in the training process
                    "indptr": torch.from_numpy(batch.indptr),
                    "indices": torch.from_numpy(batch.indices),
                    "values": torch.from_numpy(batch.values),
                }


def to_csr(batch, n_genes):
    return torch.sparse_csr_tensor(
        batch["indptr"], batch["indices"], batch["values"], size=(len(batch["indptr"]) - 1, n_genes)
    )


# batch_size=None: the dataset already yields whole batches
loader = DataLoader(TileBatches("my_dataset"), batch_size=None, num_workers=4)
n_genes = loader.dataset.ds.n_genes
for batch in loader:
    x = to_csr(batch, n_genes).to("cuda")  # move the sparse batch to the GPU first,
    x = x.to_dense()  # then make it dense there
    ...  # forward and backward pass
```

Each worker takes its own share of the tiles, so the workers decode in parallel on separate cores. Move the sparse batch to the
GPU before densifying it: a dense batch of 5,000 cells by 30,000 genes is 600 MB, while the sparse form of a batch with
2,300 nonzeros per cell is about 90 MB, so the transfer is several times smaller.

## Getting the most out of it

* **Iterate tile by tile.** A batch that spans two tiles reads and decodes both. Shuffling tiles and then cells within a tile
  (as above) keeps consecutive batches on the same tile, and it is what gives the throughput measured in the
  [benchmarks](benchmarks.md).
* **Prefetch.** `prefetch` starts fetching and parsing tiles on background threads, which hides the latency of a bucket. The
  example above looks two tiles ahead.
* **Memory.** A decoded tile takes about 8 bytes per nonzero, which is around 0.4 GB for 10,000 deep cells. `max_cached_tiles`
  (default 2) bounds the cache, and decode buffers are reused rather than reallocated.
* **Threads.** `decode_threads=1` (the default) is right when each worker has about one core. The decoder releases the Python
  GIL, so more threads decode a tile's chunks in parallel.
* **Pickling.** The dataset object can be pickled, which is how it reaches the `DataLoader` workers. Caches and threads are
  rebuilt in each worker.
* **Fewer copies.** To avoid one copy between the worker and the training process, allocate the output tensors in shared memory
  and pass them to `get_batch` (`batch_nnz` gives the size). The reference loader does this.

## Cell metadata

If the dataset has cell metadata, columns are read separately from the counts, so a loader that does not need them pays nothing.
Choose the columns up front and they are copied once into a local cache:

```python
ds = open_dataset("my_dataset", obs_columns=["cell_type", "donor_id"])
codes = ds.obs.take(batch_cell_indices, ["cell_type", "donor_id"])  # integer codes for the batch, a local lookup
ds.obs.categories("cell_type")  # the vocabulary behind the codes
```

See [Cell metadata (obs) and genes (var)](obs.md) for how the metadata is stored, cached and read.

## A fuller example

The repository has two examples, both runnable:

* [`examples/reference_loader.py`](https://github.com/cellarium-ai/deltacells/blob/main/examples/reference_loader.py) is a
  complete `IterableDataset` with tile shuffling per epoch, splitting across data-parallel replicas and workers, prefetch hints
  and shared-memory output.
* [`examples/dataloader_example.py`](https://github.com/cellarium-ai/deltacells/blob/main/examples/dataloader_example.py) runs it
  end to end on synthetic data.

These show the pattern; they are examples, not part of the package. For training, use
[cellarium-ml](https://github.com/cellarium-ai/cellarium-ml).
