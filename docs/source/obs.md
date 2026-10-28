# Cell metadata and genes

Besides the counts, a deltacells dataset can hold the cell metadata (the `obs` table of an AnnData object: cell types, donors, obs_names, etc.) and the gene table (`var`). Both are optional and both are stored *beside* the tiles. A data
loader that needs no metadata pays nothing for it, and one that needs a few columns pays only for those.

Reading and writing obs and var needs `pyarrow` and `pandas` (`pip install "deltacells[obs]"`; see [Install](install.md)).

## What the converter stores

`deltacells convert` reads `X`, `obs` and `var` from the h5ad files. Other parts of the AnnData object (`layers`, `obsm`,
`varm`, `obsp`, `uns`) are not stored.

### Obs

Each obs column is stored as one of four kinds:

| kind | what it holds | how it is stored |
|---|---|---|
| `category` | categorical columns, such as `cell_type` or `donor_id` | integer codes, with one **global vocabulary** per column |
| `numeric` | integer and floating-point columns | the numeric type of the source (a common type if files differ) |
| `bool` | boolean columns | booleans |
| `string` | text columns, such as barcodes | strings |

* **Global vocabularies.** Different input files can have different category sets, and the converter combines them into one
  vocabulary per column. A code therefore means the same thing in every tile. Code `-1` means a missing value.
* **Many categories become strings.** A categorical column with more than 20,000 distinct values (barcodes, for example) is
  stored as strings instead. The limit can be changed with `--max-categories`.
* **Cell names.** The cell names (the obs index) are stored as a string column named `obs_names`. Leave them out with
  `--no-obs-names`.
* **Unsupported columns.** Other types, such as datetimes, are an error. The message names the column; leave it out with
  `--obs-exclude name1,name2`.
* **Everything or nothing.** `--no-obs` skips obs altogether.

On disk, obs is a set of parquet files (`obs/obs_0000.parquet`, ...) with one row group per tile, so obs row group `i` always
describes the cells of tile `i`. By default each file holds up to 1,000 tiles (`--obs-tiles-per-file`). The
[format specification](format.md) has the exact layout.

### Var and gene order

The gene table is stored as `var.parquet`, with the genes in the dataset's column order (the converter's sorted order), and the
gene names as `var_names.txt`. The table is taken from the first input file. `--no-var` skips it.

The original column of each stored gene is kept in `gene_order.npy`. Because the genes are stored sorted by decreasing total counts
(see [How the encoding works](encoding.md)), the columns of a batch are in that order, not in the order of your h5ad files:

```python
from deltacells import open_dataset

ds = open_dataset("my_dataset")
ds.var_names  # gene names, in stored column order
ds.var  # the gene table (a pandas DataFrame), in stored column order
ds.gene_order  # ds.gene_order[j] is the column of the source files that is stored as column j
```

To get a batch back in the original gene order, index its columns with the inverse of `ds.gene_order`:

```python
import numpy as np

original_order = np.argsort(ds.gene_order)  # stored column of each original gene
x = ds.get_batch(cells).to_scipy()[:, original_order]
```

`ds.var_names`, `ds.var` and `ds.gene_order` are `None` if the dataset was written without them.

## Reading obs

Training never reads the obs parquet files directly. Instead, the columns you need are **localized** once: copied into a cache on
the local disk, from which they are read through memory-mapped files. Choose the columns when you open the dataset:

```python
ds = open_dataset("gs://bucket/prefix", obs_columns=["cell_type", "donor_id"])
ds.obs.columns  # every column
ds.obs.kind("cell_type")  # "category"
ds.obs.categories("cell_type")  # the vocabulary: code i is categories[i]
```

Then look up the values for a batch of cells, which is a fast local operation:

```python
codes = ds.obs.take(cell_indices, ["cell_type", "donor_id"])  # {"cell_type": integer codes, ...}
frame = ds.obs.take_pandas(cell_indices, ["cell_type", "donor_id"])  # the same, as a pandas DataFrame
```

`take` returns numpy arrays in the order of `cell_indices`: integer codes for categorical columns (`-1` for missing), object
arrays of strings, and numeric arrays. `take_pandas` returns the categorical columns as pandas categoricals that carry the global
vocabulary, so `.cat.categories` is the same in every batch.

Other ways to read:

```python
ds.obs.localize(["age"])  # add more columns whenever you like; only their bytes are read
df = ds.obs.to_pandas(["cell_type", "age"])  # the whole dataset as one DataFrame, for example for plotting
```

### Missing values

* Categorical columns use the code `-1`.
* Numeric columns with missing values come back as float64 with `NaN`.
* Boolean and string columns with missing values come back as object arrays with `None`.

### Localize up front

Pick the columns your model uses at the start (`obs_columns=[...]`, or `ds.obs.localize([...])`) so that nothing stalls training. A
column that was not localized is fetched the first time it is used, with a warning. `ds.obs.localize()` with no argument
fetches every column, which is possible but usually unnecessary. The sizes are in the
[benchmarks](benchmarks.md#cell-metadata): for 100,000 cells with 50 columns, a categorical column is about 0.1 MB, a float
column up to 0.6 MB, and a barcode string column 0.7 MB.

## The local cache

Localized columns are written to a cache directory, one memory-mapped Arrow file per column:

* **Location.** The directory is, in order of preference, the `cache_dir` argument of `open_dataset`, the `DELTACELLS_CACHE`
  environment variable, `$XDG_CACHE_HOME/deltacells`, and `~/.cache/deltacells`. `ds.obs.cache_root` shows the one in use.
  Choose a fast local disk.
* **Additive.** Asking for another column later reads only that column. The columns you already have are untouched.
* **Shared.** Every process on the machine uses the same files, so `DataLoader` workers share the memory-mapped pages. A lock
  makes concurrent processes fetch each column only once.
* **Safe to keep.** The cache is keyed by a fingerprint of the dataset's obs, so a changed dataset never reuses stale files. Old
  directories can be deleted when no process is using them; the columns are fetched again when needed.
* **Size.** The cache holds uncompressed Arrow, which is several times the size of the compressed parquet for small-integer
  columns (4.8 times for three categorical and numeric columns in the benchmark). Strings and floats cost their full size on
  disk.

Looking up a batch of 5,000 cells takes about 0.2 ms, and opening a column costs milliseconds and about a megabyte per process.

## Cost of reading remote obs

For a dataset in a bucket, localizing reads only the requested columns' bytes, with ranged reads of each parquet footer and of each
row group's column chunk. Fetching `cell_type`, `donor_id` and `age` for a 100,000-cell dataset read 0.29 MB in 31 requests (4.8%
of the obs bytes) and took 0.39 s against a store emulated at 150 MB/s with 50 ms latency (see the
[benchmarks](benchmarks.md#cell-metadata)).

There is one request per column per tile, so for a very large dataset the time is dominated by request latency, roughly
`columns × tiles × latency / threads`. At 100 million cells (10,000 tiles) this is an extrapolation, not a measurement. The
`threads` argument of `ds.obs.localize` (and `--threads` below) raises the parallelism for high-latency stores.

## From the command line

```bash
deltacells obs info my_dataset                           # columns, kinds, how many bytes each costs to fetch, which are local
deltacells obs localize my_dataset --columns cell_type,donor_id
deltacells obs localize my_dataset --all                 # every column
```

`deltacells info` also summarizes the obs and var of a dataset, and `deltacells verify` checks the obs shards along with the
tiles.

## Limitations

* Obs supports categorical, numeric, boolean and string columns. Exclude other types when converting.
* The obs shard layout and localization are tested against local files and a throttled backend. The ranged reads used to localize
  from a real Google Cloud Storage bucket (`GCSBackend.read_range`) have not been tested against a real bucket.
* Within a tile the obs rows follow the cell order of the tile, so the metadata always lines up with the cells you read. There is
  no way to filter or query obs inside the dataset; read the columns you need and select in memory.
