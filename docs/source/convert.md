# Converting h5ad files

`deltacells convert` turns any number of h5ad files into one deltacells dataset. The files can be any size, and local or in
Google Cloud Storage. The converter sorts the genes, re-chunks the cells into tiles of exactly the same size, compresses them, and
stores the cell metadata (obs) and the gene table (var) alongside.

It needs the `h5ad` and `obs` extras (`pip install "deltacells[h5ad,obs]"`; see [Install](install.md)).

## What your files need

* **Raw integer counts** between 0 and 2,147,483,647. Normalized or float data is rejected.
* **The same genes in every file.** All files must have identical `var_names`, in the same order.
* **CSR storage for `X`.** Dense `X` is converted too, but much more slowly.

## Convert

From files on disk:

```bash
deltacells convert --h5ad-glob "shards/*.h5ad" --output my_dataset
```

Or from a list of files, one path per line (blank lines and lines starting with `#` are skipped):

```bash
deltacells convert --file-list files.txt --output my_dataset
```

The cells keep the order of the files and of the cells within each file. The files are sorted naturally (`shard_2` comes before
`shard_10`); add `--keep-order` to use the order you gave instead. The converter does not shuffle cells. A tile holds consecutive
cells, so if your input is sorted by donor or tissue, shuffle the cells across files first.

When it finishes, look at the result and check it:

```bash
deltacells info my_dataset      # cells, genes, tiles, size on disk
deltacells verify my_dataset    # checks the CRC of every tile and decodes it
```

## Common options

| option | default | meaning |
|---|---|---|
| `--tile-size` | 10000 | cells per tile; every tile but the last holds exactly this many |
| `--level` | 19 | zstd level: higher is smaller but slower to write (see below) |
| `--workers` | number of cores | processes that encode tiles in parallel (limited by memory) |
| `--no-sort-genes` | off | keep the genes in their original order (this costs compression, see [Encoding](encoding.md)) |
| `--overwrite` | off | replace an existing dataset at `--output` |
| `--no-obs`, `--no-var` | off | do not store the cell metadata or the gene table |
| `--obs-exclude a,b` | none | leave out these obs columns |

Run `deltacells convert --help` for all of them.

### Choosing the compression level

The default `--level 19` gives the smallest files and takes about 25 core-seconds to compress a 10,000-cell tile. Decoding is
not slower at higher levels, so a higher level only costs time when writing. For a very large dataset, `--level 12` compresses
about 4 times faster for files that are 6% larger, and `--level 3` about 7 times faster for files 11% larger. The
[benchmarks page](benchmarks.md) has the measurements.

## Converting from and to Google Cloud Storage

Install the extra (`pip install "deltacells[gcs]"`), and give `gs://` paths for the inputs, the output, or both:

```bash
gcloud storage ls 'gs://my-bucket/h5ad/*.h5ad' > files.txt
deltacells convert --file-list files.txt \
    --output gs://my-bucket/deltacells/my_dataset \
    --staging-dir /mnt/scratch --workers 12
```

`gs://` paths are not expanded by the converter, so list them into a file as above. The list can also be piped in
(`--file-list -`).

**Credentials.** deltacells uses Google *Application Default Credentials*, which are not the login that `gcloud` and `gsutil`
use. If reading works but the first upload fails with `Forbidden ... Provided scope(s) are not authorized`, run this once:

```bash
gcloud auth application-default login
```

The account needs read access to the inputs and write access to the output. Before reading anything, the converter uploads and
deletes a small test file in the output, so a missing permission is reported within seconds.

**Disk space.** Nothing has to fit on the local disk. Each input file is downloaded into `--staging-dir` just before its first
tile needs it and deleted after its last tile is written, and each finished tile is uploaded and deleted at once. Plan for about
5 files per worker plus a few tiles (12 workers on 60 to 100 MB files need a few GB); the run prints its estimate. The staging
directory defaults to the system temp directory, so point it at a disk with room.

**Time.** Compression is the bottleneck, not the network. As a guide, converting 94 million cells from 9,447 files with 12
workers on a 16-vCPU VM took on the order of 11 hours at the default level. Use physical cores, not vCPUs, when choosing
`--workers`.

**A note for scripts.** The converter starts worker processes with `spawn`, so a Python script that calls
`deltacells.convert.convert_h5ad` needs the usual `if __name__ == "__main__":` guard. The `deltacells` command does not.

## Progress and safety

While it runs, the converter prints one line when each phase starts and ends, and every `--progress-interval` seconds (60 by
default) a progress line with the speed and an estimate of the time left.

* The input files are only read, never changed or deleted.
* `--overwrite` deletes the existing dataset at `--output` first, but **only files that a deltacells dataset has**. If the output
  holds anything else, the converter stops and deletes nothing. It also refuses to run if an input file lies inside the output.
  Without `--overwrite`, a non-empty output is an error.
* `manifest.json` is written last, so a directory or prefix without a manifest is not a dataset.
* If a conversion fails, for example on an unreadable file, the tiles written so far stay in place but there is no manifest, and
  the error says so. There is no resume: start again with `--overwrite`.

## From Python

```python
from deltacells.convert import convert_h5ad

convert_h5ad(["shard_0.h5ad", "shard_1.h5ad"], "my_dataset", tile_size=10_000)
```
