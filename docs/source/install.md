# Install

```bash
pip install deltacells
```

Prebuilt wheels are available for Linux (x86-64 and aarch64) and macOS (arm64) on Python 3.10 to 3.13. They include everything
needed, so there is nothing else to install.

On other platforms, pip builds the package from source. That needs a C compiler and **libzstd** (headers and library):

```bash
conda install zstd                      # or: brew install zstd / apt install libzstd-dev
pip install deltacells
```

If libzstd is in an unusual place, set `ZSTD_ROOT` to its install prefix before running pip.

## Optional extras

The core package needs only `numpy` and `scipy`. Install extras for the features you use:

| extra | adds | needed for |
|---|---|---|
| `h5ad` | anndata | `deltacells convert` (reading h5ad files) |
| `obs` | pyarrow, pandas | storing and reading cell metadata (obs) and the gene table (var) |
| `torch` | PyTorch | `SparseBatch.to_torch_csr()` and the PyTorch data loading examples |
| `gcs` | google-cloud-storage, fsspec, gcsfs | reading and converting from and to `gs://` paths |

For example:

```bash
pip install "deltacells[h5ad,obs,gcs]"
```

## Check the installation

```bash
python -c "import deltacells; print(deltacells.__version__)"
deltacells --help
```

## Platforms

deltacells runs on Linux and macOS, on little-endian hosts. Windows is not supported. Continuous integration covers Linux x86-64
(Python 3.10 to 3.13) and macOS arm64 (Python 3.12).

## Installing for development

To work on deltacells itself, see [DEVELOPING.md](https://github.com/cellarium-ai/deltacells/blob/main/DEVELOPING.md) in the
repository.
