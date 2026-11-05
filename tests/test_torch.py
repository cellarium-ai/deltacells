# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from conftest import make_counts, same

torch = pytest.importorskip("torch")

from torch.utils.data import IterableDataset  # noqa: E402

from deltacells import DatasetWriter, open_dataset  # noqa: E402

N, G, TILE = 240, 50, 60


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("t") / "ds")
    full = make_counts(N, G, density=0.2, seed=50)
    with DatasetWriter(root, n_genes=G, tile_size=TILE, n_chunks=2, level=3) as w:
        for lo in range(0, N, TILE):
            w.add_tile(full[lo : lo + TILE])
    return root, full


def test_to_torch_csr_matches_dense(built):
    root, full = built
    b = open_dataset(root).get_batch([7, 3, 200, 3])
    t = b.to_torch_csr()
    assert t.layout == torch.sparse_csr and t.shape == (4, G) and t.dtype == torch.float32
    assert torch.equal(t.to_dense(), torch.from_numpy(full[[7, 3, 200, 3]].toarray()))
    assert np.shares_memory(t.col_indices().numpy(), b.indices)  # no copy of the big arrays


def test_to_torch_csr_empty_batch(built):
    t = open_dataset(built[0]).get_batch([]).to_torch_csr()
    assert t.shape == (0, G)


def test_shared_memory_output_buffers(built):
    root, full = built
    ds = open_dataset(root)
    idx = np.arange(10, 90)
    need = full[idx].nnz
    oi, ov = (
        torch.empty(need, dtype=torch.int32).share_memory_(),
        torch.empty(need, dtype=torch.float32).share_memory_(),
    )
    b = ds.get_batch(idx, out_indices=oi.numpy(), out_values=ov.numpy())
    assert same(b.to_scipy(), full[idx]) and oi.is_shared()


class _Rows(IterableDataset):
    """Each worker yields the batches ``range(i*B, (i+1)*B)`` for its share of the batch numbers."""

    def __init__(self, ds, batch_size):
        self.ds, self.batch_size = ds, batch_size

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        n_batches = -(-self.ds.n_cells // self.batch_size)
        for b in range(wid, n_batches, nw):
            idx = np.arange(b * self.batch_size, min((b + 1) * self.batch_size, self.ds.n_cells))
            self.ds.prefetch_cells(idx)
            batch = self.ds.get_batch(idx)
            yield {
                "id": b,
                "indptr": torch.from_numpy(batch.indptr.astype(np.int32)),
                "indices": torch.from_numpy(batch.indices),
                "values": torch.from_numpy(batch.values),
            }


@pytest.mark.parametrize("workers", [0, 2])
def test_dataloader_with_workers(built, workers):
    root, full = built
    ds = open_dataset(root, max_cached_tiles=1)
    dl = torch.utils.data.DataLoader(_Rows(ds, 25), batch_size=None, num_workers=workers)
    got = {}
    for item in dl:
        csr = torch.sparse_csr_tensor(
            item["indptr"], item["indices"], item["values"], size=(len(item["indptr"]) - 1, G)
        )
        got[int(item["id"])] = csr.to_dense().numpy()
    assert sorted(got) == list(range(10))
    assert np.array_equal(np.vstack([got[b] for b in sorted(got)]), full.toarray())


def test_forked_dataloader_workers_after_an_in_process_read(built):
    # Workers forked from a process that has already read a tile (so that its fetch threads exist) must not wait for them.
    root, full = built
    ds = open_dataset(root, max_cached_tiles=1)
    ds.get_batch([0])
    dl = torch.utils.data.DataLoader(
        _Rows(ds, 25), batch_size=None, num_workers=2, multiprocessing_context="fork", timeout=60
    )
    got = {}
    for item in dl:
        csr = torch.sparse_csr_tensor(
            item["indptr"], item["indices"], item["values"], size=(len(item["indptr"]) - 1, G)
        )
        got[int(item["id"])] = csr.to_dense().numpy()
    assert np.array_equal(np.vstack([got[b] for b in sorted(got)]), full.toarray())


class _AlwaysCorrupt:
    """A fake GCS client whose every download fails its checksum."""

    def __init__(self, error):
        self.error = error

    def bucket(self, name):
        return self

    def blob(self, name):
        error = self.error

        class Blob:
            def download_as_bytes(self, start=None, end=None):
                raise error

        return Blob()


class _ReadsATile(IterableDataset):
    def __init__(self, backend):
        self.backend = backend

    def __iter__(self):
        yield self.backend.read("tiles/tile_000003.dct")


def test_corrupt_gcs_download_message_survives_a_dataloader_worker():
    # `DataCorruption` itself arrives blank from a worker (torch rebuilds it as `DataCorruption(message)`, which puts the
    # message where the HTTP response goes); `CorruptGCSDownloadError` must not.
    exceptions = pytest.importorskip("google.cloud.storage.exceptions")
    from deltacells import CorruptGCSDownloadError, GCSBackend

    error = exceptions.DataCorruption(None, "Checksum mismatch: expected 'AAAA==', got 'BBBB=='")
    backend = GCSBackend("bkt", "pre", client=_AlwaysCorrupt(error), max_attempts=1)
    # "fork": the fake client is not pickled, and the worker needs no other import to find it
    dl = torch.utils.data.DataLoader(
        _ReadsATile(backend), batch_size=None, num_workers=1, multiprocessing_context="fork"
    )
    with pytest.raises(CorruptGCSDownloadError) as info:
        next(iter(dl))
    for text in ("gs://bkt/pre/tiles/tile_000003.dct", "expected 'AAAA=='", "got 'BBBB=='"):
        assert text in str(info.value)
    with pytest.raises(exceptions.DataCorruption) as bare:  # the problem this guards against, demonstrated
        raise type(error)(str(error))
    assert str(bare.value) == ""
