# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""A batch that touches more tiles than ``max_prefetch_tiles`` is read as a stream: the indptr of each tile is read first (a
small ranged read), then the tiles are fetched, decoded and gathered through a window, never all held at once."""

import weakref

import numpy as np
import pytest
import scipy.sparse as sp
from conftest import make_counts, same

import deltacells.dataset as dataset_module
from deltacells import (
    Backend,
    CountingBackend,
    DatasetWriter,
    FormatError,
    LocalBackend,
    Manifest,
    open_dataset,
)

N_TILES, TILE, N_GENES, CHUNKS = 30, 20, 40, 3
N_CELLS = N_TILES * TILE - 7  # the last tile is smaller
PREFIX_MAX = 64 + 96 * CHUNKS + 8 * (TILE + 1)  # header, the most chunks, indptr: the most a prefix read asks for


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """(root, the matrix as stored). Tile 5 is nearly empty (so it has fewer chunks than the others)."""
    root = str(tmp_path_factory.mktemp("stream") / "ds")
    full = make_counts(N_CELLS, N_GENES, density=0.2, seed=71).tolil()
    full[5 * TILE : 6 * TILE, :] = 0
    full[5 * TILE + 3, [1, 2]] = [4, 9]
    full = full.tocsr()
    full.eliminate_zeros()
    with DatasetWriter(root, n_genes=N_GENES, tile_size=TILE, n_chunks=CHUNKS, level=3) as w:
        for lo in range(0, N_CELLS, TILE):
            w.add_tile(full[lo : lo + TILE])
    return root, full


def scattered(n, seed=0):
    """``n`` cells from all over, in no order, with repeats."""
    return np.random.default_rng(seed).integers(0, N_CELLS, size=n)


def counting(root, **kwargs):
    backend = CountingBackend(LocalBackend(root))
    ds = open_dataset(backend, **kwargs)
    backend.reset()
    return ds, backend


def whole_reads(backend):
    return [name for name, offset, _ in backend.log if offset is None]


def ranged_reads(backend):
    return [(name, offset, length) for name, offset, length in backend.log if offset is not None]


# ------------------------------------------------------------------------------------------------ the result


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.int32])
def test_scattered_batch_matches_scipy(built, seed, dtype):
    root, full = built
    ds = open_dataset(root, max_prefetch_tiles=3, max_cached_tiles=1)
    idx = scattered(150, seed)
    assert len(set(idx // TILE)) > 3  # more tiles than max_prefetch_tiles: the streaming path
    batch = ds.get_batch(idx, values_dtype=dtype)
    assert same(batch.to_scipy(), full[idx]) and batch.values.dtype == dtype
    assert ds.stats["streaming_batches"] == 1


def test_every_cell_in_reverse_and_repeated(built):
    root, full = built
    ds = open_dataset(root, max_prefetch_tiles=2, max_cached_tiles=1, io_threads=2)
    idx = np.concatenate([np.arange(N_CELLS)[::-1], np.arange(0, N_CELLS, 7)])
    assert same(ds.get_batch(idx).to_scipy(), full[idx])


def test_batch_nnz_and_preallocated_outputs(built):
    root, full = built
    ds = open_dataset(root, max_prefetch_tiles=2)
    idx = scattered(100, 3)
    nnz = ds.batch_nnz(idx)
    assert nnz == full[idx].nnz
    oi, ov = np.empty(nnz, dtype=np.int32), np.empty(nnz, dtype=np.float32)
    batch = ds.get_batch(idx, out_indices=oi, out_values=ov)
    assert same(batch.to_scipy(), full[idx]) and np.shares_memory(batch.indices, oi)


def test_verify_still_checks_tiles_in_the_stream(built, tmp_path):
    import shutil
    from pathlib import Path

    root = str(tmp_path / "c")
    shutil.copytree(built[0], root)
    p = Path(root) / "tiles" / "tile_000009.dct"
    data = bytearray(p.read_bytes())
    data[-5] ^= 0xFF
    p.write_bytes(data)
    with pytest.raises(FormatError, match="CRC"):
        open_dataset(root, verify=True, max_prefetch_tiles=2).get_batch(np.arange(0, N_CELLS, 5))


def test_a_tile_with_fewer_chunks_than_the_others(built):
    root, full = built
    m = Manifest.from_json(open(f"{root}/manifest.json", "rb").read())
    assert m.n_chunks == CHUNKS
    idx = np.array([5 * TILE + 3, 0, 5 * TILE + 3, 8 * TILE, 2 * TILE, N_CELLS - 1])
    ds = open_dataset(root, max_prefetch_tiles=2)
    assert same(ds.get_batch(idx).to_scipy(), full[idx])


# ---------------------------------------------------------------------- what is read: the fast path is not touched


def test_a_batch_within_max_prefetch_tiles_is_not_streamed(built):
    root, full = built
    ds, backend = counting(root, max_prefetch_tiles=4)
    idx = np.array([1, 25, 45, 65, 3, 22])  # tiles 0 1 2 3
    assert same(ds.get_batch(idx).to_scipy(), full[idx])
    assert ranged_reads(backend) == [] and len(whole_reads(backend)) == 4  # one whole read per tile, nothing else
    assert ds.stats["streaming_batches"] == 0 and ds.stats["prefix_reads"] == 0
    ds.batch_nnz(idx)  # likewise
    assert ranged_reads(backend) == []


def test_one_tile_at_a_time_reads_are_unchanged(built):
    root, full = built
    ds, backend = counting(root)  # the defaults: how a shard-at-a-time loader reads
    for t in range(N_TILES):
        lo, hi = t * TILE, min((t + 1) * TILE, N_CELLS)
        assert same(ds.get_batch(np.arange(lo, hi)).to_scipy(), full[lo:hi])
    assert ranged_reads(backend) == [] and sorted(whole_reads(backend)) == [
        Manifest.tile_name(t) for t in range(N_TILES)
    ]


def test_a_streamed_batch_reads_each_tile_whole_once_and_its_indptr_once(built):
    root, full = built
    ds, backend = counting(root, max_prefetch_tiles=3, max_cached_tiles=1)
    idx = scattered(200, 4)
    tiles = sorted(set((idx // TILE).tolist()))
    assert same(ds.get_batch(idx).to_scipy(), full[idx])
    assert sorted(whole_reads(backend)) == [Manifest.tile_name(t) for t in tiles]
    ranged = ranged_reads(backend)
    assert sorted(name for name, _, _ in ranged) == [Manifest.tile_name(t) for t in tiles]
    assert all(offset == 0 and length <= PREFIX_MAX for _, offset, length in ranged)
    assert ds.stats["prefix_reads"] == len(tiles)


def test_batch_nnz_of_a_big_batch_reads_no_whole_tile(built):
    root, full = built
    ds, backend = counting(root, max_prefetch_tiles=2)
    idx = scattered(120, 5)
    assert ds.batch_nnz(idx) == full[idx].nnz
    assert whole_reads(backend) == [] and len(ranged_reads(backend)) == len(set((idx // TILE).tolist()))


def test_tiles_that_are_cached_are_not_read_again_for_their_indptr(built):
    root, full = built
    ds, backend = counting(root, max_prefetch_tiles=4)
    ds.prefetch([2, 7])
    ds.get_tile(2), ds.get_tile(7)  # both fetched and parsed
    backend.reset()
    idx = np.array([2 * TILE + 1, 7 * TILE, 9 * TILE, 11 * TILE, 13 * TILE, 15 * TILE + 3])  # 6 tiles > 4
    assert same(ds.get_batch(idx).to_scipy(), full[idx])
    assert sorted(name for name, _, _ in ranged_reads(backend)) == [Manifest.tile_name(t) for t in (9, 11, 13, 15)]
    assert sorted(whole_reads(backend)) == [Manifest.tile_name(t) for t in (9, 11, 13, 15)]  # 2 and 7 were cached


# ------------------------------------------------------------------------------------------------ memory


def test_the_tiles_held_at_once_are_bounded_by_the_window(built, monkeypatch):
    root, full = built
    ds = open_dataset(root, max_prefetch_tiles=3, max_cached_tiles=1, io_threads=2)
    alive: list[weakref.ref] = []
    fetch = ds._fetch_tile

    def spy_fetch(i):
        tile = fetch(i)
        alive.append(weakref.ref(tile))
        return tile

    peaks = []
    gather = dataset_module.gather_into

    def spy_gather(*args, **kwargs):
        peaks.append(sum(r() is not None for r in alive))
        return gather(*args, **kwargs)

    monkeypatch.setattr(ds, "_fetch_tile", spy_fetch)
    monkeypatch.setattr(dataset_module, "gather_into", spy_gather)
    idx = np.arange(0, N_CELLS, 3)  # all 30 tiles
    assert same(ds.get_batch(idx).to_scipy(), full[idx])
    assert len(peaks) == N_TILES and max(peaks) <= 3 + 2, peaks  # the window, the tile being gathered, one finishing


# ---------------------------------------------------------------------------------------- backends


class _ReadOnly(Backend):
    """A backend written with the two required methods only: it has no ranged reads."""

    def __init__(self, root):
        self.inner = LocalBackend(root)

    def read(self, name):
        return self.inner.read(name)

    def exists(self, name):
        return self.inner.exists(name)


def test_the_capability_of_the_backends(built):
    local = LocalBackend(built[0])
    assert local.ranged_reads and CountingBackend(local).ranged_reads
    assert not _ReadOnly(built[0]).ranged_reads and not CountingBackend(_ReadOnly(built[0])).ranged_reads
    from deltacells import GCSBackend, ThrottledBackend

    assert GCSBackend("b").ranged_reads and ThrottledBackend(local).ranged_reads


def test_a_backend_without_ranged_reads_is_refused_for_a_big_batch(built):
    root, full = built
    ds = open_dataset(_ReadOnly(root), max_prefetch_tiles=3)
    idx = np.array([1, 25, 45])  # 3 tiles: within max_prefetch_tiles, so read as ever
    assert same(ds.get_batch(idx).to_scipy(), full[idx])
    with pytest.raises(NotImplementedError, match=r"read_range.*max_prefetch_tiles"):
        ds.get_batch(np.arange(0, N_CELLS, 3))
    with pytest.raises(NotImplementedError):
        ds.batch_nnz(np.arange(0, N_CELLS, 3))
    ds2 = open_dataset(_ReadOnly(root), max_prefetch_tiles=N_TILES)  # the way out: hold the tiles
    assert same(ds2.get_batch(np.arange(0, N_CELLS, 3)).to_scipy(), full[np.arange(0, N_CELLS, 3)])


# ---------------------------------------------------------------------------------- bad prefixes


class _Tamper(LocalBackend):
    """Change what a ranged read of the start of some tiles returns (``fn(tile name, data, call number)``)."""

    def __init__(self, root, fn):
        super().__init__(root)
        self.fn, self.calls = fn, {}

    def read_range(self, name, offset, length):
        data = super().read_range(name, offset, length)
        self.calls[name] = self.calls.get(name, 0) + 1
        return self.fn(name, data, self.calls[name])


def test_a_corrupt_prefix_is_read_again(built):
    root, full = built
    bad = Manifest.tile_name(4)

    def fn(name, data, n):
        return b"X" + data[1:] if name == bad and n <= 2 else data  # a flipped byte in the magic, twice

    backend = _Tamper(root, fn)
    ds = open_dataset(backend, max_prefetch_tiles=2)
    idx = np.arange(0, 10 * TILE, 5)
    assert same(ds.get_batch(idx).to_scipy(), full[idx])
    assert backend.calls[bad] == 3


def test_a_prefix_that_stays_invalid_says_which_tile_is_malformed(built):
    root, _ = built
    bad = Manifest.tile_name(4)
    backend = _Tamper(
        root, lambda name, data, n: data[:-30] if name == bad else data
    )  # truncated: cuts into the indptr
    ds = open_dataset(backend, max_prefetch_tiles=2)
    with pytest.raises(FormatError) as info:
        ds.get_batch(np.arange(0, 10 * TILE, 5))
    message = str(info.value)
    assert bad in message and "malformed" in message and "tile 4" in message and "conversion" in message
    assert backend.calls[bad] == 3  # three attempts
    assert ds.get_batch([0]).n_rows == 1  # the dataset is still usable


def _forged_prefix(root, name, length):
    """The first ``length`` bytes of a tile with one cell boundary of its indptr moved: a start that passes every check made
    of it (it is monotone, and agrees with the header, the chunk table and the manifest) but gives other cell lengths than
    the tile does."""
    import struct

    blob = bytearray(LocalBackend(root).read(name).tobytes())
    _, _, _, n_cells, nnz, _, n_chunks, indptr_off, _, _, _ = struct.unpack_from("<8sIIQQIIQQII", blob, 0)
    indptr = np.frombuffer(blob, dtype="<i8", count=n_cells + 1, offset=indptr_off).copy()
    chunk_bounds = set()
    for i in range(n_chunks):
        lo, hi = struct.unpack_from("<QQ", blob, 64 + 96 * i)
        chunk_bounds |= {lo, hi}
    for j in range(1, n_cells):
        mid = (indptr[j - 1] + indptr[j + 1]) // 2
        if j not in chunk_bounds and mid != indptr[j]:  # a boundary inside a chunk that can move and stay monotone
            indptr[j] = mid
            blob[indptr_off : indptr_off + 8 * (n_cells + 1)] = indptr.tobytes()
            return bytes(blob[:length])
    raise AssertionError("no cell boundary to move")


def test_a_prefix_that_disagrees_with_its_tile_is_never_trusted(built):
    # The prefix passes every check of its own, but the lengths of the cells differ from those in the tile that follows.
    root, _ = built
    bad = Manifest.tile_name(7)
    backend = _Tamper(root, lambda name, data, n: _forged_prefix(root, name, len(data)) if name == bad else data)
    ds = open_dataset(backend, max_prefetch_tiles=2)
    with pytest.raises(FormatError) as info:
        ds.get_batch(np.arange(0, 10 * TILE, 2))
    assert bad in str(info.value) and "tile 7" in str(info.value) and "prefix" in str(info.value)
    assert backend.calls[bad] == 2  # the layout was redone once, then the error


def test_a_transient_disagreement_is_healed_by_redoing_the_layout(built):
    root, full = built
    bad = Manifest.tile_name(7)
    backend = _Tamper(
        root, lambda name, data, n: _forged_prefix(root, name, len(data)) if name == bad and n == 1 else data
    )
    ds = open_dataset(backend, max_prefetch_tiles=2)
    idx = np.arange(0, 10 * TILE, 2)
    assert same(ds.get_batch(idx).to_scipy(), full[idx])
    assert backend.calls[bad] == 2


# ------------------------------------------------------------------------------------------- failures


def test_a_failed_fetch_in_the_stream_propagates_and_leaves_the_dataset_usable(built):
    root, full = built

    class Flaky(LocalBackend):
        fail = True

        def read(self, name):
            if self.fail and name == Manifest.tile_name(12):
                raise OSError("connection reset")
            return super().read(name)

    backend = Flaky(root)
    ds = open_dataset(backend, max_prefetch_tiles=3, io_threads=2)
    idx = np.arange(0, N_CELLS, 4)
    with pytest.raises(OSError, match="connection reset"):
        ds.get_batch(idx)
    backend.fail = False
    assert same(ds.get_batch(idx).to_scipy(), full[idx])
    ds.close()


def test_the_matrix_is_what_was_written(built):
    # a guard on the fixture: tile 5 is the nearly empty one
    root, full = built
    assert isinstance(full, sp.csr_matrix) and full[5 * TILE : 6 * TILE].nnz == 2
