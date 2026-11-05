# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import pickle
import time

import pytest
from conftest import run_in_forked_child

from deltacells import CorruptGCSDownloadError, GCSBackend, LocalBackend, ThrottledBackend, open_backend


def test_local_backend(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.bin").write_bytes(b"hello")
    b = LocalBackend(str(tmp_path))
    assert bytes(b.read("sub/x.bin")) == b"hello" and b.exists("sub/x.bin") and not b.exists("nope")
    with pytest.raises(FileNotFoundError):
        b.read("nope")
    assert pickle.loads(pickle.dumps(b)).root == b.root


def test_open_backend_dispatch(tmp_path):
    assert isinstance(open_backend(str(tmp_path)), LocalBackend)
    assert isinstance(open_backend(f"file://{tmp_path}"), LocalBackend)
    g = open_backend("gs://my-bucket/some/prefix/")
    assert isinstance(g, GCSBackend) and g.describe() == "gs://my-bucket/some/prefix"
    assert open_backend(g) is g
    with pytest.raises(ValueError, match="scheme"):
        open_backend("s3://bucket/x")


class _FakeBlob:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def download_as_bytes(self):
        return self.store[self.name]

    def exists(self):
        return self.name in self.store


class _FakeClient:
    def __init__(self, store):
        self.store = store

    def bucket(self, name):
        return self

    def blob(self, name):
        return _FakeBlob(self.store, name)


def test_gcs_backend_with_fake_client_and_pickling():
    store = {"pre/fix/manifest.json": b"{}", "pre/fix/tiles/t": b"abc"}
    b = GCSBackend("bkt", "pre/fix", client=_FakeClient(store))
    assert b.read("tiles/t") == b"abc" and b.exists("manifest.json") and not b.exists("zzz")
    state = b.__getstate__()
    assert state["_client"] is None and state["_bucket"] is None  # not pickled: workers create their own client


class _FlakyClient:
    """A fake client whose downloads raise ``error`` (made fresh each time by ``make_error``) the first ``failures`` times."""

    def __init__(self, make_error, failures, payload=b"0123456789"):
        self.make_error, self.failures, self.payload = make_error, failures, payload
        self.calls = []  # (object name, start, end) of every download attempt

    def bucket(self, name):
        return self

    def blob(self, name):
        client = self

        class Blob:
            def download_as_bytes(self, start=None, end=None):
                client.calls.append((name, start, end))
                if len(client.calls) <= client.failures:
                    raise client.make_error()
                return client.payload if start is None else client.payload[start : end + 1]

        return Blob()


@pytest.fixture
def corruption():
    """The exception ``google-cloud-storage`` raises for a checksum mismatch, as a function making one."""
    exceptions = pytest.importorskip("google.cloud.storage.exceptions")
    return lambda: exceptions.DataCorruption(None, "Checksum mismatch: expected 'AAAA==', got 'BBBB=='")


@pytest.fixture
def sleeps(monkeypatch):
    """Record the pauses between attempts instead of waiting."""
    paused = []
    monkeypatch.setattr(time, "sleep", paused.append)
    return paused


@pytest.mark.parametrize("failures", [1, 2, 3])
def test_gcs_backend_repeats_a_download_that_fails_its_checksum(corruption, sleeps, caplog, failures):
    client = _FlakyClient(corruption, failures)
    b = GCSBackend("bkt", "pre", client=client)  # 4 attempts by default
    with caplog.at_level("WARNING", logger="deltacells.backends"):
        assert b.read("tiles/t") == b"0123456789"
    assert len(client.calls) == failures + 1 and {c[0] for c in client.calls} == {"pre/tiles/t"}
    assert len(sleeps) == failures
    for k, pause in enumerate(sleeps):  # 0.5 s, 1 s, 2 s, each lengthened by up to 50%
        assert 0.5 * 2**k <= pause <= 0.75 * 2**k
    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == failures and all("gs://bkt/pre/tiles/t" in w and "Checksum mismatch" in w for w in warnings)


def test_gcs_backend_repeats_ranged_reads_too(corruption, sleeps):
    client = _FlakyClient(corruption, failures=2)
    assert GCSBackend("bkt", client=client).read_range("tiles/t", 2, 3) == b"234"
    assert client.calls == [("tiles/t", 2, 4)] * 3 and len(sleeps) == 2  # the same inclusive range every time


def test_gcs_backend_gives_up_with_an_error_that_names_the_object(corruption, sleeps):
    client = _FlakyClient(corruption, failures=99)
    b = GCSBackend("bkt", "pre/fix", client=client)
    with pytest.raises(CorruptGCSDownloadError) as info:
        b.read("tiles/tile_000007.dct")
    assert len(client.calls) == 4 and len(sleeps) == 3  # no pause after the last attempt
    err = info.value
    assert isinstance(err, OSError) and isinstance(err.__cause__, type(corruption()))
    for text in ("gs://bkt/pre/fix/tiles/tile_000007.dct", "all 4 attempts", "expected 'AAAA=='", "got 'BBBB=='"):
        assert text in str(err)
    # What a DataLoader worker does to an exception on its way to the main process must not lose the message.
    assert str(type(err)(str(err))) == str(err) and str(pickle.loads(pickle.dumps(err))) == str(err)


def test_gcs_backend_does_not_repeat_other_errors(corruption, sleeps):
    client = _FlakyClient(lambda: ConnectionError("reset"), failures=99)
    with pytest.raises(ConnectionError, match="reset"):
        GCSBackend("bkt", client=client).read("x")
    assert len(client.calls) == 1 and sleeps == []


def test_gcs_backend_max_attempts(corruption, sleeps):
    client = _FlakyClient(corruption, failures=99)
    with pytest.raises(CorruptGCSDownloadError, match="all 1 attempts"):
        GCSBackend("bkt", client=client, max_attempts=1).read("x")
    assert len(client.calls) == 1 and sleeps == []
    client = _FlakyClient(corruption, failures=5)
    assert GCSBackend("bkt", client=client, max_attempts=6).read("x") == b"0123456789" and len(client.calls) == 6
    with pytest.raises(ValueError, match="max_attempts"):
        GCSBackend("bkt", max_attempts=0)
    assert pickle.loads(pickle.dumps(GCSBackend("bkt", max_attempts=7))).max_attempts == 7


def test_gcs_backend_forked_process_makes_its_own_client(monkeypatch):
    # A client holds open connections, which a forked process must not share with its parent.
    storage = pytest.importorskip("google.cloud.storage")
    made = []

    class Client(_FakeClient):
        def __init__(self):
            super().__init__({"p/x": b"abc"})
            made.append(self)

    monkeypatch.setattr(storage, "Client", Client)
    b = GCSBackend("bkt", "p")
    assert b.read("x") == b"abc" and len(made) == 1

    def child():
        before = (b._client is None, b._bucket is None)
        return before, b.read("x"), len(made)

    assert run_in_forked_child(child) == ((True, True), b"abc", 2)  # none inherited, then one made
    assert b._client is made[0] and len(made) == 1  # the parent keeps its own


def test_gcs_backend_forked_process_keeps_a_client_it_was_given():
    client = _FakeClient({"p/x": b"abc"})
    b = GCSBackend("bkt", "p", client=client)  # it cannot make another one like it
    assert b.read("x") == b"abc"
    assert run_in_forked_child(lambda: (b._client is client, b.read("x"))) == (True, b"abc")


def test_unpickled_gcs_backend_is_also_reset_in_a_forked_process(monkeypatch):
    storage = pytest.importorskip("google.cloud.storage")
    made = []

    class Client(_FakeClient):
        def __init__(self):
            super().__init__({"p/x": b"abc"})
            made.append(self)

    monkeypatch.setattr(storage, "Client", Client)
    b = pickle.loads(pickle.dumps(GCSBackend("bkt", "p", client=_FakeClient({}))))  # its given client is not pickled
    assert b.read("x") == b"abc" and len(made) == 1
    assert run_in_forked_child(lambda: (b._client is None, b.read("x"), len(made))) == (True, b"abc", 2)


def test_throttled_backend(tmp_path):
    (tmp_path / "x").write_bytes(b"0" * 1_000_000)
    b = ThrottledBackend(LocalBackend(str(tmp_path)), latency_s=0.1, bandwidth_bytes_per_s=5_000_000)  # 0.1 + 0.2 s
    t0 = time.perf_counter()
    assert len(b.read("x")) == 1_000_000
    assert 0.28 <= time.perf_counter() - t0 < 0.6
    assert b.exists("x") and "throttled" in b.describe()
