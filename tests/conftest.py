# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

import os
import sys

import numpy as np
import pytest
import scipy.sparse as sp


# Make the tests runnable straight after `python setup.py build_ext --inplace` (src layout), and skip collecting them when the
# package or its C extension is not available, so that running pytest from a parent repository does not fail.
# Note: from the parent repository's root, `import deltacells` finds the *folder* `deltacells/` as an empty namespace package,
# so check for a real attribute instead of just trying the import.
def _real_package_importable() -> bool:
    try:
        import deltacells
    except ImportError:
        return False
    return hasattr(deltacells, "DatasetWriter")


if not _real_package_importable():
    sys.modules.pop("deltacells", None)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    if not _real_package_importable():
        collect_ignore_glob = ["test_*.py"]


def make_counts(n_cells, n_genes, density=0.1, seed=0, max_value=50, dtype=np.float32):
    """A random CSR count matrix with integer values in [1, max_value]."""
    rng = np.random.default_rng(seed)
    m = sp.random(n_cells, n_genes, density=density, format="csr", random_state=rng, dtype=np.float64)
    m.data = rng.integers(1, max_value + 1, size=m.nnz).astype(dtype)
    m.sort_indices()
    return m


@pytest.fixture
def counts():
    return make_counts


def run_in_forked_child(fn, timeout=15):
    """Run ``fn()`` in a forked child process and return its result (it must be picklable); fail, rather than hang, if the
    child does not finish. The child inherits everything the test has built, as a DataLoader worker on Linux does."""
    import multiprocessing

    if not hasattr(os, "fork"):
        pytest.skip("needs fork")
    ctx = multiprocessing.get_context("fork")
    queue = ctx.Queue()

    def target():
        try:
            queue.put(("ok", fn()))
        except BaseException as e:  # reported to the parent
            queue.put(("error", repr(e)))

    child = ctx.Process(target=target, daemon=True)
    child.start()
    try:
        kind, value = queue.get(timeout=timeout)
    except Exception:
        kind, value = "hung", None
    finally:
        child.join(1)
        if child.is_alive():
            child.kill()
            child.join()
    if kind == "hung":
        pytest.fail(f"the forked process did not finish within {timeout} s (is it waiting for missing threads?)")
    assert kind == "ok", value
    return value


def same(a, b):
    """Exact equality of two sparse matrices (shape and values)."""
    a, b = sp.csr_matrix(a), sp.csr_matrix(b)
    return a.shape == b.shape and (a != b).nnz == 0
