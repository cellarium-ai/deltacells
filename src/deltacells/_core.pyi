# Copyright Contributors to the Cellarium project.
# SPDX-License-Identifier: BSD-3-Clause

"""Type stub for the C extension ``_core.c``. The array arguments are any objects supporting the buffer protocol (read-only
ones for inputs, writable ones for outputs); offsets and lengths are non-negative integers."""

from typing import Any

# any object supporting the buffer protocol: ``bytes``, ``memoryview``, ``mmap``, a contiguous ``numpy`` array, ...
Buffer = Any

def decode_chunk(
    blob: Buffer,
    gaps_off: int,
    gaps_len: int,
    vals_off: int,
    vals_len: int,
    cell_lo: int,
    cell_hi: int,
    nnz_lo: int,
    nnz_hi: int,
    n_genes: int,
    indptr: Buffer,
    scratch: Buffer,
    out_indices: Buffer,
    out_values: Buffer,
    to_float: bool,
) -> None: ...
def apply_exceptions(
    blob: Buffer,
    exc_off: int,
    exc_len: int,
    kind: int,
    cell_lo: int,
    cell_hi: int,
    nnz_lo: int,
    nnz_hi: int,
    n_genes: int,
    indptr: Buffer,
    out_indices: Buffer,
    out_values: Buffer,
    to_float: bool,
) -> None: ...
def gather_rows(
    src_indptr: Buffer,
    src_indices: Buffer,
    src_values: Buffer,
    rows: Buffer,
    out_offsets: Buffer,
    out_indices: Buffer,
    out_values: Buffer,
    convert_to_float: bool,
) -> None: ...
def compress_bytes(src: Buffer, level: int) -> bytes: ...
def compress_bound(n_elements: int) -> int: ...
def compress_stream(src: Buffer, level: int, out: Buffer) -> int: ...
