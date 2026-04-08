# op2_native/decoders/oqg1.py
"""
Decoder for OQG1 SPC/MPC constraint force data blocks.

Layout (SORT1, real, static) — 8 words per row:

      [packed_dof_id, cp, FX, FY, FZ, MX, MY, MZ]

  packed_dof_id = 10 * GRID + device_code.
  cp            = integer coordinate system / component flag (word 1).
  FX-MZ         = float32 force components (words 2-7).

  Some files use 7-word rows (no cp word):

      [packed_dof_id, FX, FY, FZ, MX, MY, MZ]

  We auto-detect the layout by checking whether the second word of the
  first few rows is an integer in a plausible cp range (0-999).
"""
from __future__ import annotations

import struct
from typing import List

import numpy as np
import pandas as pd

from typing import List

from ..models import OP2Inventory
from .oes_peek import first_grid_force_record_after, classify_grid_force_headers


def classify_oqg_headers(inv: OP2Inventory) -> List[tuple]:
    """Return ``[(header_idx, data_rec_idx, sc_offset), ...]`` for all OQG1 data blocks."""
    return classify_grid_force_headers(inv, "OQG1")


_COLS = ["GRID", "FX", "FY", "FZ", "MX", "MY", "MZ"]

# numpy dtypes for the two possible row widths
_DTYPE_8 = np.dtype(
    [
        ("packed_id", "<i4"),
        ("cp", "<i4"),
        ("fx", "<f4"),
        ("fy", "<f4"),
        ("fz", "<f4"),
        ("mx", "<f4"),
        ("my", "<f4"),
        ("mz", "<f4"),
    ]
)  # 32 bytes

_DTYPE_7 = np.dtype(
    [
        ("packed_id", "<i4"),
        ("fx", "<f4"),
        ("fy", "<f4"),
        ("fz", "<f4"),
        ("mx", "<f4"),
        ("my", "<f4"),
        ("mz", "<f4"),
    ]
)  # 28 bytes


def _decode_grid_force_payload(
    payload: bytes,
    endian: str = "<",
    max_id: int = 100_000_000,
) -> pd.DataFrame:
    """
    Decode an OQG1/OPG1 payload into a DataFrame using numpy bulk unpacking.

    Auto-detects 8-word rows (packed_id, cp, FX..MZ) vs
    7-word rows (packed_id, FX..MZ).
    """
    n_bytes = len(payload)
    if n_bytes < 28:  # minimum 1 row of 7 words
        return pd.DataFrame(columns=_COLS)

    # --- detect row width ---
    # Probe: if n_bytes is divisible by 32 and the cp field looks plausible, use 8-word.
    # Otherwise fall back to 7-word.
    row_width = 7
    if n_bytes >= 32:
        # peek at cp word (offset 4) of first two rows assuming 8-word stride
        cp0 = struct.unpack_from(f"{endian}i", payload, 4)[0]
        cp1 = struct.unpack_from(f"{endian}i", payload, 36)[0] if n_bytes >= 64 else cp0
        if 0 <= cp0 < 1000 and 0 <= cp1 < 1000 and (n_bytes % 32 == 0):
            row_width = 8

    if row_width == 8:
        dtype = _DTYPE_8 if endian == "<" else _DTYPE_8.newbyteorder(">")
        n_rows = n_bytes // 32
        arr = np.frombuffer(payload[: n_rows * 32], dtype=dtype)
        grids = arr["packed_id"] // 10
        mask = (grids >= 1) & (grids < max_id)
        arr = arr[mask]
        grids = grids[mask]
        return pd.DataFrame(
            {
                "GRID": grids.astype(np.int32),
                "FX": arr["fx"],
                "FY": arr["fy"],
                "FZ": arr["fz"],
                "MX": arr["mx"],
                "MY": arr["my"],
                "MZ": arr["mz"],
            }
        )
    else:
        dtype = _DTYPE_7 if endian == "<" else _DTYPE_7.newbyteorder(">")
        n_rows = n_bytes // 28
        arr = np.frombuffer(payload[: n_rows * 28], dtype=dtype)
        grids = arr["packed_id"] // 10
        mask = (grids >= 1) & (grids < max_id)
        arr = arr[mask]
        grids = grids[mask]
        return pd.DataFrame(
            {
                "GRID": grids.astype(np.int32),
                "FX": arr["fx"],
                "FY": arr["fy"],
                "FZ": arr["fz"],
                "MX": arr["mx"],
                "MY": arr["my"],
                "MZ": arr["mz"],
            }
        )


def decode_oqg1(
    inv: OP2Inventory, header_index: int, ekey_idx: int | None = None
) -> pd.DataFrame:
    """
    Decode an OQG1 SPC/MPC force block.

    Parameters
    ----------
    ekey_idx : int, optional
        Direct data-record index (as returned by classify_oqg_headers).  When
        provided the forward-scan heuristic is skipped.

    Returns
    -------
    DataFrame with columns ``GRID, FX, FY, FZ, MX, MY, MZ``.
    """
    data_idx = (
        ekey_idx
        if ekey_idx is not None
        else first_grid_force_record_after(inv, header_index)
    )
    rec = inv.records[data_idx]
    df = _decode_grid_force_payload(rec.data, endian=inv.endian)
    df.attrs["header_record"] = header_index
    df.attrs["data_record"] = data_idx
    return df


# ---------------------------------------------------------------------------
# OQGCF1 — Contact forces (structurally identical to OQG1)
# ---------------------------------------------------------------------------


def classify_oqgcf1_headers(inv: OP2Inventory) -> List[tuple]:
    """Return ``[(header_idx, data_rec_idx, sc_offset), ...]`` for OQGCF1 blocks."""
    return classify_grid_force_headers(inv, "OQGCF1")


# ---------------------------------------------------------------------------
# OSPDSI1 / OSPDS1 — Contact separation distances (2 words per node)
# ---------------------------------------------------------------------------

_SEP_DTYPE_LE = np.dtype([("packed_id", "<i4"), ("distance", "<f4")])
_SEP_DTYPE_BE = np.dtype([("packed_id", ">i4"), ("distance", ">f4")])

_SEP_COLS = ["GRID", "DISTANCE"]


def _record_looks_like_separation_data(
    rec, endian: str = "<", min_rows: int = 5
) -> bool:
    """Heuristic: 2-word-per-node separation distance data records."""
    import struct as _struct

    L = rec.info.length
    row_bytes = 8  # packed_grid_id(int32) + distance(float32)
    if L < row_bytes * min_rows or L % row_bytes != 0:
        return False
    sample = min(L // row_bytes, 8)
    for i in range(sample):
        raw = _struct.unpack_from(f"{endian}i", rec.data, i * row_bytes)[0]
        if raw <= 0:
            return False
        grid = raw // 10
        if not (1 <= grid < 10_000_000):
            return False
    return True


def classify_separation_headers(inv: OP2Inventory, token: str) -> List[tuple]:
    """
    Return ``[(header_idx, data_rec_idx, sc_offset), ...]`` for all
    OSPDSI1 or OSPDS1 data blocks.
    """
    token_bytes = token.encode("ascii").ljust(8)
    boundaries = sorted(r.info.index for r in inv.records if r.info.length == 8)

    result: List[tuple] = []
    for rec in inv.records:
        if rec.info.length != 8 or rec.data[:8] != token_bytes:
            continue
        hdr_idx = rec.info.index
        next_boundary = next(
            (b for b in boundaries if b > hdr_idx), len(inv.records) + 1
        )
        sc_offset = 0
        for r in inv.records:
            if r.info.index <= hdr_idx:
                continue
            if r.info.index >= next_boundary:
                break
            if _record_looks_like_separation_data(r, inv.endian):
                result.append((hdr_idx, r.info.index, sc_offset))
                sc_offset += 1
    return result


def _decode_separation_payload(
    payload: bytes,
    endian: str = "<",
    max_id: int = 100_000_000,
) -> pd.DataFrame:
    """Decode a 2-word-per-node separation distance payload."""
    n_bytes = len(payload)
    if n_bytes < 8:
        return pd.DataFrame(columns=_SEP_COLS)
    dtype = _SEP_DTYPE_LE if endian == "<" else _SEP_DTYPE_BE
    n_rows = n_bytes // 8
    arr = np.frombuffer(payload[: n_rows * 8], dtype=dtype)
    grids = arr["packed_id"] // 10
    mask = (grids >= 1) & (grids < max_id)
    return pd.DataFrame(
        {
            "GRID": grids[mask].astype(np.int32),
            "DISTANCE": arr["distance"][mask],
        }
    )


def decode_separation(
    inv: OP2Inventory, header_index: int, ekey_idx: int | None = None
) -> pd.DataFrame:
    """
    Decode an OSPDSI1 or OSPDS1 contact separation distance block.

    Returns
    -------
    DataFrame with columns ``GRID, DISTANCE``.
    """
    data_idx = ekey_idx if ekey_idx is not None else header_index + 1
    rec = inv.records[data_idx]
    df = _decode_separation_payload(rec.data, endian=inv.endian)
    df.attrs["header_record"] = header_index
    df.attrs["data_record"] = data_idx
    return df
