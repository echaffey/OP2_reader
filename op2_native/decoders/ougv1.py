# op2_native/decoders/ougv1.py
from __future__ import annotations
from typing import List, Optional
import struct
import numpy as np
import pandas as pd

from ..models import OP2Inventory, OP2Record

ROW_FMT = "<ii6f"  # dof_id, cp (or similar), then 6 DOFs
ROW_SIZE = struct.calcsize(ROW_FMT)  # 32 bytes

# numpy dtype matching the row layout (little-endian, 2 int32 + 6 float32)
_ROW_DTYPE = np.dtype(
    [
        ("dof_id", "<i4"),
        ("cp", "<i4"),
        ("dx", "<f4"),
        ("dy", "<f4"),
        ("dz", "<f4"),
        ("rx", "<f4"),
        ("ry", "<f4"),
        ("rz", "<f4"),
    ]
)

OUT_COLS = ("GRID", "CP", "DX", "DY", "DZ", "RX", "RY", "RZ")


def find_ougv1_headers(inv: OP2Inventory) -> List[int]:
    hits: List[int] = []
    token = b"OUGV1"
    for rec in inv.records:
        if rec.info.length == 8 and token in rec.data:
            hits.append(rec.info.index)
    return hits


def classify_ougv1_headers(inv: OP2Inventory) -> List[tuple]:
    """
    Return a list of 3-tuples ``(header_idx, data_rec_idx, sc_offset)`` — one
    entry per displacement data block found within each OUGV1 table.

    Multiple subcases share one 8-byte table-name record with no repeated
    table name for later subcases, so we scan the whole table for all data
    blocks and assign sc_offset 0, 1, 2, … in order.
    """
    # Sorted indices of every 8-byte record (table name / boundary markers)
    boundaries = sorted(r.info.index for r in inv.records if r.info.length == 8)

    result: List[tuple] = []
    for rec in inv.records:
        if rec.info.length != 8 or b"OUGV1" not in rec.data:
            continue
        hdr_idx = rec.info.index
        # Next 8-byte boundary marks end of this table
        next_boundary = next(
            (b for b in boundaries if b > hdr_idx), len(inv.records) + 1
        )
        sc_offset = 0
        for r in inv.records:
            if r.info.index <= hdr_idx:
                continue
            if r.info.index >= next_boundary:
                break
            if _looks_like_ougv1_data(r):
                result.append((hdr_idx, r.info.index, sc_offset))
                sc_offset += 1
    return result


def _looks_like_ougv1_data(rec: OP2Record, min_good_rows: int = 5) -> bool:
    L = rec.info.length
    if L < ROW_SIZE or (L % ROW_SIZE) != 0:
        return False

    arr = np.frombuffer(rec.data[: min_good_rows * ROW_SIZE], dtype=_ROW_DTYPE)
    grids = arr["dof_id"] // 10
    cps = arr["cp"]
    if not (np.all(grids >= 1) & np.all(grids < 100_000_000)):
        return False
    if not (np.all(cps >= 0) & np.all(cps < 1000)):
        return False
    # all displacement values finite and magnitude < 1e6
    dofs = np.stack([arr[k] for k in ("dx", "dy", "dz", "rx", "ry", "rz")], axis=1)
    if not np.all(np.isfinite(dofs) & (np.abs(dofs) < 1e6)):
        return False
    return True


def find_ougv1_data_record(inv: OP2Inventory, header_index: int) -> int:
    start = header_index + 1
    for rec in inv.records:
        if rec.info.index < start:
            continue
        if _looks_like_ougv1_data(rec):
            return rec.info.index
    raise ValueError(f"No OUGV1 data record found after header {header_index}")


def decode_ougv1(
    inv: OP2Inventory, header_index: int, ekey_idx: Optional[int] = None
) -> pd.DataFrame:
    # ekey_idx is used here as the direct data record index (returned by
    # classify_ougv1_headers).  Fall back to forward scan when not provided.
    data_idx = ekey_idx if ekey_idx is not None else find_ougv1_data_record(inv, header_index)
    rec = next(r for r in inv.records if r.info.index == data_idx)
    data = rec.data
    L = rec.info.length

    # Trim to a whole number of rows then bulk-unpack with numpy
    n_rows = L // ROW_SIZE
    arr = np.frombuffer(data[: n_rows * ROW_SIZE], dtype=_ROW_DTYPE)

    df = pd.DataFrame(
        {
            "GRID": (arr["dof_id"] // 10).astype(np.int32),
            "CP": arr["cp"].astype(np.int32),
            "DX": arr["dx"],
            "DY": arr["dy"],
            "DZ": arr["dz"],
            "RX": arr["rx"],
            "RY": arr["ry"],
            "RZ": arr["rz"],
        }
    )
    df.attrs["source_header"] = header_index
    df.attrs["data_record"] = data_idx
    df.attrs["row_format"] = ROW_FMT
    return df
