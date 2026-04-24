# op2_native/decoders/ougv1.py
from typing import List, Optional
import struct
import numpy as np
import pandas as pd

from ..models import OP2Inventory, OP2Record
from .oes_peek import collect_data_records_after, load_data_bytes

ROW_FMT = (
    "ii6f"  # dof_id, cp (or similar), then 6 DOFs  (endian prefix added at runtime)
)
ROW_SIZE = 32  # 2 x int32 + 6 x float32


def _row_dtype(endian: str = "<") -> np.dtype:
    """Return the numpy dtype for one OUGV1 row with the given byte order."""
    return np.dtype(
        [
            ("dof_id", f"{endian}i4"),
            ("cp", f"{endian}i4"),
            ("dx", f"{endian}f4"),
            ("dy", f"{endian}f4"),
            ("dz", f"{endian}f4"),
            ("rx", f"{endian}f4"),
            ("ry", f"{endian}f4"),
            ("rz", f"{endian}f4"),
        ]
    )


# Cached little-endian dtype for backwards compatibility
_ROW_DTYPE = _row_dtype("<")

OUT_COLS = ("GRID", "TX", "TY", "TZ", "RX", "RY", "RZ")


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
        skip_until = -1
        for r in inv.records:
            if r.info.index <= hdr_idx:
                continue
            if r.info.index >= next_boundary:
                break
            if r.info.index <= skip_until:
                continue
            if _looks_like_ougv1_data(r, inv.endian):
                result.append((hdr_idx, r.info.index, sc_offset))
                sc_offset += 1
                # Skip continuation records so they are not mistaken for new subcases.
                conts = collect_data_records_after(
                    inv, r.info.index, min_data_bytes=ROW_SIZE
                )
                if len(conts) > 1:
                    skip_until = conts[-1]
    return result


def _looks_like_ougv1_data(
    rec: OP2Record, endian: str = "<", min_good_rows: int = 5
) -> bool:
    L = rec.info.length
    if L < ROW_SIZE or (L % ROW_SIZE) != 0:
        return False

    arr = np.frombuffer(rec.data[: min_good_rows * ROW_SIZE], dtype=_row_dtype(endian))
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
    data_idx = (
        ekey_idx if ekey_idx is not None else find_ougv1_data_record(inv, header_index)
    )
    payload, _first, all_recs = load_data_bytes(
        inv, header_index, min_data_bytes=ROW_SIZE, first_idx=data_idx
    )

    # Trim to a whole number of rows then bulk-unpack with numpy
    n_rows = len(payload) // ROW_SIZE
    arr = np.frombuffer(payload[: n_rows * ROW_SIZE], dtype=_row_dtype(inv.endian))

    df = pd.DataFrame(
        {
            "GRID": (arr["dof_id"] // 10).astype(np.int32),
            "TX": arr["dx"],
            "TY": arr["dy"],
            "TZ": arr["dz"],
            "RX": arr["rx"],
            "RY": arr["ry"],
            "RZ": arr["rz"],
        }
    )
    df.attrs["source_header"] = header_index
    df.attrs["data_record"] = data_idx
    df.attrs["all_data_records"] = all_recs
    df.attrs["row_format"] = ROW_FMT
    return df
