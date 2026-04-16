# op2_native/decoders/oes_peek.py
from __future__ import annotations
from typing import List, Literal, Dict
import struct
import math
import pandas as pd

from ..models import OP2Inventory


def _view_record_as(
    inv: OP2Inventory,
    rec_index: int,
    kind: Literal["int32", "float32", "float64"],
    cols: int = 8,
    max_items: int | None = 128,
) -> pd.DataFrame:
    b = inv.records[rec_index].data
    step = 4 if kind in ("int32", "float32") else 8
    n = len(b) // step
    if max_items is not None:
        n = min(n, max_items)
    vals = []
    if kind == "int32":
        fmt = "<i"
    elif kind == "float32":
        fmt = "<f"
    else:
        fmt = "<d"
    unpack = struct.unpack
    for i in range(n):
        vals.append(unpack(fmt, b[i * step : (i + 1) * step])[0])
    rows = math.ceil(len(vals) / cols)
    data = [vals[r * cols : (r + 1) * cols] for r in range(rows)]
    df = pd.DataFrame(data)
    df.attrs["rec_index"] = rec_index
    df.attrs["length_bytes"] = len(b)
    df.attrs["kind"] = kind
    return df


# op2_native/decoders/oes_peek.py


def _record_looks_like_header(rec) -> bool:
    """
    Heuristic: OES header/control records are often:
      - relatively small, OR
      - dominated by 0, -1, and 0x20202020 (= 538976288) padding.
    If that’s the case, we *don’t* want to treat them as data.
    """
    L = rec.info.length
    if L < 64:
        return True  # tiny → header/control

    # look at first 64 bytes as int32
    import struct

    head = rec.data[:64]
    ints = []
    for i in range(0, len(head), 4):
        if i + 4 > len(head):
            break
        ints.append(struct.unpack("<i", head[i : i + 4])[0])

    if not ints:
        return True

    # count common header-ish values
    zeros = sum(1 for v in ints if v == 0)
    neg_ones = sum(1 for v in ints if v == -1)
    spaces = sum(1 for v in ints if v == 538976288)  # 0x20202020 => "    "

    # if most ints are 0/-1/spaces, it's very likely a header/filler record
    if (zeros + neg_ones + spaces) / len(ints) > 0.7:
        return True

    return False


def _record_looks_like_stress_data(rec, endian: str = "<") -> bool:
    """
    Heuristic: stress data records are:
      - reasonably large, and
      - have many finite float32 values of nontrivial magnitude.
    """
    L = rec.info.length
    if L < 64:
        return False

    import struct

    head = rec.data[:256]  # small sample
    floats = []
    for i in range(0, len(head), 4):
        if i + 4 > len(head):
            break
        floats.append(struct.unpack(f"{endian}f", head[i : i + 4])[0])

    if not floats:
        return False

    import math

    finite = [v for v in floats if math.isfinite(v)]
    if not finite:
        return False

    # count values that aren't essentially zero, but also not astronomically huge
    interesting = [v for v in finite if 1e-3 < abs(v) < 1e9]
    # require at least a handful of "interesting" floats
    return len(interesting) >= 8


def first_stress_record_after(
    inv: OP2Inventory, header_index: int, max_ahead: int = 80
) -> int:
    """
    Starting after header_index, return index of the first record that
    does *not* look like a header and *does* look like stress data.
    """
    start = header_index + 1
    end = min(len(inv.records), start + max_ahead)
    for i in range(start, end):
        rec = inv.records[i]
        if _record_looks_like_header(rec):
            continue
        if _record_looks_like_stress_data(rec, inv.endian):
            return i
    raise ValueError(f"No stress-like record found after header {header_index}")


def first_data_record_after_ekey(
    inv: OP2Inventory,
    ekey_index: int,
    min_data_bytes: int = 1000,
    max_ahead: int = 30,
) -> int:
    """
    Return the index of the first record with ``length >= min_data_bytes``
    after *ekey_index*, without applying any content heuristic.

    This is suitable when the caller already knows the position of an EKEY
    record and just needs the next data block.  The content-based
    ``first_stress_record_after`` heuristic can fail for element types with
    many zero-padded words (e.g. CBEAM with 111-word / 11-station layout
    where only 1–2 stations are populated).
    """
    start = ekey_index + 1
    end = min(len(inv.records), start + max_ahead)
    for i in range(start, end):
        if inv.records[i].info.length >= min_data_bytes:
            return i
    raise ValueError(
        f"No data record (>= {min_data_bytes}B) found within "
        f"{max_ahead} records after ekey {ekey_index}"
    )


def collect_data_records_after(
    inv: OP2Inventory, first_data_idx: int, min_data_bytes: int = 1000
) -> list:
    """
    Starting from *first_data_idx* (a large data record), collect all
    subsequent data records that belong to the same logical data block.

    Nastran OP2 splits oversized result tables into multiple Fortran records.
    Each pair of data records is separated by exactly one 4-byte 'key' record.
    The sequence ends when two or more consecutive small records appear (which
    signals the start of the next table section).

    Parameters
    ----------
    inv : OP2Inventory
    first_data_idx : int
        Index of the first data record (returned by
        :func:`first_stress_record_after`).
    min_data_bytes : int
        Minimum byte length to be considered a "data" record.
        Default 1000.

    Returns
    -------
    list of int
        Record indices of all data records (large blocks only, no separators).
    """
    records = [first_data_idx]
    i = first_data_idx + 1
    while i < len(inv.records):
        r = inv.records[i]
        if r.info.length >= min_data_bytes:
            # Another data block belonging to the same logical group
            records.append(i)
            i += 1
        elif r.info.length == 4:
            # Single 4-byte separator: look ahead to see if more data follows
            if (
                i + 1 < len(inv.records)
                and inv.records[i + 1].info.length >= min_data_bytes
            ):
                i += 1  # skip separator, continue to next data block
            else:
                break  # no data after the separator → end of group
        else:
            break  # unexpected record size → end of group
    return records


def load_data_bytes(
    inv: OP2Inventory,
    header_index: int,
    min_data_bytes: int = 1000,
    *,
    first_idx: int = None,
):
    """
    Locate and concatenate all data records for the table that starts at
    *header_index*.

    Combines :func:`first_stress_record_after` with
    :func:`collect_data_records_after` to handle tables whose data is spread
    across multiple Fortran records.

    Parameters
    ----------
    inv : OP2Inventory
    header_index : int
    min_data_bytes : int
    first_idx : int, optional
        If supplied, skip the ``first_stress_record_after`` search and use
        this record index directly as the first data record.  Pass this when
        the correct data record is already known (e.g. from an EKEY scan).

    Returns
    -------
    payload : bytes
        Concatenated raw bytes from all data records.
    first_idx : int
        Index of the first data record.
    all_indices : list of int
        Indices of all data records collected.
    """
    if first_idx is None:
        first_idx = first_stress_record_after(inv, header_index)
    all_indices = collect_data_records_after(inv, first_idx, min_data_bytes)
    if len(all_indices) == 1:
        return inv.get_record_data(first_idx), first_idx, all_indices
    payload = b"".join(inv.get_record_data(i) for i in all_indices)
    return payload, first_idx, all_indices


def _record_looks_like_grid_force_data(
    rec, endian: str = "<", row_widths=(8, 7), min_rows: int = 10
) -> bool:
    """
    Heuristic for OQG1/OPG1-style records where forces may be all zero.
    Checks that:
      - record is large enough to hold at least min_rows rows
      - length is divisible by one of the row widths (in words)
      - the first-column words (packed IDs) follow the pattern: positive,
        divisible by 10, in a plausible grid-ID range, and monotonically
        non-decreasing for the first several rows
    """
    import struct

    L = rec.info.length
    for row_width in row_widths:
        row_bytes = row_width * 4
        if L < row_bytes * min_rows:
            continue
        if L % row_bytes != 0:
            continue
        n_rows = L // row_bytes
        # Sample up to 8 rows to check packed IDs
        sample = min(n_rows, 8)
        ids = []
        ok = True
        for r in range(sample):
            raw = struct.unpack(
                f"{endian}i", rec.data[r * row_bytes : r * row_bytes + 4]
            )[0]
            if raw <= 0:
                ok = False
                break
            grid = raw // 10
            comp = raw % 10
            if not (1 <= grid < 10_000_000) or comp not in range(10):
                ok = False
                break
            ids.append(raw)
        if ok and len(ids) >= 1:
            # IDs should be non-decreasing (SORT1 ordering)
            if all(ids[i] <= ids[i + 1] for i in range(len(ids) - 1)):
                return True
    return False


def first_grid_force_record_after(
    inv: OP2Inventory, header_index: int, max_ahead: int = 80
) -> int:
    """
    Starting after header_index, return index of the first record that
    looks like a grid-referenced force/load data block (OQG1, OPG1).
    Uses a looser heuristic than first_stress_record_after — force values
    may be near-zero or exactly zero, so we validate on packed grid IDs
    rather than float magnitudes.
    """
    start = header_index + 1
    end = min(len(inv.records), start + max_ahead)
    for i in range(start, end):
        rec = inv.records[i]
        if rec.info.length < 28:  # smaller than one 7-word row
            continue
        if _record_looks_like_grid_force_data(rec, inv.endian):
            return i
    raise ValueError(f"No grid-force-like record found after header {header_index}")


def classify_grid_force_headers(inv: OP2Inventory, token: str) -> List[tuple]:
    """
    Return ``[(header_idx, data_rec_idx, sc_offset), ...]`` for every data block
    found within each table named *token* (e.g. ``"OQG1"`` or ``"OPG1"``).

    Multiple subcases share a single 8-byte table-name record, so we scan the
    whole table interior for all grid-force data records and assign sc_offset
    0, 1, 2, … in discovery order.
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
            if _record_looks_like_grid_force_data(r, inv.endian):
                result.append((hdr_idx, r.info.index, sc_offset))
                sc_offset += 1
    return result


def peek_oes_block(inv: OP2Inventory, header_index: int) -> Dict[str, pd.DataFrame]:
    """
    Return a dict of DataFrames for the first stress-like record after an OES header:
       {
         "int32":   DataFrame view,
         "float32": DataFrame view,
         "float64": DataFrame view,
       }
    """
    data_rec = first_stress_record_after(inv, header_index)
    out = {}
    for kind in ("int32", "float32", "float64"):
        out[kind] = _view_record_as(inv, data_rec, kind=kind, cols=8, max_items=128)
    return out
