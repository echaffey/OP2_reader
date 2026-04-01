# op2_native/decoders/lama.py
"""
Decoder for LAMA (real eigenvalue) tables.

Nastran OP2 LAMA layout  (real modes, SORT1)
--------------------------------------------
The LAMA table contains one record per subcase with eigenvalue summary data.
Each row corresponds to one mode and occupies 7 words:

    Word 0 : MODE    int32   mode number (1-based)
    Word 1 : ORDER   int32   extraction order
    Word 2 : EIGENVALUE  float64  (words 2-3, 8 bytes) real eigenvalue (rad²/s²)
    Word 4 : RADIANS     float32  natural frequency in rad/s (sqrt of eigenvalue)
    Word 5 : CYCLES      float32  natural frequency in Hz (cycles/s)
    Word 6 : GENM        float32  generalised mass
    Word 7 : GENSTIF     float32  generalised stiffness

Note: the float64 eigenvalue spans words 2-3 in some versions; in others it
is two float32 words.  We handle both by trying float64 first, validating
against RADIANS/CYCLES, then falling back to float32.

Output columns
--------------
    MODE       int     mode number
    ORDER      int     extraction order
    EIGENVALUE float   eigenvalue (ω², rad²/s²)
    RADIANS    float   angular frequency (ω, rad/s)
    CYCLES     float   frequency (f, Hz)
    GENM       float   generalised mass
    GENSTIF    float   generalised stiffness
"""
from __future__ import annotations

import struct
from typing import Dict, List

import numpy as np
import pandas as pd

from ..models import OP2Inventory

_LAMA_COLS = ["MODE", "ORDER", "EIGENVALUE", "RADIANS", "CYCLES", "GENM", "GENSTIF"]

# Two possible row widths depending on whether EIGENVALUE is stored as
# float64 (8 bytes = 2 words) or two float32 values (2 words).
_STRIDE_F64 = 8  # MODE(i4) + ORDER(i4) + EIGENVALUE(f8) + RADIANS(f4) + CYCLES(f4) + GENM(f4) + GENSTIF(f4)
_STRIDE_F32 = 7  # same but EIGENVALUE as two f4 words


def _find_lama_headers(inv: OP2Inventory) -> List[int]:
    """Return record indices of 8-byte LAMA table-name records."""
    return [
        r.info.index for r in inv.records if r.info.length == 8 and b"LAMA" in r.data
    ]


def _decode_lama_record(data: bytes, endian: str = "<") -> pd.DataFrame:
    """
    Decode a single LAMA data record into a DataFrame.

    Tries float64 eigenvalue layout first; falls back to float32 if the
    decoded RADIANS/CYCLES values are inconsistent.
    """
    bo = ">" if endian == ">" else "<"
    n_bytes = len(data)

    rows: List[list] = []

    # Try stride=8 (float64 eigenvalue, 32 bytes per row)
    row_bytes_f64 = _STRIDE_F64 * 4
    if n_bytes % row_bytes_f64 == 0 and n_bytes >= row_bytes_f64:
        n_rows = n_bytes // row_bytes_f64
        valid = True
        tmp_rows: List[list] = []
        for i in range(n_rows):
            off = i * row_bytes_f64
            mode = struct.unpack_from(f"{bo}i", data, off)[0]
            order = struct.unpack_from(f"{bo}i", data, off + 4)[0]
            eig = struct.unpack_from(f"{bo}d", data, off + 8)[0]  # float64
            rad = struct.unpack_from(f"{bo}f", data, off + 16)[0]
            cyc = struct.unpack_from(f"{bo}f", data, off + 20)[0]
            genm = struct.unpack_from(f"{bo}f", data, off + 24)[0]
            gens = struct.unpack_from(f"{bo}f", data, off + 28)[0]
            # Sanity: RADIANS should be ≈ sqrt(|eigenvalue|)
            if eig > 0 and rad > 0:
                expected_rad = eig**0.5
                if abs(rad - expected_rad) / max(abs(rad), 1e-10) > 0.01:
                    valid = False
                    break
            tmp_rows.append([mode, order, eig, rad, cyc, genm, gens])
        if valid and tmp_rows:
            rows = tmp_rows

    # Fallback: stride=7 (two float32 words for eigenvalue, 28 bytes per row)
    if not rows:
        row_bytes_f32 = _STRIDE_F32 * 4
        if n_bytes % row_bytes_f32 == 0 and n_bytes >= row_bytes_f32:
            n_rows = n_bytes // row_bytes_f32
            for i in range(n_rows):
                off = i * row_bytes_f32
                mode = struct.unpack_from(f"{bo}i", data, off)[0]
                order = struct.unpack_from(f"{bo}i", data, off + 4)[0]
                # eigenvalue stored as two consecutive float32 — take as float64 via reinterpret
                eig_bytes = data[off + 8 : off + 16]
                try:
                    eig = struct.unpack_from(f"{bo}d", eig_bytes)[0]
                except struct.error:
                    eig = float("nan")
                rad = struct.unpack_from(f"{bo}f", data, off + 16)[0]
                cyc = struct.unpack_from(f"{bo}f", data, off + 20)[0]
                genm = struct.unpack_from(f"{bo}f", data, off + 24)[0]
                rows.append([mode, order, eig, rad, cyc, genm, float("nan")])

    if not rows:
        return pd.DataFrame(columns=_LAMA_COLS)

    df = pd.DataFrame(rows, columns=_LAMA_COLS)
    df["MODE"] = df["MODE"].astype(int)
    df["ORDER"] = df["ORDER"].astype(int)
    return df


def _subcase_for_lama_header(inv: OP2Inventory, header_idx: int) -> int:
    """
    Read the subcase ID from the IDENT record following a LAMA header.
    Returns 1 if not determinable.
    """
    for i in range(header_idx + 1, min(len(inv.records), header_idx + 10)):
        rec = inv.records[i]
        if rec.info.length == 28:
            try:
                words = struct.unpack("<7i", rec.data)
                sc = words[6]
                if sc > 0:
                    return sc
            except struct.error:
                pass
    return 1


def _first_data_record_after(inv: OP2Inventory, header_idx: int) -> int:
    """Return index of the first non-tiny record after header_idx."""
    for i in range(header_idx + 1, min(len(inv.records), header_idx + 30)):
        rec = inv.records[i]
        if rec.info.length >= 28:
            return i
    return header_idx + 1


def decode_lama(inv: OP2Inventory) -> Dict[int, pd.DataFrame]:
    """
    Decode all LAMA (real eigenvalue) tables in the inventory.

    Returns
    -------
    dict
        ``{subcase_id: DataFrame}`` with columns
        ``MODE, ORDER, EIGENVALUE, RADIANS, CYCLES, GENM, GENSTIF``.
        Returns an empty dict if no LAMA tables are found.
    """
    headers = _find_lama_headers(inv)
    if not headers:
        return {}

    result: Dict[int, pd.DataFrame] = {}
    for hdr in headers:
        sc = _subcase_for_lama_header(inv, hdr)
        data_idx = _first_data_record_after(inv, hdr)
        rec = inv.records[data_idx]
        df = _decode_lama_record(rec.data, inv.endian)
        if df.empty:
            continue
        df.attrs["header_record"] = hdr
        df.attrs["data_record"] = data_idx
        if sc in result:
            result[sc] = pd.concat([result[sc], df], ignore_index=True)
        else:
            result[sc] = df

    return result
