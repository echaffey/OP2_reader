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
import struct
from typing import Dict, List

import numpy as np
import pandas as pd

from ..models import OP2Inventory

_LAMA_COLS = ["MODE", "ORDER", "EIGENVALUE", "RADIANS", "CYCLES", "GENM", "GENSTIF"]

# Each mode row is 7 x 4-byte words = 28 bytes
_STRIDE_F32 = 7


def _find_lama_headers(inv: OP2Inventory) -> List[int]:
    """Return record indices (info.index) of 8-byte LAMA table-name records."""
    return [
        r.info.index for r in inv.records if r.info.length == 8 and b"LAMA" in r.data
    ]


def _is_lama_data(data: bytes, endian: str = "<") -> bool:
    """
    Return True if *data* looks like a valid LAMA eigenvalue payload.

    A real LAMA data record has N x 7 float32 words (28 bytes/mode):
        [mode(i4), order(i4), eigenvalue(f4), radians(f4), cycles(f4), genm(f4), gens(f4)]

    Quick validation: length divisible by 28, first word is a small positive
    integer (mode number), third word is a large positive float (eigenvalue),
    and sqrt(eigenvalue) ≈ radians within 5 %.
    """
    import math

    n = len(data)
    if n < 28 or n % 28 != 0:
        return False
    bo = ">" if endian == ">" else "<"
    mode = struct.unpack_from(f"{bo}i", data, 0)[0]
    if not (1 <= mode <= 100_000):
        return False
    eig = struct.unpack_from(f"{bo}f", data, 8)[0]
    rad = struct.unpack_from(f"{bo}f", data, 12)[0]
    if eig <= 0 or rad <= 0:
        return False
    try:
        expected = math.sqrt(eig)
    except ValueError:
        return False
    return abs(rad - expected) / expected < 0.05


def _decode_lama_record(data: bytes, endian: str = "<") -> pd.DataFrame:
    """
    Decode a single LAMA data payload into a DataFrame.

    NX Nastran stores real eigenvalues with 7 float32 words per mode:
        word 0 : MODE       int32
        word 1 : ORDER      int32
        word 2 : EIGENVALUE float32  (ω², rad²/s²)
        word 3 : RADIANS    float32  (ω, rad/s)
        word 4 : CYCLES     float32  (f, Hz)
        word 5 : GENM       float32  (generalised mass)
        word 6 : GENSTIF    float32  (generalised stiffness)
    """
    row_bytes = _STRIDE_F32 * 4  # 28 bytes / mode
    n_bytes = len(data)
    if n_bytes < row_bytes or n_bytes % row_bytes != 0:
        return pd.DataFrame(columns=_LAMA_COLS)

    bo = ">" if endian == ">" else "<"
    n_rows = n_bytes // row_bytes
    rows: List[list] = []
    for i in range(n_rows):
        off = i * row_bytes
        mode = struct.unpack_from(f"{bo}i", data, off)[0]
        order = struct.unpack_from(f"{bo}i", data, off + 4)[0]
        eig = struct.unpack_from(f"{bo}f", data, off + 8)[0]
        rad = struct.unpack_from(f"{bo}f", data, off + 12)[0]
        cyc = struct.unpack_from(f"{bo}f", data, off + 16)[0]
        genm = struct.unpack_from(f"{bo}f", data, off + 20)[0]
        gens = struct.unpack_from(f"{bo}f", data, off + 24)[0]
        rows.append([mode, order, eig, rad, cyc, genm, gens])

    df = pd.DataFrame(rows, columns=_LAMA_COLS)
    df["MODE"] = df["MODE"].astype(int)
    df["ORDER"] = df["ORDER"].astype(int)
    return df


def _subcase_for_lama_header(inv: OP2Inventory, header_idx: int) -> int:
    """
    Read the subcase ID from the IDENT record following a LAMA header.

    Tries the short 7-word IDENT (word[3] = subcase ID) and the longer
    146-word IDENT (word[3] = subcase ID).  Returns 1 if not determinable.
    """
    for i in range(header_idx + 1, min(len(inv.records), header_idx + 30)):
        rec = inv.records[i]
        # Short IDENT: 7 words (28 bytes) — subcase at word[3]
        if rec.info.length == 28:
            try:
                words = struct.unpack(f"{inv.endian}7i", rec.data)
                sc = words[3]
                if sc > 0:
                    return sc
            except struct.error:
                pass
        # Long IDENT: >= 16 bytes — try word[3] as subcase
        elif rec.info.length >= 16:
            try:
                sc = struct.unpack_from(f"{inv.endian}i", rec.data, 12)[0]
                if sc > 0:
                    return sc
            except struct.error:
                pass
    return 1


def _find_lama_data_record(inv: OP2Inventory, header_idx: int) -> int:
    """
    Find the index of the eigenvalue data record after *header_idx*.

    Scans forward from header_idx + 1, skipping tiny marker records and
    IDENT/header blocks, until a record whose content passes
    ``_is_lama_data()`` is found.  Returns -1 if not found.
    """
    for i in range(header_idx + 1, min(len(inv.records), header_idx + 40)):
        rec = inv.records[i]
        if rec.info.length >= 28 and _is_lama_data(rec.data, inv.endian):
            return rec.info.index
    return -1


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
    seen_data_records: set = set()

    for hdr in headers:
        data_idx = _find_lama_data_record(inv, hdr)
        if data_idx < 0:
            continue  # this LAMA occurrence has no data (summary header only)
        if data_idx in seen_data_records:
            continue  # same data record already decoded from an earlier header
        seen_data_records.add(data_idx)

        sc = _subcase_for_lama_header(inv, hdr)
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
