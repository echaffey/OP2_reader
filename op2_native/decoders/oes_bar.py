# op2_native/decoders/oes_bar.py
"""
Decoder for bar/beam element (CBAR/CBEAM) stress/strain blocks
from OES1X1 tables.

Nastran OP2 real SORT1 CBAR stress layout  (num_wide = 16)
----------------------------------------------------------
Each element occupies exactly 16 words:

    Word  0 : packed_eid_device  (10*EID + device_code)
    Word  1 : BEND1A  float  bending stress, end A, pt 1
    Word  2 : BEND2A  float  bending stress, end A, pt 2
    Word  3 : BEND3A  float  bending stress, end A, pt 3
    Word  4 : BEND4A  float  bending stress, end A, pt 4
    Word  5 : AXIAL   float  axial stress
    Word  6 : SMAX_A  float  max stress at end A
    Word  7 : SMIN_A  float  min stress at end A
    Word  8 : MS_A    float  margin of safety, tension, end A
    Word  9 : BEND1B  float  bending stress, end B, pt 1
    Word 10 : BEND2B  float  bending stress, end B, pt 2
    Word 11 : BEND3B  float  bending stress, end B, pt 3
    Word 12 : BEND4B  float  bending stress, end B, pt 4
    Word 13 : SMAX_B  float  max stress at end B
    Word 14 : SMIN_B  float  min stress at end B
    Word 15 : MS_C  float  margin of safety, compression, end B

Nastran OP2 real SORT1 CBEAM stress layout  (num_wide = 111)
------------------------------------------------------------
Each element occupies 111 words:
  Word  0       : packed_eid_device  (10*EID + device_code)
  Words 1-110   : 11 stations x 10 words per station

Each 10-word station row:
    [GRID, SD, SXC, SXD, SXE, SXF, SMAX, SMIN, MS_T, MS_C]

Output columns (CBAR)
---------------------
  EID, BEND1A, BEND2A, BEND3A, BEND4A, AXIAL,
  SMAX_A, SMIN_A, MS_A, BEND1B, BEND2B, BEND3B, BEND4B, SMAX_B, SMIN_B, MS_C

Output columns (CBEAM -- one row per station per element)
---------------------------------------------------------
  EID, GRID, SD, SXC, SXD, SXE, SXF, SMAX, SMIN, MS_T, MS_C
"""
import struct
from typing import List, Optional

import numpy as np
import pandas as pd

from ..models import OP2Inventory
from .oes_peek import load_data_bytes, first_data_record_after_ekey

# Element type codes
_CBEAM_ETYPE = 2
_CBAR_ETYPE = 34

# Row widths (words)
_CBAR_NUM_WIDE = 16  # 1 packed_eid + 15 stress values
_CBEAM_NUM_WIDE = 111  # 1 packed_eid + 11 stations x 10 words

_CBAR_COLS = [
    "EID",
    "BEND1A",
    "BEND2A",
    "BEND3A",
    "BEND4A",
    "AXIAL",
    "SMAX_A",
    "SMIN_A",
    "MS_A",
    "BEND1B",
    "BEND2B",
    "BEND3B",
    "BEND4B",
    "SMAX_B",
    "SMIN_B",
    "MS_C",
]

# CBEAM: one output row per station (up to 11 per element)
_CBEAM_COLS = [
    "EID",
    "GRID",
    "SD",
    "SXC",
    "SXD",
    "SXE",
    "SXF",
    "SMAX",
    "SMIN",
    "MS_T",
    "MS_C",
]


def _elem_type_from_ekey(inv: OP2Inventory, start_index: int) -> Optional[int]:
    """Return element type from a 584-byte EKEY record.

    If ``start_index`` is itself a 584-byte EKEY record it is read directly;
    otherwise the records following ``start_index`` are scanned.
    """
    from .oes_search import _etype_from_ekey_words

    rec = inv.records[start_index]
    if rec.info.length == 584:
        words = struct.unpack(f"{inv.endian}146i", rec.data)
        return _etype_from_ekey_words(words)
    for i in range(start_index + 1, min(len(inv.records), start_index + 30)):
        rec = inv.records[i]
        if rec.info.length == 584:
            words = struct.unpack(f"{inv.endian}146i", rec.data)
            return _etype_from_ekey_words(words)
    return None


def _decode_cbar_payload(
    data: bytes,
    endian: str = "<",
    max_eid: int = 99_999_999,
) -> pd.DataFrame:
    """Decode CBAR stress: 16 words per element."""
    n_words = len(data) // 4
    stride = _CBAR_NUM_WIDE
    if n_words < stride:
        return pd.DataFrame(columns=_CBAR_COLS)

    bo = "<" if endian == "<" else ">"
    ints = np.frombuffer(data[: n_words * 4], dtype=f"{bo}i4")
    floats = np.frombuffer(data[: n_words * 4], dtype=f"{bo}f4")

    rows: List[list] = []
    n_elems = n_words // stride
    for elem in range(n_elems):
        base = elem * stride
        raw = int(ints[base])
        if raw <= 0:
            break
        eid = raw // 10
        loc = raw % 10
        if not (1 <= loc <= 9 and 1 <= eid <= max_eid):
            break
        vals = floats[base + 1 : base + stride]
        if not np.all(np.isfinite(vals)):
            break
        rows.append([eid] + vals.tolist())

    return pd.DataFrame(rows, columns=_CBAR_COLS)


def _decode_cbeam_payload(
    data: bytes,
    endian: str = "<",
    max_eid: int = 99_999_999,
) -> pd.DataFrame:
    """
    Decode CBEAM stress: 111 words per element.
      Word 0        : packed_eid_device
      Words 1-110   : 11 stations x 10 words each
                      [grid, sd, sxc, sxd, sxe, sxf, smax, smin, ms_t, ms_c]
    Stations with grid==0 AND sd==0.0 are padding rows and are skipped.
    """
    n_words = len(data) // 4
    stride = _CBEAM_NUM_WIDE
    n_per_station = 10
    n_stations = 11

    if n_words < stride:
        return pd.DataFrame(columns=_CBEAM_COLS)

    bo = "<" if endian == "<" else ">"
    ints = np.frombuffer(data[: n_words * 4], dtype=f"{bo}i4")
    floats = np.frombuffer(data[: n_words * 4], dtype=f"{bo}f4")

    rows: List[list] = []
    n_elems = n_words // stride
    for elem in range(n_elems):
        base = elem * stride
        raw = int(ints[base])
        if raw <= 0:
            break
        eid = raw // 10
        loc = raw % 10
        if not (1 <= loc <= 9 and 1 <= eid <= max_eid):
            break

        for st in range(n_stations):
            st_base = base + 1 + st * n_per_station
            grid = int(ints[st_base])
            sd = float(floats[st_base + 1])
            # Skip blank padding stations (grid==0 and sd==0.0)
            if grid == 0 and sd == 0.0:
                continue
            rest = floats[st_base + 2 : st_base + n_per_station]
            if not np.all(np.isfinite(rest)):
                continue
            rows.append([eid, grid, sd] + rest.tolist())

    return pd.DataFrame(rows, columns=_CBEAM_COLS)


def decode_oes_bar(
    inv: OP2Inventory, header_index: int, ekey_index: int = None
) -> pd.DataFrame:
    """
    Decode a bar/beam element stress block from OES1X1.

    Returns a DataFrame whose columns depend on element type:

    CBAR (element type 34):
      EID, S1A, S2A, S3A, S4A, AXIAL, SMAXA, SMINA, MS_A,
      S1B, S2B, S3B, S4B, SMAXB, SMINB, MS_C

    CBEAM (element type 2), one row per station:
      EID, GRID, SD, SXC, SXD, SXE, SXF, SMAX, SMIN, MS_T, MS_C
    """
    target = ekey_index if ekey_index is not None else header_index
    etype = _elem_type_from_ekey(inv, target)

    # When an EKEY index is provided, use the direct record lookup to avoid
    # the stress-data content heuristic failing on zero-padded blocks (e.g.
    # CBEAM with a 111-word / 11-station layout where only 1–2 stations carry
    # data and the rest are zeros).
    if ekey_index is not None:
        first_idx = first_data_record_after_ekey(inv, ekey_index)
        payload, data_idx, all_recs = load_data_bytes(
            inv, ekey_index, first_idx=first_idx
        )
    else:
        payload, data_idx, all_recs = load_data_bytes(inv, header_index)

    if etype == _CBEAM_ETYPE:
        df = _decode_cbeam_payload(payload, inv.endian)
    else:
        # CBAR or any other bar-family type
        df = _decode_cbar_payload(payload, inv.endian)

    df.attrs["header_record"] = header_index
    df.attrs["data_record"] = data_idx
    df.attrs["all_data_records"] = all_recs
    df.attrs["element_type"] = etype
    return df
