# op2_native/decoders/oesnlxr.py
"""
Decoder for nonlinear element stress results from the OESNLXR table.

Supported element types
-----------------------
etype  94  CBEAM   nonlinear  numwde=51
etype 226  CBUSH   nonlinear  numwde=19
etype  85  CTETRA  nonlinear  numwde=82

CBEAM NL binary layout  (51 words per element)
----------------------------------------------
  word  0          : packed_eid_device  (10*EID + device_code)
  Station A  (words  1..25):
    word  1        : GRID_A  (int, station A node ID)
    Fiber C (words 2..7)   : [fiber_coord, STRESS, EQ_STRESS, TOTAL_STRAIN,
                               EFF_STRAIN_PLAS, EFF_CREEP]
    Fiber D (words 8..13)  : same layout
    Fiber E (words 14..19) : same layout
    Fiber F (words 20..25) : same layout
  Station B  (words 26..50):
    word 26        : GRID_B  (int, station B node ID)
    Fibers C-F     : same layout as station A

Output columns (one row per fiber per element):
  EID, GRID, FIBER, STRESS, EQ_STRESS, TOTAL_STRAIN, EFF_STRAIN_PLAS, EFF_CREEP

CBUSH NL binary layout  (19 words per element)
-----------------------------------------------
  word  0 : packed_eid_device
  words 1-9  : FORCE_X, FORCE_Y, FORCE_Z, STRESS_TX, STRESS_TY, STRESS_TZ,
               STRAIN_TX, STRAIN_TY, STRAIN_TZ
  words 10-18: MOMENT_X, MOMENT_Y, MOMENT_Z, STRESS_RX, STRESS_RY, STRESS_RZ,
               STRAIN_RX, STRAIN_RY, STRAIN_RZ

Output columns (one row per element):
  EID, FORCE_X, FORCE_Y, FORCE_Z, STRESS_TX, STRESS_TY, STRESS_TZ,
  STRAIN_TX, STRAIN_TY, STRAIN_TZ, MOMENT_X, MOMENT_Y, MOMENT_Z,
  STRESS_RX, STRESS_RY, STRESS_RZ, STRAIN_RX, STRAIN_RY, STRAIN_RZ

CTETRA NL binary layout  (82 words per element)
------------------------------------------------
  word  0          : packed_eid_device
  word  1          : type flag  (integer, not in output)
  5 x 16-word node blocks  (centroid then 4 corner nodes):
    word  0        : GRID_ID  (int;  0 for centroid)
    words  1..6    : SX, SY, SZ, SXY, SYZ, SZX  (stresses)
    words  7..9    : EQ_STRESS, EFF_STRAIN_PLAS, EFF_CREEP
    words 10..15   : EX, EY, EZ, EXY, EYZ, EZX  (total strains)

Output columns (one row per node per element):
  EID, GRID, SX, SY, SZ, SXY, SYZ, SZX, EQ_STRESS, EFF_STRAIN_PLAS,
  EFF_CREEP, EX, EY, EZ, EXY, EYZ, EZX
"""
from __future__ import annotations

import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..models import OP2Inventory
from .oes_peek import load_data_bytes, first_data_record_after_ekey

# Element type codes
_CBEAM_NL_ETYPE = 94
_CBUSH_NL_ETYPE = 226
_CTETRA_NL_ETYPE = 85

# Row widths in words
_CBEAM_NL_NUMWDE = 51
_CBUSH_NL_NUMWDE = 19
_CTETRA_NL_NUMWDE = 82

# Output column names
_CBEAM_NL_COLS = [
    "EID",
    "GRID",
    "FIBER",
    "STRESS",
    "EQ_STRESS",
    "TOTAL_STRAIN",
    "EFF_STRAIN_PLAS",
    "EFF_CREEP",
]

_CBUSH_NL_COLS = [
    "EID",
    "FORCE_X",
    "FORCE_Y",
    "FORCE_Z",
    "STRESS_TX",
    "STRESS_TY",
    "STRESS_TZ",
    "STRAIN_TX",
    "STRAIN_TY",
    "STRAIN_TZ",
    "MOMENT_X",
    "MOMENT_Y",
    "MOMENT_Z",
    "STRESS_RX",
    "STRESS_RY",
    "STRESS_RZ",
    "STRAIN_RX",
    "STRAIN_RY",
    "STRAIN_RZ",
]

_CTETRA_NL_COLS = [
    "EID",
    "GRID",
    "SX",
    "SY",
    "SZ",
    "SXY",
    "SYZ",
    "SZX",
    "EQ_STRESS",
    "EFF_STRAIN_PLAS",
    "EFF_CREEP",
    "EX",
    "EY",
    "EZ",
    "EXY",
    "EYZ",
    "EZX",
]

# Fiber labels in output order
_FIBER_LABELS = ["C", "D", "E", "F"]


def _decode_cbeam_nl_payload(
    data: bytes,
    endian: str = "<",
    max_eid: int = 10_000_000,
) -> pd.DataFrame:
    """Decode CBEAM NL stress: 51 words per element, 8 rows per element."""
    stride = _CBEAM_NL_NUMWDE
    n_words = len(data) // 4
    if n_words < stride:
        return pd.DataFrame(columns=_CBEAM_NL_COLS)

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

        # Two stations; each station = 1 grid word + 4 fibers * 6 words
        for st_idx, st_offset in enumerate((1, 26)):
            grid = int(ints[base + st_offset])
            if grid <= 0:
                continue
            for fib_idx, fiber_label in enumerate(_FIBER_LABELS):
                # fiber block starts at st_offset + 1 + fib_idx*6
                fb = base + st_offset + 1 + fib_idx * 6
                # word[fb+0] = fiber_coord (position, not reported)
                stress = float(floats[fb + 1])
                eq_stress = float(floats[fb + 2])
                total_strain = float(floats[fb + 3])
                eff_plas = float(floats[fb + 4])
                eff_creep = float(floats[fb + 5])
                rows.append(
                    [
                        eid,
                        grid,
                        fiber_label,
                        stress,
                        eq_stress,
                        total_strain,
                        eff_plas,
                        eff_creep,
                    ]
                )

    return pd.DataFrame(rows, columns=_CBEAM_NL_COLS)


def _decode_cbush_nl_payload(
    data: bytes,
    endian: str = "<",
    max_eid: int = 10_000_000,
) -> pd.DataFrame:
    """Decode CBUSH NL: 19 words per element, 1 row per element."""
    stride = _CBUSH_NL_NUMWDE
    n_words = len(data) // 4
    if n_words < stride:
        return pd.DataFrame(columns=_CBUSH_NL_COLS)

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
        vals = floats[base + 1 : base + stride].tolist()
        rows.append([eid] + vals)

    return pd.DataFrame(rows, columns=_CBUSH_NL_COLS)


def _decode_ctetra_nl_payload(
    data: bytes,
    endian: str = "<",
    max_eid: int = 10_000_000,
) -> pd.DataFrame:
    """Decode CTETRA NL: 82 words per element, 5 rows per element.

    Layout:  word[0] packed EID,  word[1] type flag,  then 5 x 16-word blocks.
    Each block: grid_id + 6 stresses + 3 scalars + 6 strains = 16 words.
    """
    stride = _CTETRA_NL_NUMWDE
    n_words = len(data) // 4
    if n_words < stride:
        return pd.DataFrame(columns=_CTETRA_NL_COLS)

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

        # word[1] = type flag (skip), then 5 nodes starting at word[2]
        node_base = base + 2
        for node_idx in range(5):
            nb = node_base + node_idx * 16
            grid = int(ints[nb])
            sx = float(floats[nb + 1])
            sy = float(floats[nb + 2])
            sz = float(floats[nb + 3])
            sxy = float(floats[nb + 4])
            syz = float(floats[nb + 5])
            szx = float(floats[nb + 6])
            eq_s = float(floats[nb + 7])
            eff_plas = float(floats[nb + 8])
            eff_creep = float(floats[nb + 9])
            ex = float(floats[nb + 10])
            ey = float(floats[nb + 11])
            ez = float(floats[nb + 12])
            exy = float(floats[nb + 13])
            eyz = float(floats[nb + 14])
            ezx = float(floats[nb + 15])
            rows.append(
                [
                    eid,
                    grid,
                    sx,
                    sy,
                    sz,
                    sxy,
                    syz,
                    szx,
                    eq_s,
                    eff_plas,
                    eff_creep,
                    ex,
                    ey,
                    ez,
                    exy,
                    eyz,
                    ezx,
                ]
            )

    return pd.DataFrame(rows, columns=_CTETRA_NL_COLS)


# ---------------------------------------------------------------------------
# EKEY scanner
# ---------------------------------------------------------------------------


def _find_oesnlxr_ekeys(
    inv: OP2Inventory, header_idx: int
) -> List[Tuple[int, int, int, int]]:
    """
    Scan OESNLXR table interior and return one entry per element-type block.

    Returns
    -------
    list of (ekey_idx, first_data_idx, etype, numwde)
    """
    from .oes_search import _find_ekeys_in_table

    return _find_ekeys_in_table(inv, header_idx)


# ---------------------------------------------------------------------------
# Header classifier
# ---------------------------------------------------------------------------


def classify_oesnlxr_headers(
    inv: OP2Inventory,
) -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    Classify every OESNLXR element-type sub-block by element type.

    Returns
    -------
    cbeam_blocks, cbush_blocks, ctetra_blocks
        Each is a list of ``(header_idx, ekey_idx, sc_offset)`` 3-tuples.
        ``sc_offset`` is 0 for the first subcase block, 1 for the second, etc.
    """
    from .oes_search import _find_token

    token_hits = _find_token(inv, "OESNLXR")
    cbeam: List[Tuple] = []
    cbush: List[Tuple] = []
    ctetra: List[Tuple] = []
    for hdr in token_hits:
        etype_count: Dict[int, int] = {}
        for ekey_idx, _first_data, etype, _numwde in _find_oesnlxr_ekeys(inv, hdr):
            sc_offset = etype_count.get(etype, 0)
            etype_count[etype] = sc_offset + 1
            entry = (hdr, ekey_idx, sc_offset)
            if etype == _CBEAM_NL_ETYPE:
                cbeam.append(entry)
            elif etype == _CBUSH_NL_ETYPE:
                cbush.append(entry)
            elif etype == _CTETRA_NL_ETYPE:
                ctetra.append(entry)
    return cbeam, cbush, ctetra


# ---------------------------------------------------------------------------
# Public decode entry points
# ---------------------------------------------------------------------------


def decode_oesnlxr_cbeam(
    inv: OP2Inventory, header_index: int, ekey_index: int = None
) -> pd.DataFrame:
    """Decode CBEAM NL stress block from OESNLXR."""
    if ekey_index is not None:
        first_idx = first_data_record_after_ekey(inv, ekey_index)
        payload, data_idx, all_recs = load_data_bytes(
            inv, ekey_index, first_idx=first_idx
        )
    else:
        payload, data_idx, all_recs = load_data_bytes(inv, header_index)

    df = _decode_cbeam_nl_payload(payload, inv.endian)
    df.attrs["header_record"] = header_index
    df.attrs["data_record"] = data_idx
    df.attrs["all_data_records"] = all_recs
    df.attrs["element_type"] = _CBEAM_NL_ETYPE
    return df


def decode_oesnlxr_cbush(
    inv: OP2Inventory, header_index: int, ekey_index: int = None
) -> pd.DataFrame:
    """Decode CBUSH NL stress block from OESNLXR."""
    if ekey_index is not None:
        first_idx = first_data_record_after_ekey(inv, ekey_index)
        payload, data_idx, all_recs = load_data_bytes(
            inv, ekey_index, first_idx=first_idx
        )
    else:
        payload, data_idx, all_recs = load_data_bytes(inv, header_index)

    df = _decode_cbush_nl_payload(payload, inv.endian)
    df.attrs["header_record"] = header_index
    df.attrs["data_record"] = data_idx
    df.attrs["all_data_records"] = all_recs
    df.attrs["element_type"] = _CBUSH_NL_ETYPE
    return df


def decode_oesnlxr_ctetra(
    inv: OP2Inventory, header_index: int, ekey_index: int = None
) -> pd.DataFrame:
    """Decode CTETRA NL stress block from OESNLXR."""
    if ekey_index is not None:
        first_idx = first_data_record_after_ekey(inv, ekey_index)
        payload, data_idx, all_recs = load_data_bytes(
            inv, ekey_index, first_idx=first_idx
        )
    else:
        payload, data_idx, all_recs = load_data_bytes(inv, header_index)

    df = _decode_ctetra_nl_payload(payload, inv.endian)
    df.attrs["header_record"] = header_index
    df.attrs["data_record"] = data_idx
    df.attrs["all_data_records"] = all_recs
    df.attrs["element_type"] = _CTETRA_NL_ETYPE
    return df
