# op2_native/decoders/oef1.py
"""
Decoder for OEF1 / OEF1X element force data blocks.

Layout (SORT1, real, static or modal):
  Each record is a flat array of 4-byte words.
  Every row has the form:

      [packed_id, grid_or_cen, n_corners, F1..F8]  (11 words for CQUAD4/CTRIA3)
  or
      [packed_id, F1..F8]                          (9 words for CBAR etc.)

  where packed_id = 10 * EID + device_code.

Element-type column names (per Nastran QRG)
-------------------------------------------
CQUAD4 / CTRIA3 (etype 33, 73, 74, 144):   11 words/elem
  NX  = membrane force x (force/length)
  NY  = membrane force y (force/length)
  NXY = membrane shear force (force/length)
  MX  = bending moment x (force·length/length)
  MY  = bending moment y (force·length/length)
  MXY = twisting moment (force·length/length)
  QX  = transverse shear force x (force/length)
  QY  = transverse shear force y (force/length)

CBAR (etype 34):                            9 words/elem
  BM1A, BM2A, BM1B, BM2B, TS1, TS2, AF, TRQ

CBEAM (etype 2):                            9 words/elem (centroid only in OEF1)
  BM1A, BM2A, BM1B, BM2B, TS1, TS2, AF, TRQ

Generic fallback:                           9 words/elem
  F1..F8
"""
import struct
from typing import Dict, List, Optional

import pandas as pd

from ..models import OP2Inventory
from .oes_peek import load_data_bytes, first_data_record_after_ekey

# ---------------------------------------------------------------------------
# Column name tables per element type
# ---------------------------------------------------------------------------
_CQUAD4_FORCE_COLS = ["EID", "NX", "NY", "NXY", "MX", "MY", "MXY", "QX", "QY"]
_CBAR_FORCE_COLS = ["EID", "BM1A", "BM2A", "BM1B", "BM2B", "TS1", "TS2", "AF", "TRQ"]
_CBUSH_FORCE_COLS = ["EID", "FX", "FY", "FZ", "MX", "MY", "MZ"]
_CGAP_FORCE_COLS = [
    "EID",
    "COMP_X",
    "SHEAR_Y",
    "SHEAR_Z",
    "AXIAL_U",
    "TOTAL_V",
    "TOTAL_W",
    "SLIP_V",
    "SLIP_W",
]
_GENERIC_FORCE_COLS = ["EID", "LOC", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]

# CBEAM per-station force output (OEF1 NUMWDE=100, one row per active station)
_CBEAM_FORCE_COLS = ["EID", "GRID", "SD", "BM1", "BM2", "WS1", "WS2", "AF", "TRQ"]
_CBEAM_OEF_NUM_WIDE = 100  # 1 packed_eid + 11 stations × 9 words/station
_CBEAM_OEF_STATIONS = 11
_CBEAM_OEF_WORDS_PER_STATION = 9

# CQUAD4 corner-force columns (EID=element, GRID=0 for centroid, >0 for corner)
# Force layout per row: NX, NY, NXY, MX, MY, MXY, QX, QY  (8 floats)
_CQUAD4_FORCE_CORNER_COLS = [
    "EID",
    "GRID",
    "NX",
    "NY",
    "NXY",
    "MX",
    "MY",
    "MXY",
    "QX",
    "QY",
]

# CQUAD4 corner layout constants (NUMWDE=47):
#   w0        : packed_eid  (10*EID + device_code)
#   w1        : 'CEN/' ASCII marker
#   w2        : n_corners (=4)
#   w3..w10   : 8 centroid force floats
#   w11       : corner-1 grid id (plain integer, NOT packed)
#   w12..w19  : 8 corner-1 force floats
#   w20..w28  : packed corner-2 id + 8 floats
#   w29..w37  : packed corner-3 id + 8 floats
#   w38..w46  : packed corner-4 id + 8 floats
#   Total: 3 + 8 + 4*9 = 47  ✓
_CQUAD4_CORNER_NUMWDE = 47
_CQUAD4_CORNER_CEN_OFFSET = 3  # centroid forces start at word 3
_CQUAD4_CORNER_CEN_WORDS = 8  # 8 force floats per layer (NX..QY)
_CQUAD4_CORNER_STRIDE = 9  # 1 grid id + 8 forces per corner
_CQUAD4_N_CORNERS = 4
CEN_MARKER_UINT = 793658691  # b'CEN/' as little-endian uint32

# Shell element types
_SHELL_ETYPES = {33, 73, 74, 144, 64, 75, 82, 70}
# Bar/beam element types
_BAR_ETYPES = {2, 34, 100}
# Bush/spring element types
_BUSH_ETYPES = {102}  # CBUSH
# Gap element types
_GAP_ETYPES = {38}  # CGAP


def _force_cols_for_etype(etype: Optional[int]) -> List[str]:
    """Return the appropriate column list for a given element type."""
    if etype in _SHELL_ETYPES:
        return _CQUAD4_FORCE_COLS
    if etype in _BAR_ETYPES:
        return _CBAR_FORCE_COLS
    if etype in _BUSH_ETYPES:
        return _CBUSH_FORCE_COLS
    if etype in _GAP_ETYPES:
        return _CGAP_FORCE_COLS
    return _GENERIC_FORCE_COLS


def _etype_for_oef_header(inv: OP2Inventory, start_index: int) -> Optional[int]:
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


def classify_oef_headers(inv: OP2Inventory):
    """
    Classify every OEF1* element-type sub-block by element category.

    Returns
    -------
    shell_blocks, bar_blocks, bush_blocks, gap_blocks, other_blocks
        Each is a list of ``(header_idx, ekey_idx, sc_offset)`` 3-tuples.
    """
    from .oes_search import find_oef_tables, _find_ekeys_in_table

    tables = find_oef_tables(inv)
    all_hdrs = sorted(idx for hits in tables.values() for idx in hits)
    shells, bars, bushes, gaps, others = [], [], [], [], []
    for hdr in all_hdrs:
        etype_count: dict = {}
        for ekey_idx, _first_data, etype, _numwde in _find_ekeys_in_table(inv, hdr):
            sc_offset = etype_count.get(etype, 0)
            etype_count[etype] = sc_offset + 1
            entry = (hdr, ekey_idx, sc_offset)
            if etype in _SHELL_ETYPES:
                shells.append(entry)
            elif etype in _BAR_ETYPES:
                bars.append(entry)
            elif etype in _BUSH_ETYPES:
                bushes.append(entry)
            elif etype in _GAP_ETYPES:
                gaps.append(entry)
            else:
                others.append(entry)
    return shells, bars, bushes, gaps, others


# ---------------------------------------------------------------------------
# Payload decoders
# ---------------------------------------------------------------------------


def _decode_oef1_shell_payload(
    payload: bytes,
    endian: str = "<",
    float_thr: float = 1e-6,
    max_eid: int = 1_000_000,
) -> pd.DataFrame:
    """
    Decode CQUAD4/CTRIA3 element forces: 11 words per element.

    Word layout per centroid row:
      [packed_eid, CEN/(4bytes), n_corners, NX, NY, NXY, MX, MY, MXY, QX, QY]
    Only the centroid row is returned (identified by the 3-word near-zero
    float marker that precedes each element in this OP2 variant).
    """
    import numpy as np

    n_words = len(payload) // 4
    cols = _CQUAD4_FORCE_COLS
    if n_words < 11:
        return pd.DataFrame(columns=cols)

    bo = "<" if endian == "<" else ">"
    floats = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}f4")
    words_i = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}i4")
    words_u = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}u4")

    # Vectorised marker detection
    near_zero = np.abs(floats) < float_thr
    matches = np.where(near_zero[:-2] & near_zero[1:-1] & near_zero[2:])[0].tolist()

    rows: List[list] = []
    seen: set = set()

    if matches:
        for m in matches:
            found = None
            for j in range(3):
                idx = m + j
                if idx >= n_words:
                    break
                for val in (int(words_u[idx]), int(words_i[idx])):
                    if val >= 10:
                        loc = val % 10
                        eid = val // 10
                        if 1 <= loc <= 9 and 1 <= eid <= max_eid:
                            found = eid
                            break
                if found:
                    break
            if not found or found in seen:
                continue
            start_f = m + 3
            if start_f + 8 <= n_words:
                forces = floats[start_f : start_f + 8]
                if not np.all(np.isfinite(forces)):
                    continue
                rows.append([found] + forces.tolist())
                seen.add(found)

    # Conventional fixed-stride fallback (stride=11)
    if not rows:
        stride = 11
        for offset in range(0, n_words - stride + 1, stride):
            raw_id = int(words_i[offset])
            if raw_id >= 10:
                loc = raw_id % 10
                eid = raw_id // 10
                if 1 <= loc <= 9 and 1 <= eid <= max_eid:
                    forces = floats[offset + 3 : offset + 11]
                    if np.all(np.isfinite(forces)):
                        rows.append([eid] + forces.tolist())

    return pd.DataFrame(rows, columns=cols)


def _decode_oef1_ctria3_payload(
    payload: bytes,
    endian: str = "<",
    max_eid: int = 1_000_000,
) -> pd.DataFrame:
    """
    Decode CTRIA3 element forces (NUMWDE=9, centroid only, EID-first layout).

    Word layout per element (9 words):
        w0:     packed_eid  (10*EID + device_code)
        w1..w8: NX, NY, NXY, MX, MY, MXY, QX, QY
    """
    import numpy as np

    n_words = len(payload) // 4
    cols = _CQUAD4_FORCE_COLS  # same 9 columns: EID, NX..QY
    if n_words < 9:
        return pd.DataFrame(columns=cols)

    bo = "<" if endian == "<" else ">"
    floats = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}f4")
    words_i = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}i4")

    stride = 9
    rows = []
    for offset in range(0, n_words - stride + 1, stride):
        raw_id = int(words_i[offset])
        if raw_id < 10:
            continue
        eid = raw_id // 10
        if not (1 <= eid <= max_eid):
            continue
        forces = floats[offset + 1 : offset + 9]
        if not np.all(np.isfinite(forces)):
            continue
        rows.append([eid] + forces.tolist())

    return pd.DataFrame(rows, columns=cols)


def _decode_oef1_shell_corner_payload(
    payload: bytes,
    endian: str = "<",
    max_eid: int = 1_000_000,
    max_grid: int = 10_000_000,
) -> pd.DataFrame:
    """
    Decode CQUAD4 element forces with corner output (NUMWDE=47).

    Layout per element (47 words):
      w0        : packed_eid  (10*EID + device_code)
      w1        : 'CEN/' ASCII marker  (0x434E452F)
      w2        : n_corners  (= 4)
      w3..w10   : centroid forces (8 floats: NX, NY, NXY, MX, MY, MXY, QX, QY)
      w11       : corner-1 grid id (plain integer, NOT packed)
      w12..w19  : corner-1 forces
      w20..w46  : (corner-2, corner-3, corner-4) same pattern

    Returns a DataFrame with one row per (EID, location) pair.
    GRID=0 for the centroid row; GRID=actual grid ID for corner rows.
    """
    import numpy as np

    n_words = len(payload) // 4
    cols = _CQUAD4_FORCE_CORNER_COLS
    stride = _CQUAD4_CORNER_NUMWDE

    if n_words < stride:
        return pd.DataFrame(columns=cols)

    bo = "<" if endian == "<" else ">"
    floats = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}f4")
    uints = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}u4")
    ints = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}i4")

    rows: List[list] = []
    i = 0
    while i + stride <= n_words:
        raw_eid = int(uints[i])
        # Validate: packed as 10*EID + loc, loc in 1..9, EID in range
        if not (
            raw_eid >= 10 and 1 <= (raw_eid % 10) <= 9 and raw_eid // 10 <= max_eid
        ):
            i += 1
            continue
        # Must be followed by CEN/ marker
        if int(uints[i + 1]) != CEN_MARKER_UINT:
            i += 1
            continue

        eid = raw_eid // 10
        # Centroid forces: words i+3 .. i+10
        cen_f = floats[
            i
            + _CQUAD4_CORNER_CEN_OFFSET : i
            + _CQUAD4_CORNER_CEN_OFFSET
            + _CQUAD4_CORNER_CEN_WORDS
        ].tolist()
        rows.append([eid, 0] + cen_f)

        # Corner rows
        corner_base = i + _CQUAD4_CORNER_CEN_OFFSET + _CQUAD4_CORNER_CEN_WORDS
        for _ in range(_CQUAD4_N_CORNERS):
            if corner_base + _CQUAD4_CORNER_STRIDE > n_words:
                break
            raw_grid = int(uints[corner_base])
            # Corner GRID IDs are stored as plain integers (not packed with
            # device code the way element IDs are).
            if raw_grid == 0:
                break
            grid_id = raw_grid
            if not (0 < grid_id <= max_grid):
                break
            cf = floats[corner_base + 1 : corner_base + _CQUAD4_CORNER_STRIDE].tolist()
            rows.append([eid, grid_id] + cf)
            corner_base += _CQUAD4_CORNER_STRIDE

        i += stride

    return pd.DataFrame(rows, columns=cols)


def _decode_oef1_bar_payload(
    payload: bytes,
    endian: str = "<",
    float_thr: float = 1e-6,
    max_eid: int = 1_000_000,
) -> pd.DataFrame:
    """
    Decode CBAR/CBEAM element forces: 9 words per element.

    Word layout: [packed_eid, BM1A, BM2A, BM1B, BM2B, TS1, TS2, AF, TRQ]
    """
    import numpy as np

    n_words = len(payload) // 4
    cols = _CBAR_FORCE_COLS
    if n_words < 9:
        return pd.DataFrame(columns=cols)

    bo = "<" if endian == "<" else ">"
    floats = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}f4")
    words_i = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}i4")

    rows: List[list] = []
    stride = 9
    for offset in range(0, n_words - stride + 1, stride):
        raw_id = int(words_i[offset])
        if raw_id >= 10:
            loc = raw_id % 10
            eid = raw_id // 10
            if 1 <= loc <= 9 and 1 <= eid <= max_eid:
                forces = floats[offset + 1 : offset + 9]
                if np.all(np.isfinite(forces)):
                    rows.append([eid] + forces.tolist())

    return pd.DataFrame(rows, columns=cols)


def _decode_oef1_bush_payload(
    payload: bytes,
    endian: str = "<",
    max_eid: int = 1_000_000,
) -> pd.DataFrame:
    """
    Decode CBUSH element forces: 7 words per element.

    Word layout: [packed_eid, FX, FY, FZ, MX, MY, MZ]
    where packed_eid = 10 * EID + device_code.
    """
    import numpy as np

    n_words = len(payload) // 4
    cols = _CBUSH_FORCE_COLS
    if n_words < 7:
        return pd.DataFrame(columns=cols)

    bo = "<" if endian == "<" else ">"
    floats = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}f4")
    words_i = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}i4")

    rows: List[list] = []
    stride = 7
    for offset in range(0, n_words - stride + 1, stride):
        raw_id = int(words_i[offset])
        if raw_id >= 10:
            eid = raw_id // 10
            if 1 <= eid <= max_eid:
                forces = floats[offset + 1 : offset + 7]
                if np.all(np.isfinite(forces)):
                    rows.append([eid] + forces.tolist())

    return pd.DataFrame(rows, columns=cols)


def _decode_oef1_generic_payload(
    payload: bytes,
    endian: str = "<",
    float_thr: float = 1e-6,
    max_eid: int = 1_000_000,
) -> pd.DataFrame:
    """Decode OEF1 with generic F1..F8 column names (fallback for unknown types)."""
    import numpy as np

    n_words = len(payload) // 4
    cols = _GENERIC_FORCE_COLS
    if n_words < 9:
        return pd.DataFrame(columns=cols)

    bo = "<" if endian == "<" else ">"
    floats = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}f4")
    words_i = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}i4")
    words_u = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}u4")

    near_zero = np.abs(floats) < float_thr
    matches = np.where(near_zero[:-2] & near_zero[1:-1] & near_zero[2:])[0].tolist()

    rows: List[list] = []
    seen: set = set()

    if matches:
        for m in matches:
            found = None
            for j in range(3):
                idx = m + j
                if idx >= n_words:
                    break
                for val in (int(words_u[idx]), int(words_i[idx])):
                    if val >= 10:
                        loc = val % 10
                        eid = val // 10
                        if 1 <= loc <= 9 and 1 <= eid <= max_eid:
                            found = (eid, loc)
                            break
                if found:
                    break
            if not found or found[0] in seen:
                continue
            start_f = m + 3
            if start_f + 8 <= n_words:
                forces = floats[start_f : start_f + 8]
                if not np.all(np.isfinite(forces)):
                    continue
                rows.append([found[0], found[1]] + forces.tolist())
                seen.add(found[0])

    if not rows:
        for offset in range(0, n_words - 8, 9):
            raw_id = int(words_i[offset])
            if raw_id >= 10:
                loc = raw_id % 10
                eid = raw_id // 10
                if 1 <= loc <= 9 and 1 <= eid <= max_eid:
                    forces = floats[offset + 1 : offset + 9]
                    if np.all(np.isfinite(forces)):
                        rows.append([eid, loc] + forces.tolist())

    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# CBEAM per-station force decoder (NUMWDE = 100)
# ---------------------------------------------------------------------------


def _decode_oef1_cbeam_payload(
    payload: bytes,
    endian: str = "<",
    max_eid: int = 10_000_000,
) -> pd.DataFrame:
    """
    Decode CBEAM element forces from OEF1.

    Layout: NUMWDE=100 words per element.
      Word 0       : packed_eid_device  (10*EID + device_code)
      Words 1-99   : 11 stations x 9 words each
                     [GRID, SD, BM1, BM2, WS1, WS2, AF, TRQ, WARPING]

    Stations where GRID==0 AND SD==0.0 are padding rows and are skipped.
    WARPING is stored but dropped from the output (zero in static analyses).
    """
    import numpy as np

    n_words = len(payload) // 4
    stride = _CBEAM_OEF_NUM_WIDE
    n_per_station = _CBEAM_OEF_WORDS_PER_STATION
    n_stations = _CBEAM_OEF_STATIONS

    if n_words < stride:
        return pd.DataFrame(columns=_CBEAM_FORCE_COLS)

    bo = "<" if endian == "<" else ">"
    ints = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}i4")
    floats = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}f4")

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
            if grid == 0 and sd == 0.0:
                continue
            bm1 = float(floats[st_base + 2])
            bm2 = float(floats[st_base + 3])
            ws1 = float(floats[st_base + 4])
            ws2 = float(floats[st_base + 5])
            af = float(floats[st_base + 6])
            trq = float(floats[st_base + 7])
            # WARPING at st_base+8 is always zero for static analysis, skip
            rows.append([eid, grid, sd, bm1, bm2, ws1, ws2, af, trq])

    return pd.DataFrame(rows, columns=_CBEAM_FORCE_COLS)


def _decode_oef1_cgap_payload(
    payload: bytes,
    endian: str = "<",
    max_eid: int = 10_000_000,
) -> pd.DataFrame:
    """
    Decode CGAP element forces/displacements: 9 words per element.

    Word layout per element:
      Word 0: packed_eid  (10 * EID + device_code)
      Word 1: COMP_X   — axial compressive force (element X)
      Word 2: SHEAR_Y  — shear force (element Y)
      Word 3: SHEAR_Z  — shear force (element Z)
      Word 4: AXIAL_U  — axial displacement (element X)
      Word 5: TOTAL_V  — total displacement (element Y)
      Word 6: TOTAL_W  — total displacement (element Z)
      Word 7: SLIP_V   — slip displacement (element Y)
      Word 8: SLIP_W   — slip displacement (element Z)

    Column names match the Nastran F06 header:
      EID, COMP_X, SHEAR_Y, SHEAR_Z, AXIAL_U, TOTAL_V, TOTAL_W, SLIP_V, SLIP_W
    """
    import numpy as np

    n_words = len(payload) // 4
    stride = 9
    cols = _CGAP_FORCE_COLS
    if n_words < stride:
        return pd.DataFrame(columns=cols)

    bo = "<" if endian == "<" else ">"
    ints = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}i4")
    floats = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}f4")

    n_elem = n_words // stride
    raw_ids = ints[::stride][:n_elem]
    eids = raw_ids // 10
    locs = raw_ids % 10
    mask = (eids >= 1) & (eids <= max_eid) & (locs >= 1) & (locs <= 9)

    rows_eid = eids[mask]
    base = np.where(mask)[0] * stride

    result = pd.DataFrame(
        {
            "EID": rows_eid.astype("int32"),
            "COMP_X": floats[base + 1],
            "SHEAR_Y": floats[base + 2],
            "SHEAR_Z": floats[base + 3],
            "AXIAL_U": floats[base + 4],
            "TOTAL_V": floats[base + 5],
            "TOTAL_W": floats[base + 6],
            "SLIP_V": floats[base + 7],
            "SLIP_W": floats[base + 8],
        }
    )
    return result


# ---------------------------------------------------------------------------
# Public decoder
# ---------------------------------------------------------------------------


def decode_oef1(
    inv: OP2Inventory, header_index: int, ekey_index: int = None
) -> pd.DataFrame:
    """
    Decode an OEF1 element force block.

    The element type is read from the EKEY record to select the correct
    column names:

    * CQUAD4/CTRIA3 shell (centroid only, NUMWDE=11 or 9):
      ``EID, NX, NY, NXY, MX, MY, MXY, QX, QY``
    * CQUAD4 shell with corner output (NUMWDE=47):
      ``EID, GRID, NX, NY, NXY, MX, MY, MXY, QX, QY``
      (GRID=0 for centroid, actual grid ID for corner rows)
    * CBAR/CBEAM bar elements:
      ``EID, BM1A, BM2A, BM1B, BM2B, TS1, TS2, AF, TRQ``
    * CGAP gap elements:
      ``EID, COMP_X, SHEAR_Y, SHEAR_Z, AXIAL_U, TOTAL_V, TOTAL_W, SLIP_V, SLIP_W``
    * Unknown element types:
      ``EID, LOC, F1 ... F8``
    """
    etype = _etype_for_oef_header(
        inv, ekey_index if ekey_index is not None else header_index
    )

    # Determine NUMWDE early so we can set the correct minimum data record size.
    # Small element counts (e.g. 2 CTRIA3 elements) produce data records that are
    # only numwde*4 bytes each — well below the default 1000-byte threshold.
    numwde = None
    if ekey_index is not None:
        rec_e = inv.records[ekey_index]
        if rec_e.info.length == 584:
            import struct as _struct

            numwde = _struct.unpack(f"{inv.endian}146i", rec_e.data)[9]

    min_db = numwde * 4 if numwde else 1000

    if ekey_index is not None:
        first_idx = first_data_record_after_ekey(inv, ekey_index, min_data_bytes=min_db)
        payload, data_idx, _all_recs = load_data_bytes(
            inv, ekey_index, first_idx=first_idx, min_data_bytes=min_db
        )
    else:
        payload, data_idx, _all_recs = load_data_bytes(
            inv, header_index, min_data_bytes=min_db
        )

    if etype in _SHELL_ETYPES:
        if numwde == _CQUAD4_CORNER_NUMWDE:
            # CQUAD4/CQUADR with corner output
            df = _decode_oef1_shell_corner_payload(payload, endian=inv.endian)
        elif numwde == 9:
            # CTRIA3 centroid-only (EID-first, 9 words/element)
            df = _decode_oef1_ctria3_payload(payload, endian=inv.endian)
        else:
            # Centroid-only CQUAD4 (NUMWDE=11)
            df = _decode_oef1_shell_payload(payload, endian=inv.endian)
    elif etype in _BAR_ETYPES:
        # CBEAM uses a per-station layout (NUMWDE=100) identical in spirit to the
        # OES stress layout.  CBAR uses the simpler 9-word/element layout.
        if etype == 2 and numwde == _CBEAM_OEF_NUM_WIDE:
            df = _decode_oef1_cbeam_payload(payload, endian=inv.endian)
        else:
            df = _decode_oef1_bar_payload(payload, endian=inv.endian)
    elif etype in _BUSH_ETYPES:
        df = _decode_oef1_bush_payload(payload, endian=inv.endian)
    elif etype in _GAP_ETYPES:
        df = _decode_oef1_cgap_payload(payload, endian=inv.endian)
    else:
        df = _decode_oef1_generic_payload(payload, endian=inv.endian)

    df.attrs["header_record"] = header_index
    df.attrs["data_record"] = data_idx
    df.attrs["all_data_records"] = _all_recs
    df.attrs["element_type"] = etype
    df.attrs["numwde"] = numwde
    return df
