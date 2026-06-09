# op2_native/decoders/oes_solid.py
"""
Decoder for solid element (CHEXA/CPENTA/CTETRA) stress/strain blocks
from OES1X1 / OSTR1X tables.

Nastran OP2 real SORT1 solid stress layout
------------------------------------------
Each element block in the data record looks like:

    [packed_eid_device,  # word 0: 10*EID + device_code
     <grid_point rows>]

Each grid-point row is 8 words:

    [grid_id,            # int32 (0 = centroid/corner average)
     SX, SY, SZ,        # float32 normal stresses
     SXY, SYZ, SZX,     # float32 shear stresses
     von_mises_or_P]     # float32 (von Mises or mean pressure)

Number of grid-point rows per element depends on element type:
  CTETRA (element type 39) : 5  rows (1 centroid + 4 corner nodes)
  CPENTA (element type 67) : 7  rows (1 centroid + 6 corner nodes)
  CHEXA  (element type 68) : 9  rows (1 centroid + 8 corner nodes)

The element type is encoded in the EKEY (geometry) record that precedes
the data record.  We read it from that record (word index 7 if available),
but also try to infer the row width directly from the data when the
geometry record is not available.

Output columns
--------------
  EID          element ID
  GRID         grid point id (0 = centroid)
  SX SY SZ     normal stresses
  SXY SYZ SZX  shear stresses
  VON_MISES    von Mises stress (or pressure P for some output requests)
"""
import struct
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..models import OP2Inventory
from .oes_peek import load_data_bytes

# Map element-type code → number of grid-point rows per element
# (centroid row + one row per corner node)
_ETYPE_TO_NROWS: Dict[int, int] = {
    39: 5,  # CTETRA  (4 corners + 1 centroid)
    67: 7,  # CPENTA  (6 corners + 1 centroid)
    68: 9,  # CHEXA   (8 corners + 1 centroid)
    # Additional codes sometimes seen in NX Nastran:
    85: 5,  # CTETRA (NX variant)
    91: 7,  # CPENTA (NX variant)
    93: 9,  # CHEXA  (NX variant)
}

_WORDS_PER_GP = 8  # grid_id + 7 stress/strain floats
_OUT_COLS = ["EID", "GRID", "SX", "SY", "SZ", "SXY", "SYZ", "SZX", "VON_MISES"]

# Extended MSC CTETRA/CPENTA/CHEXA format (NUMWDE 109/169/256):
#   Each element block: 1 EID word + 1 centroid row + N corner rows
#   Centroid row words  = centroid_wds
#   Corner   row words  = corner_wds
# numwde = 1 + centroid_wds + N_corners * corner_wds
# CTETRA: 1 + 24 + 4*21 = 109
# CPENTA: 1 + 24 + 6*21 = 169  (unverified)
# CHEXA:  1 + 24 + 8*21 = 193  (unverified, but 193 ≠ common NUMWDE=219)
# We only verify 109 here; other values fall back to _infer_nrows_per_elem.
_EXT_CENTROID_WDS = 24  # words for centroid row (incl. grid_id word)
_EXT_CORNER_WDS = 21  # words per corner row (incl. grid_id word)
_EXT_NUMWDE_NROWS: Dict[int, int] = {
    109: 4,  # CTETRA MSC extended: 4 corners
    169: 6,  # CPENTA MSC extended: 6 corners (tentative)
}
_EXT_NCOLS = 8  # EID + GRID + 6 stress floats mapped to SX..SZX
# For the extended format we output the first 6 float fields of each row.
# Centroid row: grid=0, fields w[1..6] → SX SY SZ SXY SYZ SZX (VON_MISES not stored separately)
# Corner rows:  grid=grid_id, fields w[1..6]
_EXT_OUT_COLS = ["EID", "GRID", "SX", "SY", "SZ", "SXY", "SYZ", "SZX", "VON_MISES"]


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


def _infer_nrows_per_elem(data: bytes, endian: str = "<") -> Optional[int]:
    """
    Try to infer the number of grid-point rows per element by looking at
    the first few element blocks.  We expect:
      word[0] = packed EID (10*EID + device, small digit 1-9)
      word[1..] = grid point rows of _WORDS_PER_GP words each
    We try nrows = 5, 7, 9 and pick the one that gives the most consistent
    packed-EID pattern across the first ~5 elements.
    """
    n_words = len(data) // 4
    ei = f"{endian}i"

    def read_int(i: int) -> int:
        return struct.unpack(ei, data[i * 4 : i * 4 + 4])[0]

    def looks_packed_eid(v: int) -> bool:
        if v <= 0:
            return False
        loc = v % 10
        eid = v // 10
        return 1 <= loc <= 9 and 1 <= eid <= 99_999_999

    for nrows in (5, 7, 9):
        stride = 1 + nrows * _WORDS_PER_GP  # words per element block
        hits = 0
        offset = 0
        for _ in range(8):
            if offset + stride > n_words:
                break
            if looks_packed_eid(read_int(offset)):
                hits += 1
            offset += stride
        if hits >= 3:
            return nrows
    return None


def _decode_solid_payload(
    data: bytes,
    nrows_per_elem: int,
    endian: str = "<",
    max_eid: int = 99_999_999,
) -> pd.DataFrame:
    """
    Decode a raw solid stress/strain data record.

    Parameters
    ----------
    data : bytes
    nrows_per_elem : int
        Number of grid-point rows per element (5/7/9 for CTETRA/CPENTA/CHEXA).
    """
    stride = 1 + nrows_per_elem * _WORDS_PER_GP  # words per element block
    n_words = len(data) // 4
    if n_words < stride:
        return pd.DataFrame(columns=_OUT_COLS)

    bo = "<" if endian == "<" else ">"
    ints = np.frombuffer(data[: n_words * 4], dtype=f"{bo}i4")
    floats = np.frombuffer(data[: n_words * 4], dtype=f"{bo}f4")

    rows: List[list] = []
    offset = 0

    while offset + stride <= n_words:
        raw_eid = int(ints[offset])
        if raw_eid <= 0:
            break
        loc = raw_eid % 10
        eid = raw_eid // 10
        if not (1 <= loc <= 9 and 1 <= eid <= max_eid):
            break

        gp_base = offset + 1
        for gp in range(nrows_per_elem):
            gp_off = gp_base + gp * _WORDS_PER_GP
            grid = int(ints[gp_off])
            stresses = floats[gp_off + 1 : gp_off + 8]
            if not np.all(np.isfinite(stresses)):
                continue
            rows.append([eid, grid] + stresses.tolist())

        offset += stride

    return pd.DataFrame(rows, columns=_OUT_COLS)


def _decode_solid_payload_extended(
    data: bytes,
    n_corners: int,
    endian: str = "<",
    max_eid: int = 99_999_999,
) -> pd.DataFrame:
    """
    Decode extended MSC solid stress (NUMWDE=109/169).

    Layout per element (NUMWDE words total):
      w[0]                             packed EID (10*EID + device)
      w[1 .. centroid_wds]             centroid row (24 words): same field layout
                                       as corner rows (21 words) plus 3 trailing
                                       words that are ignored
      w[centroid_wds+1 ..]             corner rows (21 words each)

    Both centroid (GRID=0) and corner rows are emitted.

    Centroid row layout (24 words, standard 8-word format + extra data):
      [0]  GRID = 0
      [1]  SX   (Normal-X)
      [2]  SY   (Normal-Y)
      [3]  SZ   (Normal-Z)
      [4]  SXY  (Shear-XY)
      [5]  SYZ  (Shear-YZ)
      [6]  SZX  (Shear-ZX)
      [7]  Von Mises
      [8-23] additional centroid data (principal stresses, etc.), ignored

    Corner row layout (21 words, extended format):
      [0]  grid_id
      [1]  Normal-X  (SX)
      [2]  Shear-XY  (SXY)
      [3]  Principal-A
      [4-6] LX direction cosines
      [7]  Mean Pressure
      [8]  Von Mises  (VM)
      [9]  Normal-Y  (SY)
      [10] Shear-YZ  (SYZ)
      [11] Principal-B
      [12-14] LY direction cosines
      [15] Normal-Z  (SZ)
      [16] Shear-ZX  (SZX)
      [17] Principal-C
      [18-20] LZ direction cosines
    """
    centroid_wds = _EXT_CENTROID_WDS  # 24
    corner_wds = _EXT_CORNER_WDS  # 21
    stride = 1 + centroid_wds + n_corners * corner_wds
    n_words = len(data) // 4
    if n_words < stride:
        return pd.DataFrame(columns=_EXT_OUT_COLS)

    bo = "<" if endian == "<" else ">"
    ints = np.frombuffer(data[: n_words * 4], dtype=f"{bo}i4")
    floats = np.frombuffer(data[: n_words * 4], dtype=f"{bo}f4")

    rows: List[list] = []
    offset = 0

    def _extract_centroid(base: int) -> None:
        # Standard 8-word layout: [GRID=0, SX, SY, SZ, SXY, SYZ, SZX, VM, ...]
        sx = float(floats[base + 1])
        sy = float(floats[base + 2])
        sz = float(floats[base + 3])
        sxy = float(floats[base + 4])
        syz = float(floats[base + 5])
        szx = float(floats[base + 6])
        vm = float(floats[base + 7])
        if np.isfinite(sx) and np.isfinite(sy) and np.isfinite(sz):
            rows.append([eid, 0, sx, sy, sz, sxy, syz, szx, vm])

    def _extract_corner(base: int, grid: int) -> None:
        # Extended corner layout: SX[1], SXY[2], VM[8], SY[9], SYZ[10], SZ[15], SZX[16]
        sx = float(floats[base + 1])
        sxy = float(floats[base + 2])
        sy = float(floats[base + 9])
        syz = float(floats[base + 10])
        sz = float(floats[base + 15])
        szx = float(floats[base + 16])
        vm = float(floats[base + 8])
        if np.isfinite(sx) and np.isfinite(sy) and np.isfinite(sz):
            rows.append([eid, grid, sx, sy, sz, sxy, syz, szx, vm])

    while offset + stride <= n_words:
        raw_eid = int(ints[offset])
        if raw_eid <= 0:
            break
        loc = raw_eid % 10
        eid = raw_eid // 10
        if not (1 <= loc <= 9 and 1 <= eid <= max_eid):
            break

        # Centroid row (GRID=0): standard 8-word format.
        _extract_centroid(offset + 1)

        # Corner rows: extended 21-word format.
        for k in range(n_corners):
            cr_base = offset + 1 + centroid_wds + k * corner_wds
            grid = int(ints[cr_base])
            _extract_corner(cr_base, grid)

        offset += stride

    return pd.DataFrame(rows, columns=_EXT_OUT_COLS)


def decode_oes_solid(
    inv: OP2Inventory, header_index: int, ekey_index: int = None
) -> pd.DataFrame:
    """
    Decode a solid element stress block from OES1X1.

    Returns
    -------
    DataFrame with columns ``EID, GRID, SX, SY, SZ, SXY, SYZ, SZX, VM``.
    ``GRID == 0`` rows are the centroid (average) value.
    """
    ekey_or_hdr = ekey_index if ekey_index is not None else header_index

    # Read etype and NUMWDE from EKEY record
    etype = _elem_type_from_ekey(inv, ekey_or_hdr)
    numwde: Optional[int] = None
    rec = inv.records[ekey_or_hdr]
    if rec.info.length == 584:
        words = struct.unpack(f"{inv.endian}146i", rec.data)
        numwde = words[9]

    payload, data_idx, all_recs = load_data_bytes(inv, ekey_or_hdr)

    # Dispatch to extended decoder if NUMWDE matches a known extended layout
    n_corners = _EXT_NUMWDE_NROWS.get(numwde) if numwde is not None else None
    if n_corners is not None:
        df = _decode_solid_payload_extended(payload, n_corners, inv.endian)
    else:
        nrows = _ETYPE_TO_NROWS.get(etype) if etype is not None else None
        if nrows is None:
            nrows = _infer_nrows_per_elem(payload, inv.endian)
        if nrows is None:
            raise ValueError(
                f"Cannot determine solid element row width for header rec {header_index} "
                f"(element type code: {etype}, NUMWDE: {numwde})"
            )
        df = _decode_solid_payload(payload, nrows, inv.endian)

    df.attrs["header_record"] = header_index
    df.attrs["data_record"] = data_idx
    df.attrs["all_data_records"] = all_recs
    df.attrs["element_type"] = etype
    df.attrs["numwde"] = numwde
    return df
