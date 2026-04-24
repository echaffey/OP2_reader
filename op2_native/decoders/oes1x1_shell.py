# op2_native/decoders/oes1x1_shell.py
from typing import List, Tuple
import struct
import numpy as np
import pandas as pd

from ..models import OP2Inventory
from .oes_peek import load_data_bytes


HEADER_WORDS = 3  # first 3 words are header/control
# Marker-driven format: each element block begins with
#   [packed_eid, CEN/(4bytes), n_corners] then 8+8 float stress words.
# Effective stride = 19 words (1 packed_eid + 2 marker words + 16 stresses).
WORDS_PER_ELEM = 19  # 1 packed_eid + 2 CEN/ marker words + 16 stress floats

# Named stress columns for CQUAD4/CTRIA3/shell elements
# Two fiber layers (Z1=bottom, Z2=top):
#   FD        = fiber distance
#   SX, SY    = normal stresses in element system
#   TXY       = in-plane shear stress
#   ANG       = angle to principal axis (degrees)
#   MAX_PRIN  = maximum principal stress (OMAX)
#   MIN_PRIN  = minimum principal stress (OMIN)
#   VON_MISES = Von Mises stress (or Max Shear, depending on STRESS case control)
# (8 words per fiber layer x 2 layers = 16 words total)
SHELL_STRESS_COLS = [
    "EID",
    # fiber 1 (bottom / Z1)
    "FD1",
    "SX1",
    "SY1",
    "TXY1",
    "ANG1",
    "MAX_PRIN1",
    "MIN_PRIN1",
    "VON_MISES1",
    # fiber 2 (top / Z2)
    "FD2",
    "SX2",
    "SY2",
    "TXY2",
    "ANG2",
    "MAX_PRIN2",
    "MIN_PRIN2",
    "VON_MISES2",
]

# Fallback generic columns (used when full 16-column layout is not available)
_GENERIC_COLS = ["EID", "LOC"] + [f"S{i+1}" for i in range(16)]


def _name_stress_cols(vals: list) -> list:
    """
    Return the appropriate column list for a decoded row.

    If the row has exactly 16 stress values (2 fibers x 8 words each),
    return ``SHELL_STRESS_COLS`` (17 total including EID).
    Otherwise fall back to generic ``S1..Sn`` names.
    """
    n = len(vals)
    if n == 16:
        return SHELL_STRESS_COLS  # 1 + 16 = 17 columns
    return ["EID"] + [f"S{i+1}" for i in range(n)]


def _decode_packed_eid(val: int) -> Tuple[int, int]:
    """
    Decode Nastran-style packed ID used in this OES block:

        val = 10 * EID + loc

    where 'loc' is a small integer (e.g. 1=top, 2=bottom, etc. depending on table).
    If it doesn't look packed, we fall back to treating 'val' as the EID and loc=0.
    """
    loc = val % 10
    eid = val // 10
    if 1 <= loc <= 9 and eid > 0:
        return eid, loc
    # fallback: not packed in the expected way
    return val, 0


def _decode_oes1x1_payload(
    payload: bytes, endian: str = "<", max_eid: int = 10_000_000
) -> pd.DataFrame:
    """
    Decode an OES1X1-style shell stress block:

        [header_w0, header_w1, header_w2,
         elem0_f0..f15, elem0_id,
         elem1_f0..f15, elem1_id,
         ... ]

    Each float is 4 bytes (float32), each id is 4-byte int32.

    In this file, the id word is encoded as:

        id = 10 * EID + loc

    Returns DataFrame columns:
        EID, LOC, S1..S16

    where S1..S16 are the raw 16 stress floats per element.
    """
    n_words = len(payload) // 4
    if n_words < HEADER_WORDS + WORDS_PER_ELEM:
        raise ValueError("Payload too small to be OES1X1 shell block")

    # convenience view of the payload as 4-byte chunks
    words = [payload[i * 4 : (i + 1) * 4] for i in range(n_words)]

    elems: List[Tuple[int, int]] = []  # (EID, LOC)
    data: List[List[float]] = []

    # start just after the 3-word header
    offset = HEADER_WORDS
    while offset + WORDS_PER_ELEM <= n_words:
        chunk = words[offset : offset + WORDS_PER_ELEM]

        # last word in the 17-word packet is normally the packed ID (int32).
        # However some OP2 variants embed a small marker value (e.g. 12,22,32,...)
        # somewhere in the 17-word packet where the low digit is a small locator
        # (commonly '2'). Detect those marker words and prefer them as the id
        # source so we recover the true EID (marker // 10).
        # search a small window around the current chunk for a plausible packed-id
        # (some OP2 variants put the marker slightly before/after the 17-word packet)
        win_start = max(0, offset - 2)
        win_end = min(n_words, offset + WORDS_PER_ELEM + 2)
        packed_id = None
        for iwin in range(win_start, win_end):
            try:
                v = struct.unpack(f"{endian}i", words[iwin])[0]
            except struct.error:
                continue
            # plausibility: trailing digit 1..9 is loc, EID must be in range
            if v >= 10 and (v % 10) in range(1, 10):
                e_cand = v // 10
                if 1 <= e_cand <= max_eid:
                    packed_id = v
                    break

        # fallback to the conventional last-word ID
        if packed_id is None:
            packed_id = struct.unpack(f"{endian}i", chunk[-1])[0]

        eid, loc = _decode_packed_eid(packed_id)

        # bail out if EID is obviously nonsense (we probably hit another header)
        if not (1 <= eid <= max_eid):
            break

        # first 16 words are stress floats
        vals = [struct.unpack(f"{endian}f", w)[0] for w in chunk[:-1]]

        elems.append((eid, loc))
        data.append(vals)

        offset += WORDS_PER_ELEM

    cols = _name_stress_cols(data[0] if data else [])
    rows = []
    for (eid, _loc), vals in zip(elems, data):
        rows.append([eid] + vals)

    df = pd.DataFrame(rows, columns=cols)
    df.attrs["n_words"] = n_words
    df.attrs["decoded_elems"] = len(df)
    return df


def detect_eid_alignment(
    payload: bytes,
    endian: str = "<",
    header_words: int = HEADER_WORDS,
    words_per_elem: int = WORDS_PER_ELEM,
    n_check: int = 20,
    max_eid: int = 100_000,
):
    """Relaxed alignment detection.

    Instead of assuming elements appear in order 1..N, look for the column
    (start,pos) inside each words_per_elem packet that contains many marker-like
    packed IDs where (val % 10) looks like a locator (commonly 2) and the
    decoded EID (val//10) is a small positive integer. Score candidates by the
    count of such marker occurrences among the first n_check chunks.

    Returns a dict with keys: start, pos, signed, score, marker_count, sample_eids
    """
    import struct

    n_words = len(payload) // 4
    if n_words == 0:
        return {"score": 0, "start": header_words, "pos": 0, "signed": True}

    # build signed and unsigned int views once
    words_s = [
        struct.unpack(f"{endian}i", payload[i * 4 : (i + 1) * 4])[0]
        for i in range(n_words)
    ]
    words_u = [
        struct.unpack(f"{endian}I", payload[i * 4 : (i + 1) * 4])[0]
        for i in range(n_words)
    ]

    best = {
        "score": -1,
        "start": None,
        "pos": None,
        "signed": None,
        "marker_count": 0,
        "sample_eids": [],
    }

    for start in range(header_words, header_words + words_per_elem):
        for pos in range(words_per_elem):
            for signed in (True, False):
                marker_count = 0
                eids = []
                for k in range(n_check):
                    word_idx = start + k * words_per_elem + pos
                    if word_idx < 0 or word_idx >= n_words:
                        continue
                    v = words_s[word_idx] if signed else words_u[word_idx]
                    if v >= 10:
                        loc = v % 10
                        eid = v // 10
                        # treat common locator values (1..9) as markers; prefer loc==2 but accept others
                        if 1 <= loc <= 9 and 1 <= eid <= max_eid:
                            marker_count += 1
                            eids.append(eid)

                # prefer candidates with more markers; tie-breaker: lower start then lower pos
                score = marker_count
                if score > best["score"] or (
                    score == best["score"]
                    and best["start"] is not None
                    and (start, pos) < (best["start"], best["pos"])
                ):
                    best.update(
                        {
                            "score": score,
                            "start": start,
                            "pos": pos,
                            "signed": signed,
                            "marker_count": marker_count,
                            "sample_eids": eids[:10],
                        }
                    )

    return best


def _decode_oes1x1_by_alignment(
    payload: bytes, endian: str, start: int, pos: int, signed: bool
) -> pd.DataFrame:
    """Decode payload using known (start,pos) within each words_per_elem packet where
    the packed id resides. This reads each chunk's 16 floats and the id word at
    (start + k*WORDS_PER_ELEM + pos), decoding EID = (val//10) when val%10==2.
    """
    import struct

    n_words = len(payload) // 4
    words = [payload[i * 4 : (i + 1) * 4] for i in range(n_words)]

    elems = []
    data = []

    offset = start
    # ensure we don't run off the end
    while offset + WORDS_PER_ELEM <= n_words:
        chunk_words = words[offset : offset + WORDS_PER_ELEM]
        # floats are the full chunk except we don't know which of the words are the id
        # To be consistent with the original layout, read the first 16 as floats from the
        # chunk starting at offset (this is conservative) and pull the id from the computed pos
        vals = [struct.unpack(f"{endian}f", w)[0] for w in chunk_words[:16]]

        id_word_idx = offset + pos
        if id_word_idx < 0 or id_word_idx >= n_words:
            break
        id_b = payload[id_word_idx * 4 : id_word_idx * 4 + 4]
        v = (
            struct.unpack(f"{endian}i", id_b)[0]
            if signed
            else struct.unpack(f"{endian}I", id_b)[0]
        )
        if v >= 10 and (v % 10) == 2:
            eid = v // 10
            loc = v % 10
        else:
            # fall back: if not a marker, try treating last word of the chunk as id
            v2 = struct.unpack(f"{endian}i", chunk_words[-1])[0]
            eid, loc = _decode_packed_eid(v2)

        if not (eid > 0):
            break

        elems.append((eid, loc))
        data.append(vals)

        offset += WORDS_PER_ELEM

    cols = _name_stress_cols(data[0] if data else [])
    rows = [[eid] + vals for (eid, _loc), vals in zip(elems, data)]
    df = pd.DataFrame(rows, columns=cols)
    df.attrs["n_words"] = n_words
    df.attrs["decoded_elems"] = len(df)
    df.attrs["alignment_start"] = start
    df.attrs["alignment_pos"] = pos
    df.attrs["alignment_signed"] = signed
    return df


def decode_oes1x1_tria3_payload(
    payload: bytes,
    endian: str = "<",
    max_eid: int = 1_000_000,
) -> pd.DataFrame:
    """
    Decode OES1X1 CTRIA3 stress blocks (NUMWDE=17, centroid only).

    Word layout per element (17 words):
        w0:      packed_eid  (10*EID + device_code)
        w1..w16: FD1,SX1,SY1,TXY1,ANG1,MAX_PRIN1,MIN_PRIN1,VON_MISES1,
                 FD2,SX2,SY2,TXY2,ANG2,MAX_PRIN2,MIN_PRIN2,VON_MISES2

    EID is at the START of each record (unlike CQUAD4 centroid-only format
    where it is at the end).
    """
    import numpy as np

    n_words = len(payload) // 4
    if n_words < 17:
        return pd.DataFrame(columns=SHELL_STRESS_COLS)

    bo = "<" if endian == "<" else ">"
    floats = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}f4")
    words_i = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}i4")

    stride = 17
    rows = []
    for offset in range(0, n_words - stride + 1, stride):
        raw_id = int(words_i[offset])
        if raw_id < 10:
            continue
        eid = raw_id // 10
        if not (1 <= eid <= max_eid):
            continue
        stresses = floats[offset + 1 : offset + 17]
        if not np.all(np.isfinite(stresses)):
            continue
        rows.append([eid] + stresses.tolist())

    df = pd.DataFrame(rows, columns=SHELL_STRESS_COLS)
    df.attrs["decoded_elems"] = len(df)
    return df


def decode_oes1x1_by_marker(
    payload: bytes, endian: str = "<", float_thr: float = 1e-6, max_eid: int = 1_000_000
) -> pd.DataFrame:
    """Decode OES blocks by locating a 3-word float-marker preceding each element.

    Heuristic:
      - find indices where three consecutive float32 words are very small in magnitude
        (|f| < float_thr) — these often mark element starts in this OP2 variant.
      - for each candidate triple, examine the three words (as signed/unsigned ints)
        and accept a word as packed-id if val%10 in 1..9 and 1 <= val//10 <= max_eid.
      - read the 16 floats immediately after the triple as stresses.
      - return DataFrame rows with EID, LOC and S1..S16. Duplicate EIDs are ignored
        (keep first occurrence).

    Uses numpy bulk unpacking for the hot path.
    """
    import numpy as np

    n_words = len(payload) // 4
    if n_words < HEADER_WORDS + WORDS_PER_ELEM:
        return pd.DataFrame(columns=SHELL_STRESS_COLS)

    # Bulk unpack all three views in one pass each (much faster than per-word struct.unpack)
    byte_order = "<" if endian == "<" else ">"
    floats = np.frombuffer(payload[: n_words * 4], dtype=f"{byte_order}f4")
    words_i = np.frombuffer(payload[: n_words * 4], dtype=f"{byte_order}i4")
    words_u = np.frombuffer(payload[: n_words * 4], dtype=f"{byte_order}u4")

    # Vectorised marker detection: positions where 3 consecutive words are near-zero float
    thr = float_thr
    near_zero = np.abs(floats) < thr
    matches = np.where(near_zero[:-2] & near_zero[1:-1] & near_zero[2:])[0].tolist()

    seen_eids: set = set()
    rows = []
    for m in matches:
        # scan up to 3 words within this match for a packed-id
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
                        found = (idx, val, eid, loc)
                        break
            if found:
                break
        if not found:
            continue
        _marker_idx, _raw_val, eid, loc = found
        if eid in seen_eids:
            continue
        start_f = m + 3
        if start_f + 16 <= n_words:
            stresses = floats[start_f : start_f + 16]
            if not np.all(np.isfinite(stresses)):
                continue
            row = [eid] + stresses.tolist()
            rows.append(row)
            seen_eids.add(eid)

    cols = _name_stress_cols(
        rows[0][1:] if rows else []
    )  # rows[0] = [eid, f1..f16], skip eid
    df = pd.DataFrame(rows, columns=cols)
    df.attrs["n_words"] = n_words
    df.attrs["decoded_elems"] = len(df)
    df.attrs["marker_matches"] = len(matches)
    return df


# ---------------------------------------------------------------------------
# Corner-aware decoder
# ---------------------------------------------------------------------------

# Full block layout per CQUAD4 element (87 words):
#   w0        : packed_eid_cen  (10*EID + device_code)
#   w1        : CEN/ ASCII marker (4 bytes)
#   w2        : n_corners (=4 for CQUAD4)
#   w3..w18   : 16 centroid stress floats (8 per fiber × 2 fibers)
#   w19       : corner-1 grid id (plain integer, NOT packed)
#   w20..w35  : 16 corner-1 stress floats
#   w36..w52  : corner-2 grid id + 16 floats
#   w53..w69  : corner-3 grid id + 16 floats
#   w70..w86  : corner-4 grid id + 16 floats
# Total: 3 + 5×17 = 3 + 85 = 88 ... but measured gap is 87.
# Actual: 1(eid) + 2(CEN/,4) + 16(cen_stresses) + 4×(1+16) = 19+68 = 87  ✓

_ELEM_BLOCK_WORDS = 87  # words per CQUAD4 element including centroid + 4 corners
_CEN_STRESS_OFFSET = 3  # centroid stresses start at word 3 within the block
_CORNER_STRIDE = 17  # 1 grid-id + 16 stresses
_N_CORNERS = 4

# Column list with an extra GRID column (centroid uses GRID=0)
# Skip EID (index 0) from SHELL_STRESS_COLS to get FD1..VM2, then prepend EID+GRID.
SHELL_STRESS_CORNER_COLS = ["EID", "GRID"] + SHELL_STRESS_COLS[
    1:
]  # skip EID, prepend EID+GRID


def decode_oes1x1_shell_corners(
    inv: OP2Inventory,
    oes_header_index: int,
    ekey_index: int = None,
    float_thr: float = 1e-6,
    max_eid: int = 1_000_000,
    max_grid: int = 10_000_000,
) -> pd.DataFrame:
    """
    Decode shell stresses including all four corner nodes.

    Returns a DataFrame with one row per (element, location): the centroid
    row (GRID=0) followed by up to four corner rows (GRID = the corner grid ID).

    Columns
    -------
    EID, GRID, FD1, SX1, SY1, TXY1, ANG1, MAX_PRIN1, MIN_PRIN1, VON_MISES1,
    FD2, SX2, SY2, TXY2, ANG2, MAX_PRIN2, MIN_PRIN2, VON_MISES2
    """
    payload, data_rec_idx, _all_recs = load_data_bytes(
        inv, ekey_index if ekey_index is not None else oes_header_index
    )
    endian = inv.endian

    n_words = len(payload) // 4
    byte_order = "<" if endian == "<" else ">"
    floats = np.frombuffer(payload[: n_words * 4], dtype=f"{byte_order}f4")
    ints = np.frombuffer(payload[: n_words * 4], dtype=f"{byte_order}i4")
    uints = np.frombuffer(payload[: n_words * 4], dtype=f"{byte_order}u4")

    def _read_floats(start: int, count: int) -> List[float]:
        return floats[start : start + count].tolist()

    def _decode_id(raw: int):
        """Return (id_value, is_grid) — id_value is EID or GRID."""
        if raw >= 10 and 1 <= (raw % 10) <= 9:
            return raw // 10, raw % 10
        return raw, 0

    rows: List[list] = []

    # Scan for element blocks by locating the near-zero 3-word marker at word 0 of each block.
    # Word 0 = packed EID (small integer, looks near-zero as float32).
    # Word 1 = CEN/ = 0x434E452F = 1129270575 as uint, ~1.87e-10 as float (near-zero).
    # Word 2 = 4 (n_corners, very small as float32).
    CEN_MARKER_UINT = 793658691  # b'CEN/' as little-endian uint32

    i = 0
    while i + _ELEM_BLOCK_WORDS <= n_words:
        # Check word i looks like a packed EID (small positive int, loc digit 1-9)
        raw_eid = uints[i]
        if not (
            raw_eid >= 10 and 1 <= (raw_eid % 10) <= 9 and raw_eid // 10 <= max_eid
        ):
            i += 1
            continue
        # Word i+1 should be the CEN/ ASCII marker
        if uints[i + 1] != CEN_MARKER_UINT:
            i += 1
            continue

        eid = raw_eid // 10

        # Centroid stresses: words i+3 .. i+18
        cen_s = _read_floats(i + _CEN_STRESS_OFFSET, 16)

        rows.append([eid, 0] + cen_s)  # GRID=0 for centroid

        # Corner rows
        corner_start = i + _CEN_STRESS_OFFSET + 16  # = i + 19
        for _c in range(_N_CORNERS):
            if corner_start + _CORNER_STRIDE > n_words:
                break
            raw_grid = uints[corner_start]
            # Corner GRID IDs are stored as plain integers (not packed with
            # device code the way element IDs are).
            if raw_grid == 0:
                break
            grid_id = int(raw_grid)
            if not (0 < grid_id <= max_grid):
                break
            c_s = _read_floats(corner_start + 1, 16)
            rows.append([eid, grid_id] + c_s)
            corner_start += _CORNER_STRIDE

        i += _ELEM_BLOCK_WORDS

    cols = SHELL_STRESS_CORNER_COLS
    df = pd.DataFrame(rows, columns=cols)
    df.attrs["header_record"] = oes_header_index
    df.attrs["data_record"] = data_rec_idx
    df.attrs["all_data_records"] = _all_recs
    df.attrs["decode_method"] = "corner"
    return df


def decode_oes1x1_shell(
    inv: OP2Inventory,
    oes_header_index: int,
    ekey_index: int = None,
    max_eid: int = 10_000_000,
    float_thr: float = 1e-6,
) -> pd.DataFrame:
    """
    High-level helper: given the index of an OES1/OES1X1 header record in the
    OP2 inventory, locate the first 'stress-like' record after it and decode
    shell stresses into a DataFrame.

    Columns:
        EID, LOC, S1..S16  (raw stress floats)
    """
    payload, data_rec_idx, _all_recs = load_data_bytes(
        inv, ekey_index if ekey_index is not None else oes_header_index
    )

    def _set_attrs(df: pd.DataFrame, method: str) -> pd.DataFrame:
        df.attrs["header_record"] = oes_header_index
        df.attrs["data_record"] = data_rec_idx
        df.attrs["all_data_records"] = _all_recs
        df.attrs["decode_method"] = method
        return df

    # Fast path: numpy marker-driven decoder — try this first to avoid the
    # Python struct.unpack loop entirely for files that use the marker format.
    try:
        df_marker = decode_oes1x1_by_marker(
            payload, endian=inv.endian, float_thr=float_thr
        )
    except Exception:
        df_marker = pd.DataFrame()

    if not df_marker.empty:
        df_marker.attrs["marker_matches"] = df_marker.attrs.get("marker_matches")
        return _set_attrs(df_marker, "marker")

    # Slow fallback: Python-loop decoder (handles non-marker variants).
    df = _decode_oes1x1_payload(payload, endian=inv.endian, max_eid=max_eid)

    # If the slow decoder also produced very few rows, try alignment detection.
    try:
        decoded_elems = int(df.attrs.get("decoded_elems", 0))
    except Exception:
        decoded_elems = 0

    if decoded_elems < 12:
        best = detect_eid_alignment(payload, endian=inv.endian, n_check=20)
        if best.get("score", 0) >= 8:
            df_alt = _decode_oes1x1_by_alignment(payload, endian=inv.endian, start=best["start"], pos=best["pos"], signed=best["signed"])  # type: ignore[arg-type]
            df_alt.attrs["alignment_score"] = best.get("score")
            return _set_attrs(df_alt, "alignment")

    return _set_attrs(df, "conventional")
