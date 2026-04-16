# op2_native/decoders/oes_search.py
from __future__ import annotations
import struct
from typing import Dict, List, Tuple

from ..models import OP2Inventory

# Element type → category
_SHELL_ETYPES = {
    33,  # CQUAD4 (MSC)
    73,  # CQUAD4 (NX)
    74,  # CTRIA3
    144,  # CQUAD4 (alt)
    64,  # CQUAD8
    75,  # CTRIA6
    82,  # CQUADR
    70,  # CTRIAR
}
_SOLID_ETYPES = {
    39,  # CTETRA (MSC)
    67,  # CPENTA (MSC)
    68,  # CHEXA  (MSC)
    85,  # CTETRA (NX)
    91,  # CPENTA (NX)
    93,  # CHEXA  (NX)
}
_BAR_ETYPES = {
    2,  # CBEAM
    34,  # CBAR
    100,  # CBAR (alt)
}
_BUSH_ETYPES = {
    102,  # CBUSH
}


def _find_token(inv: OP2Inventory, token: str) -> List[int]:
    """Return record indices of 8-byte table-name records containing *token*."""
    t = token.encode("ascii", "ignore")
    return [r.info.index for r in inv.records if r.info.length == 8 and t in r.data]


def _etype_from_ekey_words(words) -> int:
    """
    Return the element type from a 146-word EKEY record.

    Word[2] carries the element type in NX 2019+ files; word[7] carries it
    in some older NX/MSC variants where word[2] is zero.  Fall back to
    word[7] when word[2] is zero.
    """
    return words[2] if words[2] != 0 else words[7]


def _etype_for_header(inv: OP2Inventory, header_index: int) -> int:
    """
    Return the element type code from the first 584-byte EKEY record that
    follows a result table header.  Returns 0 if not found.
    """
    for i in range(header_index + 1, min(len(inv.records), header_index + 30)):
        rec = inv.records[i]
        if rec.info.length == 584:
            words = struct.unpack(f"{inv.endian}146i", rec.data)
            return _etype_from_ekey_words(words)
    return 0


def _find_ekeys_in_table(
    inv: OP2Inventory, header_idx: int
) -> List[Tuple[int, int, int, int]]:
    """
    Scan a result table and return one entry per element-type sub-block.

    Scans forward from ``header_idx + 1`` until the next 8-byte table-name
    record (which always marks the start of a new table/subcase group) or
    the end of the inventory is reached.  No fixed-record-count limit is
    imposed; the 8-byte boundary alone is sufficient because Nastran writes
    all subcases of a given result type under a single table-name record.

    Returns
    -------
    list of (ekey_idx, first_data_idx, etype, numwde)
        Only sub-blocks that have at least one valid data record are included.
    """
    results: List[Tuple[int, int, int, int]] = []
    records = inv.records
    n = len(records)
    i = header_idx + 1
    while i < n:
        r = records[i]
        if r.info.length == 8:  # next table header
            break
        if r.info.length == 584:  # EKEY record
            ekey_i = i
            words = struct.unpack(f"{inv.endian}146i", r.data)
            etype = _etype_from_ekey_words(words)
            numwde = words[9]
            if etype > 0 and numwde > 0:
                # Look for a data block within the next 20 records.
                # Use numwde*4 as the minimum data size (1 element worth of bytes)
                # so small tables (e.g. 2 CTRIA3 elements = 72 bytes) are not missed.
                min_data_bytes = numwde * 4
                first_data: int = -1
                for j in range(ekey_i + 1, min(n, ekey_i + 20)):
                    rj = records[j]
                    if rj.info.length == 8:  # next table
                        break
                    if rj.info.length >= min_data_bytes:
                        first_data = j
                        break
                if first_data >= 0:
                    results.append((ekey_i, first_data, etype, numwde))
        i += 1
    return results


def find_oes_tables(inv: OP2Inventory) -> Dict[str, List[int]]:
    """Return ``{token: [record_indices]}`` for every OES* table header found.

    Each 8-byte record is assigned to the *longest* matching token only,
    so ``OES1X1`` is never double-counted as also matching ``OES1X``/``OES1``.
    """
    tokens = ["OES1X1", "OES1X", "OES1C", "OES1"]  # longest first
    out: Dict[str, List[int]] = {}
    claimed: set = set()
    for tok in tokens:
        hits = [i for i in _find_token(inv, tok) if i not in claimed]
        if hits:
            out[tok] = hits
            claimed.update(hits)
    return out


def classify_oes_headers(
    inv: OP2Inventory,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Classify every OES* element-type sub-block by element category.

    A single table header may contain multiple element-type sub-blocks
    (one per EKEY record within the table).  Each sub-block is returned
    as a ``(table_header_idx, ekey_idx, sc_offset)`` 3-tuple.

    ``sc_offset`` is 0 for the first occurrence of each element type
    within a table, 1 for the second (i.e., subcase 2 data follows
    subcase 1 in the same table without a new IDENT record), and so on.
    The caller should add ``sc_offset`` to the base subcase ID read from
    the IDENT record to obtain the correct result subcase.

    Returns
    -------
    shell_blocks, solid_blocks, bar_blocks, bush_blocks
        Each is a list of ``(header_idx, ekey_idx, sc_offset)`` 3-tuples.
    """
    tables = find_oes_tables(inv)
    all_hdrs = sorted(idx for hits in tables.values() for idx in hits)
    shells: List[Tuple[int, int, int]] = []
    solids: List[Tuple[int, int, int]] = []
    bars: List[Tuple[int, int, int]] = []
    bushes: List[Tuple[int, int, int]] = []
    for hdr in all_hdrs:
        # Track how many times each element type has appeared in this table
        etype_count: dict = {}
        for ekey_idx, _first_data, etype, _numwde in _find_ekeys_in_table(inv, hdr):
            sc_offset = etype_count.get(etype, 0)
            etype_count[etype] = sc_offset + 1
            entry = (hdr, ekey_idx, sc_offset)
            if etype in _SOLID_ETYPES:
                solids.append(entry)
            elif etype in _BAR_ETYPES:
                bars.append(entry)
            elif etype in _BUSH_ETYPES:
                bushes.append(entry)
            else:
                shells.append(entry)
    return shells, solids, bars, bushes


def find_oef_tables(inv: OP2Inventory) -> Dict[str, List[int]]:
    """Return ``{token: [record_indices]}`` for every OEF* table header found.

    Each record is assigned to the *longest* matching token only.
    """
    tokens = ["OEF1X", "OEF1"]  # longest first
    out: Dict[str, List[int]] = {}
    claimed: set = set()
    for tok in tokens:
        hits = [i for i in _find_token(inv, tok) if i not in claimed]
        if hits:
            out[tok] = hits
            claimed.update(hits)
    return out


def find_ostr_tables(inv: OP2Inventory) -> Dict[str, List[int]]:
    """Return ``{token: [record_indices]}`` for OSTR1* (strain) headers.

    Each record is assigned to the *longest* matching token only.
    ``OSTR1EL`` (element-CS strains) is intentionally excluded here; use
    :func:`find_ostr_el_tables` for those.
    """
    # OSTR1EL must be claimed first so it is NOT caught by the shorter OSTR1 token.
    tokens = ["OSTR1X", "OSTR1EL", "OSTR1C", "OSTR1"]  # longest first
    out: Dict[str, List[int]] = {}
    claimed: set = set()
    for tok in tokens:
        hits = [i for i in _find_token(inv, tok) if i not in claimed]
        if hits:
            claimed.update(hits)
            if tok != "OSTR1EL":  # keep EL out of the basic-CS result
                out[tok] = hits
    return out


def find_ostr_el_tables(inv: OP2Inventory) -> Dict[str, List[int]]:
    """Return ``{token: [record_indices]}`` for ``OSTR1EL`` (element-CS strains)."""
    hits = _find_token(inv, "OSTR1EL")
    return {"OSTR1EL": hits} if hits else {}


def classify_ostr_el_headers(
    inv: OP2Inventory,
) -> Tuple[List, List, List, List]:
    """
    Classify every ``OSTR1EL`` element-type sub-block by element category.

    Identical logic to :func:`classify_ostr_headers` but applied to the
    element-coordinate-system strain table ``OSTR1EL``.
    Returns ``(shell_blocks, solid_blocks, bar_blocks, bush_blocks)``.
    """
    tables = find_ostr_el_tables(inv)
    all_hdrs = sorted(idx for hits in tables.values() for idx in hits)
    shells: List[Tuple[int, int, int]] = []
    solids: List[Tuple[int, int, int]] = []
    bars: List[Tuple[int, int, int]] = []
    bushes: List[Tuple[int, int, int]] = []
    for hdr in all_hdrs:
        etype_count: dict = {}
        for ekey_idx, _first_data, etype, _numwde in _find_ekeys_in_table(inv, hdr):
            sc_offset = etype_count.get(etype, 0)
            etype_count[etype] = sc_offset + 1
            entry = (hdr, ekey_idx, sc_offset)
            if etype in _SOLID_ETYPES:
                solids.append(entry)
            elif etype in _BAR_ETYPES:
                bars.append(entry)
            elif etype in _BUSH_ETYPES:
                bushes.append(entry)
            else:
                shells.append(entry)
    return shells, solids, bars, bushes


def classify_ostr_headers(
    inv: OP2Inventory,
) -> Tuple[List, List, List, List]:
    """
    Classify every OSTR1* element-type sub-block by element category.

    Identical logic to :func:`classify_oes_headers` but applied to OSTR1
    (strain) tables.  Returns
    ``(shell_blocks, solid_blocks, bar_blocks, bush_blocks)``
    where each entry is a ``(header_idx, ekey_idx, sc_offset)`` 3-tuple.
    """
    tables = find_ostr_tables(inv)
    all_hdrs = sorted(idx for hits in tables.values() for idx in hits)
    shells: List[Tuple[int, int, int]] = []
    solids: List[Tuple[int, int, int]] = []
    bars: List[Tuple[int, int, int]] = []
    bushes: List[Tuple[int, int, int]] = []
    for hdr in all_hdrs:
        etype_count: dict = {}
        for ekey_idx, _first_data, etype, _numwde in _find_ekeys_in_table(inv, hdr):
            sc_offset = etype_count.get(etype, 0)
            etype_count[etype] = sc_offset + 1
            entry = (hdr, ekey_idx, sc_offset)
            if etype in _SOLID_ETYPES:
                solids.append(entry)
            elif etype in _BAR_ETYPES:
                bars.append(entry)
            elif etype in _BUSH_ETYPES:
                bushes.append(entry)
            else:
                shells.append(entry)
    return shells, solids, bars, bushes
