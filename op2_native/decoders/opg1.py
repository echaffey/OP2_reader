# op2_native/decoders/opg1.py
"""
Decoder for OPG1 applied load data blocks.

Layout is identical to OQG1 (same 8-word SORT1 row format):

    [packed_id, FX, FY, FZ, MX, MY, MZ, pad]

So we reuse the same low-level decoder.
"""
import pandas as pd
from typing import List, Optional

from ..models import OP2Inventory
from .oes_peek import (
    first_grid_force_record_after,
    classify_grid_force_headers,
    load_data_bytes,
)
from .oqg1 import _decode_grid_force_payload


def classify_opg_headers(inv: OP2Inventory) -> List[tuple]:
    """Return ``[(header_idx, data_rec_idx, sc_offset), ...]`` for all OPG1 data blocks."""
    return classify_grid_force_headers(inv, "OPG1")


def decode_opg1(
    inv: OP2Inventory, header_index: int, ekey_idx: Optional[int] = None
) -> pd.DataFrame:
    """
    Decode an OPG1 applied load block.

    Parameters
    ----------
    ekey_idx : int, optional
        Direct data-record index (as returned by classify_opg_headers).  When
        provided the forward-scan heuristic is skipped.

    Returns
    -------
    DataFrame with columns ``GRID, FX, FY, FZ, MX, MY, MZ``.
    """
    data_idx = (
        ekey_idx
        if ekey_idx is not None
        else first_grid_force_record_after(inv, header_index)
    )
    payload, _first, all_recs = load_data_bytes(
        inv, header_index, min_data_bytes=32, first_idx=data_idx
    )
    df = _decode_grid_force_payload(payload, endian=inv.endian)
    df.attrs["header_record"] = header_index
    df.attrs["data_record"] = data_idx
    df.attrs["all_data_records"] = all_recs
    return df
