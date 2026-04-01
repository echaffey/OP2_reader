# op2_native/decoders/oes_cbush.py
"""
Decoder for CBUSH element stress/deformation blocks from OES1X1 tables.

Nastran OP2 real SORT1 CBUSH layout  (etype=102, NUMWDE=7)
-----------------------------------------------------------
Each element occupies exactly 7 words:

    Word 0 : packed_eid_device  (10 * EID + device_code)
    Word 1 : EX    float  translational deformation, x
    Word 2 : EY    float  translational deformation, y
    Word 3 : EZ    float  translational deformation, z
    Word 4 : ETX   float  rotational deformation, x
    Word 5 : ETY   float  rotational deformation, y
    Word 6 : ETZ   float  rotational deformation, z

Output columns
--------------
  EID, EX, EY, EZ, ETX, ETY, ETZ
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ..models import OP2Inventory
from .oes_peek import load_data_bytes

_CBUSH_STRESS_COLS = ["EID", "EX", "EY", "EZ", "ETX", "ETY", "ETZ"]
_CBUSH_NUM_WIDE = 7


def _decode_oes_cbush_payload(
    payload: bytes,
    endian: str = "<",
    max_eid: int = 1_000_000,
) -> pd.DataFrame:
    """
    Decode CBUSH spring deformation data: 7 words per element.

    Word layout: [packed_eid, EX, EY, EZ, ETX, ETY, ETZ]
    where packed_eid = 10 * EID + device_code.
    """
    n_words = len(payload) // 4
    if n_words < _CBUSH_NUM_WIDE:
        return pd.DataFrame(columns=_CBUSH_STRESS_COLS)

    bo = "<" if endian == "<" else ">"
    floats = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}f4")
    words_i = np.frombuffer(payload[: n_words * 4], dtype=f"{bo}i4")

    rows: List[list] = []
    stride = _CBUSH_NUM_WIDE
    for offset in range(0, n_words - stride + 1, stride):
        raw_id = int(words_i[offset])
        if raw_id >= 10:
            eid = raw_id // 10
            if 1 <= eid <= max_eid:
                vals = floats[offset + 1 : offset + 7]
                if np.all(np.isfinite(vals)):
                    rows.append([eid] + vals.tolist())

    return pd.DataFrame(rows, columns=_CBUSH_STRESS_COLS)


def decode_oes_cbush(inv: OP2Inventory, header_index: int, ekey_index: int = None) -> pd.DataFrame:
    """
    Decode a CBUSH OES1X1 block.

    Returns a DataFrame with columns ``EID, EX, EY, EZ, ETX, ETY, ETZ``
    where EX/EY/EZ are translational deformations and ETX/ETY/ETZ are
    rotational deformations.
    """
    payload, data_idx, all_recs = load_data_bytes(
        inv, ekey_index if ekey_index is not None else header_index
    )
    df = _decode_oes_cbush_payload(payload, endian=inv.endian)
    df.attrs["header_record"] = header_index
    df.attrs["data_record"] = data_idx
    df.attrs["all_data_records"] = all_recs
    df.attrs["element_type"] = 102
    return df
