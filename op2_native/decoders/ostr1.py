# op2_native/decoders/ostr1.py
"""
Decoder for OSTR1 shell strain data blocks.

The physical layout of an OSTR1X1 record is byte-for-byte identical to
OES1X1: each element has a 17-word packet (16 floats + 1 packed ID).
The only difference is semantic -- the 16 floats represent strain
components rather than stress components.

We re-use the OES1X1 shell decoder and rename the output columns to
use strain-specific names (same layout as stress but with 'E' prefix):
  FD1, EX1, EY1, EXY1, EANG1, EMAJOR1, EMINOR1, EVM1,
  FD2, EX2, EY2, EXY2, EANG2, EMAJOR2, EMINOR2, EVM2
"""
from __future__ import annotations

import pandas as pd

from ..models import OP2Inventory
from .oes1x1_shell import decode_oes1x1_shell

# Strain column names matching the 16 data words per element
SHELL_STRAIN_COLS = [
    "EID",
    # fiber 1 (bottom / Z1)
    "FD1",
    "EX1",
    "EY1",
    "EXY1",
    "EANG1",
    "EMAX_PRIN1",
    "EMIN_PRIN1",
    "EVON_MISES1",
    # fiber 2 (top / Z2)
    "FD2",
    "EX2",
    "EY2",
    "EXY2",
    "EANG2",
    "EMAX_PRIN2",
    "EMIN_PRIN2",
    "EVON_MISES2",
]

# Mapping from stress names -> strain names (EID and FD passthrough; rest renamed)
_STRESS_TO_STRAIN = {
    "FD1": "FD1",
    "SX1": "EX1",
    "SY1": "EY1",
    "TXY1": "EXY1",
    "ANG1": "EANG1",
    "MAX_PRIN1": "EMAX_PRIN1",
    "MIN_PRIN1": "EMIN_PRIN1",
    "VON_MISES1": "EVON_MISES1",
    "FD2": "FD2",
    "SX2": "EX2",
    "SY2": "EY2",
    "TXY2": "EXY2",
    "ANG2": "EANG2",
    "MAX_PRIN2": "EMAX_PRIN2",
    "MIN_PRIN2": "EMIN_PRIN2",
    "VON_MISES2": "EVON_MISES2",
}


def decode_ostr1(inv: OP2Inventory, header_index: int, ekey_index: int = None) -> pd.DataFrame:
    """
    Decode an OSTR1 shell strain block.

    Returns
    -------
    DataFrame with columns
    ``EID, FD1, EX1, EY1, EXY1, EANG1, EMAX_PRIN1, EMIN_PRIN1, EVON_MISES1,
    FD2, EX2, EY2, EXY2, EANG2, EMAX_PRIN2, EMIN_PRIN2, EVON_MISES2``.
    """
    df = decode_oes1x1_shell(inv, header_index, ekey_index=ekey_index)
    # Rename using the stress->strain mapping; fall back gracefully for
    # any unexpected column names (e.g. S1..S16 from the generic fallback)
    rename = {}
    for col in df.columns:
        if col in _STRESS_TO_STRAIN:
            rename[col] = _STRESS_TO_STRAIN[col]
        elif col.startswith("S") and col[1:].isdigit():
            rename[col] = "E" + col[1:]
    if rename:
        df = df.rename(columns=rename)
    return df
