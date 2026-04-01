# op2_native/decoders/ogpwg.py
"""
Decoder for the OGPWG (Grid Point Weight Generator) table.

The OGPWG table reports the total model mass, centre of gravity, and
mass/inertia matrices computed about a reference point.

OP2 record layout (SORT1, real, 78 words = 312 bytes)
------------------------------------------------------
Words  0-35  (36 floats) : S  -- 6x6 mass/inertia matrix in the
                                  reference coordinate system (row-major)
Words 36-44  (9  floats) : IQ -- 3x3 direction-cosine matrix of the
                                  principal inertia axes
Words 45-53  (9  floats) : (unused padding / second block)
Words 54-56  (3  floats) : MO -- total mass diagonal [mass, mass, mass]
                                  (words 45,49,53 in the file)
Words 57-65  (9  floats) : S1 -- 6x6 upper-left 3x3 portion (mass in
                                  principal frame) -- actually 3x3
Words 66-68  (3  floats) : CG -- centre of gravity [X_cg, Y_cg, Z_cg]
                                  (in the reference coord system)
Words 69-77  (9  floats) : Q  -- 3x3 direction cosines of principal axes

Practical output
----------------
``decode_ogpwg`` returns a dict with:

  mass         -- scalar total mass (float)
  cg           -- array [X, Y, Z] centre of gravity in reference CS
  S            -- 6x6 ndarray mass matrix
  IQ           -- 3x3 ndarray direction cosines (principal axes in ref CS)
  S1           -- 3x3 ndarray mass matrix in principal frame

and a convenience DataFrame (``summary``) with human-readable rows.
"""
from __future__ import annotations

import struct
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..models import OP2Inventory


# Expected size of the data record in words
_OGPWG_WORDS = 78


def _find_ogpwg_data(inv: OP2Inventory) -> Optional[bytes]:
    """
    Locate the OGPWG data record (312-byte block) in the inventory.

    The OGPWG header is an 8-byte table-name record (``b'OGPWG   '``).
    The actual weight data follows a few records later as a 312-byte record.
    """
    for i, rec in enumerate(inv.records):
        if rec.info.length == 8 and b"OGPWG" in rec.data:
            # look ahead for the 312-byte data record
            for j in range(i + 1, min(len(inv.records), i + 20)):
                r = inv.records[j]
                if r.info.length == 312:
                    return r.data
    return None


def decode_ogpwg(inv: OP2Inventory) -> Optional[Dict]:
    """
    Decode the OGPWG grid-point weight generator data.

    Returns
    -------
    dict or None
        ``None`` if no OGPWG table is present.  Otherwise a dict with:

        ``mass`` : float
            Total model mass.
        ``cg`` : list[float, float, float]
            Centre of gravity [X, Y, Z] in the reference coordinate system.
        ``S`` : list[list[float]]
            6x6 mass/inertia matrix (row-major) in the reference CS.
        ``IQ`` : list[list[float]]
            3x3 direction-cosine matrix of the principal inertia axes.
        ``S1`` : list[list[float]]
            3x3 mass matrix in the principal inertia frame.
        ``Q`` : list[list[float]]
            3x3 principal-axis direction cosines.
        ``summary`` : pd.DataFrame
            One-row DataFrame with ``mass``, ``CG_X``, ``CG_Y``, ``CG_Z``,
            ``IXX``, ``IYY``, ``IZZ``, ``IXY``, ``IXZ``, ``IYZ``.
    """
    data = _find_ogpwg_data(inv)
    if data is None:
        return None

    n_words = len(data) // 4
    if n_words < _OGPWG_WORDS:
        return None

    floats = list(struct.unpack(f"<{_OGPWG_WORDS}f", data[: _OGPWG_WORDS * 4]))

    # S: 6x6 matrix (words 0-35)
    S_flat = floats[0:36]
    S = [S_flat[r * 6 : (r + 1) * 6] for r in range(6)]

    # IQ: 3x3 direction cosines (words 36-44)
    IQ_flat = floats[36:45]
    IQ = [IQ_flat[r * 3 : (r + 1) * 3] for r in range(3)]

    # Words 45-53 appear to be a second (repeated) 3x3 block in some variants;
    # words 54-56 are unused in this layout.

    # S1: 3x3 mass matrix in principal frame (words 57-65)
    S1_flat = floats[57:66]
    S1 = [S1_flat[r * 3 : (r + 1) * 3] for r in range(3)]

    # CG: centre of gravity (words 66-68)
    cg = floats[66:69]

    # Q: principal axis direction cosines (words 69-77)
    Q_flat = floats[69:78]
    Q = [Q_flat[r * 3 : (r + 1) * 3] for r in range(3)]

    # Total mass = S[0][0] (the (1,1) element of the 6x6 mass matrix is the
    # translational mass, same in all 3 translational DOFs for a lumped model)
    mass = float(S[0][0])

    # Inertia components (off-diagonal of the lower-right 3x3 of S)
    # S[3][3]=Ixx, S[4][4]=Iyy, S[5][5]=Izz
    # S[3][4]=S[4][3]=-Ixy, S[3][5]=S[5][3]=-Ixz, S[4][5]=S[5][4]=-Iyz
    ixx = S[3][3]
    iyy = S[4][4]
    izz = S[5][5]
    ixy = -S[3][4]
    ixz = -S[3][5]
    iyz = -S[4][5]

    summary = pd.DataFrame(
        [
            {
                "mass": mass,
                "CG_X": cg[0],
                "CG_Y": cg[1],
                "CG_Z": cg[2],
                "IXX": ixx,
                "IYY": iyy,
                "IZZ": izz,
                "IXY": ixy,
                "IXZ": ixz,
                "IYZ": iyz,
            }
        ]
    )

    return {
        "mass": mass,
        "cg": cg,
        "S": S,
        "IQ": IQ,
        "S1": S1,
        "Q": Q,
        "summary": summary,
    }
