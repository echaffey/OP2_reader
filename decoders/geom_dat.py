# op2_native/decoders/geom_dat.py
"""
Parser for Nastran bulk data (.dat / .bdf) geometry.

Reads the BULK DATA section of a Nastran input file and extracts:

    * GRID cards  → node coordinates (GID, X, Y, Z, CP, CD)
    * CTETRA cards → solid element connectivity (EID, PID, G1-G4 corner nodes)
    * CHEXA cards  → hex element connectivity (EID, PID, G1-G8 corner nodes)
    * CPENTA cards → penta element connectivity (EID, PID, G1-G6 corner nodes)
    * CBAR cards   → bar element connectivity (EID, PID, GA, GB)
    * CBEAM cards  → beam element connectivity (EID, PID, GA, GB)
    * CBUSH cards  → bush element connectivity (EID, PID, GA, GB)
    * CROD cards   → rod element connectivity (EID, PID, GA, GB)

Cards can be in either:
    * Small-field format (8-char fields, 10 fields per line)
    * Free-field format (comma-separated)

Continuations ('+' at col 72, or '+' at start of next line) are handled.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class GeomData:
    """
    Geometry parsed from a Nastran bulk data file.

    Attributes
    ----------
    grids : pd.DataFrame
        One row per GRID card.  Columns: GID (int), X (float), Y (float),
        Z (float), CP (int, input coordinate system), CD (int, output CS).
    elements : dict
        ``{etype_str: DataFrame}`` — one entry per element type found.
        Element types: ``'CTETRA'``, ``'CHEXA'``, ``'CPENTA'``,
        ``'CBAR'``, ``'CBEAM'``, ``'CBUSH'``, ``'CROD'``.
        Each DataFrame has at minimum columns ``EID``, ``PID``, ``G1``,
        ``G2`` (line elements also have ``GA``/``GB`` aliases).
    source_path : Path
        The file this was parsed from.
    """

    grids: pd.DataFrame
    elements: Dict[str, pd.DataFrame]
    source_path: Path


# ---------------------------------------------------------------------------
# Nasran field parsing helpers
# ---------------------------------------------------------------------------


def _is_free_field(line: str) -> bool:
    """Return True if the card uses comma-separated fields."""
    return "," in line[:72]


def _split_fields(line: str) -> List[str]:
    """
    Return a list of up to 8 data field strings from one Nastran card line.

    Handles both small-field (8-char) and free-field (comma-separated) formats.
    Field 0 (card name) and field 9 (continuation marker) are excluded.
    """
    # Free-field (comma-separated)
    if _is_free_field(line):
        parts = line.rstrip("\n").split(",")
        # field 0 is the card name, data is parts 1..8
        return [p.strip() for p in parts[1:9]]

    # Small-field (8-char per field, fields 1-8 occupy cols 8-71)
    fields = []
    for i in range(1, 9):
        start = i * 8
        end = start + 8
        if start >= len(line):
            fields.append("")
        else:
            fields.append(line[start:end].strip())
    return fields


def _has_continuation(line: str) -> bool:
    """Return True if this card line has a continuation onto the next line."""
    s = line.rstrip("\n")
    # Explicit marker in col 72 (small-field)
    if len(s) >= 73 and s[72].strip():
        return True
    # Marker at end of truncated line (some writers truncate trailing spaces)
    stripped = s.rstrip()
    return stripped.endswith("+") or stripped.endswith("*")


def _to_int(s: str, default: int = 0) -> int:
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        return default


def _to_float(s: str, default: float = 0.0) -> float:
    if not s:
        return default
    # Nastran allows 'D' exponent notation
    s = s.replace("D", "E").replace("d", "e")
    try:
        return float(s)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

# Element types that have two end-nodes (line elements)
_LINE_ETYPES = frozenset({"CBAR", "CBEAM", "CBUSH", "CROD", "CONROD"})

# Solid element node counts (corner nodes only)
_SOLID_CORNERS = {
    "CTETRA": 4,
    "CPENTA": 6,
    "CHEXA": 8,
    "CPYRAM": 5,
}


def parse_dat(path: Union[str, Path]) -> GeomData:
    """
    Parse grid coordinates and element connectivity from a Nastran .dat/.bdf.

    Parameters
    ----------
    path : str or Path
        Path to the Nastran input file.

    Returns
    -------
    GeomData
        Contains ``grids`` DataFrame and ``elements`` dict.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    grid_rows: List[Tuple] = []  # (GID, X, Y, Z, CP, CD)
    elem_rows: Dict[str, List[Tuple]] = {}  # etype → list of tuples

    with open(path, "r", errors="replace") as fh:
        raw_lines = fh.readlines()

    in_bulk = False
    skip_next = 0  # lines to skip (consumed continuations)
    n = len(raw_lines)
    i = 0

    while i < n:
        if skip_next:
            skip_next -= 1
            i += 1
            continue

        line = raw_lines[i].rstrip("\n")

        # ---- Section markers ------------------------------------------------
        upper = line.strip().upper()
        if upper.startswith("BEGIN BULK") or upper == "BEGIN":
            in_bulk = True
            i += 1
            continue
        if upper.startswith("ENDDATA") or upper == "END":
            break
        if not in_bulk:
            i += 1
            continue

        # ---- Skip comments, blanks, continuation lines ----------------------
        if not line.strip() or line[0] in ("$", "\n", "\r"):
            i += 1
            continue
        # Continuation cards are consumed by their parent
        if line[0] in ("+", "*") or (line[0] == " " and line[1:8].strip() == ""):
            i += 1
            continue

        # ---- Card name -------------------------------------------------------
        card = line[0:8].strip().upper()

        # ---- GRID ------------------------------------------------------------
        if card == "GRID":
            f = _split_fields(line)
            gid = _to_int(f[0])
            cp = _to_int(f[1], 0)
            x = _to_float(f[2])
            y = _to_float(f[3])
            z = _to_float(f[4])
            cd = _to_int(f[5], 0) if len(f) > 5 else 0
            if gid > 0:
                grid_rows.append((gid, x, y, z, cp, cd))

        # ---- Solid elements (possibly multi-line) ----------------------------
        elif card in _SOLID_CORNERS:
            n_corners = _SOLID_CORNERS[card]
            f = _split_fields(line)
            eid = _to_int(f[0])
            pid = _to_int(f[1])
            nodes = [_to_int(f[j]) for j in range(2, len(f))]

            # Consume continuations if we don't have enough corner nodes yet
            while len(nodes) < n_corners and _has_continuation(
                raw_lines[i].rstrip("\n")
            ):
                i += 1
                if i >= n:
                    break
                cont = raw_lines[i].rstrip("\n")
                cf = _split_fields(cont)
                nodes.extend(_to_int(v) for v in cf if v)

            if eid > 0:
                corner_nodes = nodes[:n_corners]
                # Pad with 0 if fewer nodes than expected
                while len(corner_nodes) < n_corners:
                    corner_nodes.append(0)
                row_tuple = (eid, pid) + tuple(corner_nodes)
                if card not in elem_rows:
                    elem_rows[card] = []
                elem_rows[card].append(row_tuple)

            # Skip any remaining continuation lines for this card
            while _has_continuation(raw_lines[i].rstrip("\n")):
                i += 1
                if i >= n:
                    break
                # check if *this* continuation has another continuation
                if not _has_continuation(raw_lines[i].rstrip("\n")):
                    break

        # ---- Line elements (CBAR, CBEAM, CBUSH, CROD, CONROD) ----------------
        elif card in _LINE_ETYPES:
            f = _split_fields(line)
            eid = _to_int(f[0])
            pid = _to_int(f[1])
            ga = _to_int(f[2])
            gb = _to_int(f[3])
            if eid > 0:
                if card not in elem_rows:
                    elem_rows[card] = []
                elem_rows[card].append((eid, pid, ga, gb))

        i += 1

    # --------------------------------------------------------------------------
    # Build DataFrames
    # --------------------------------------------------------------------------
    grids_df = pd.DataFrame(
        grid_rows,
        columns=["GID", "X", "Y", "Z", "CP", "CD"],
    )
    grids_df["GID"] = grids_df["GID"].astype("int32")
    grids_df["CP"] = grids_df["CP"].astype("int16")
    grids_df["CD"] = grids_df["CD"].astype("int16")
    grids_df = grids_df.sort_values("GID").reset_index(drop=True)

    elements: Dict[str, pd.DataFrame] = {}

    for etype, rows in elem_rows.items():
        if etype in _SOLID_CORNERS:
            nc = _SOLID_CORNERS[etype]
            cols = ["EID", "PID"] + [f"G{j+1}" for j in range(nc)]
        else:
            cols = ["EID", "PID", "GA", "GB"]
        df = pd.DataFrame(rows, columns=cols)
        df["EID"] = df["EID"].astype("int32")
        df["PID"] = df["PID"].astype("int16")
        df = df.sort_values("EID").reset_index(drop=True)
        elements[etype] = df

    return GeomData(grids=grids_df, elements=elements, source_path=path)
