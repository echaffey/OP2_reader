from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from .fortran_io import RecordInfo  # reuse the existing definition


@dataclass
class OP2Record:
    info: RecordInfo
    data: bytes
    ascii_hint: str  # printable preview from payload head
    probable_table_name: Optional[str]


@dataclass
class OP2Inventory:
    path: Path
    endian: str
    marker_size: int
    records: List[OP2Record]


@dataclass
class SubcaseMeta:
    """Metadata extracted from a result table header (IDENT block)."""

    subcase: int
    """Subcase ID (ISUBCASE)."""

    acode: int
    """Analysis approach code.
    1=statics, 2=normal modes, 3=differential stiffness,
    4=differential stiffness buckling, 5=frequency response,
    6=piecewise linear transient, 7=pre-buckling, 8=post-buckling,
    9=nonlinear statics, 10=nonlinear buckling, 11=geometric nonlinear."""

    tcode: int
    """Table code identifying the result type (1=OES, 10=OEF, 17=OUG, …)."""

    table_name: str
    """8-character OP2 table token (e.g. ``'OES1X1'``)."""

    title: str = ""
    """Analysis TITLE string from case control."""

    subtitle: str = ""
    """SUBTITLE string from case control."""

    label: str = ""
    """LABEL string from case control."""
