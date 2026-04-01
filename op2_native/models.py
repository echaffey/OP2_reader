from __future__ import annotations
from dataclasses import dataclass
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
