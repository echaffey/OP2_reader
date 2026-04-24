import struct as _struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from .fortran_io import RecordInfo  # reuse the existing definition

# Records whose payload exceeds this threshold are stored truncated in .data
# during the inventory scan.  Full payloads are fetched on demand via
# OP2Inventory.get_record_data().  64 KB is large enough for every
# header/control/EKEY/LAMA/PVT record; only result data blocks (OES, OEF,
# OUG, …) and large geometry tables (EQEXIN for big models) are truncated.
_HEAD_BYTES: int = 65_536


@dataclass
class OP2Record:
    info: RecordInfo
    data: bytes  # first min(length, _HEAD_BYTES) bytes of the payload
    ascii_hint: str  # printable preview from payload head
    probable_table_name: Optional[str]


@dataclass
class OP2Inventory:
    path: Path
    endian: str
    marker_size: int
    records: List[OP2Record]

    def get_record_data(self, index: int) -> bytes:
        """Return the *full* payload for record *index*.

        If the record was stored truncated during the inventory scan (i.e.
        ``len(rec.data) < rec.info.length``), the file is re-opened and the
        payload is read from the stored offset.  For small records the cached
        bytes in ``rec.data`` are returned directly.
        """
        rec = self.records[index]
        if len(rec.data) == rec.info.length:
            return rec.data  # already have the full payload
        # Payload was truncated — seek to the record in the file.
        marker_fmt = self.endian + ("I" if self.marker_size == 4 else "q")
        with open(self.path, "rb") as fp:
            fp.seek(rec.info.offset + self.marker_size)  # skip leading length marker
            return fp.read(rec.info.length)


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
