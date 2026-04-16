from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import re

from .fortran_io import FortranUnformattedReader, RecordInfo
from .models import OP2Record, OP2Inventory, _HEAD_BYTES

ASCII_HINT_RE = re.compile(rb"[A-Z0-9_]{3,12}")


class OP2Reader:
    """
    Phase 1: non-decoding 'peeker'
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def _ascii_hint(self, payload: bytes, max_bytes: int = 96) -> str:
        raw = payload[:max_bytes]
        return "".join(chr(b) if 32 <= b <= 126 else "." for b in raw)

    def _probable_table(self, payload: bytes) -> Optional[str]:
        head = payload[:128]
        cands = ASCII_HINT_RE.findall(head)
        tokens = [c.decode("ascii", "ignore") for c in cands if 4 <= len(c) <= 8]
        if not tokens:
            return None

        def score(tok: str) -> int:
            s = 0
            if tok.startswith(("O", "G", "C", "E")):
                s += 2
            if tok in (
                "OUG",
                "OES",
                "OEF",
                "OGP",
                "OQG",
                "OPG",
                "OGS",
                "OGPWG",
                "GEOM1",
                "GEOM2",
                "CASECC",
            ):
                s += 3
            s += min(len(tok), 8)
            return s

        tokens.sort(key=score, reverse=True)
        return tokens[0]

    def peek_inventory(self, limit_records: Optional[int] = None) -> OP2Inventory:
        recs: List[OP2Record] = []
        with FortranUnformattedReader.open(
            self.path
        ) as reader:  # context mgr supported
            endian, msize = reader.detect()
            for info, data in reader:
                ascii_hint = self._ascii_hint(data)
                name = self._probable_table(data)
                # Truncate large payloads — only keep the head bytes needed for
                # heuristics and headers.  Full data is fetched on demand via
                # OP2Inventory.get_record_data().
                stored = data if len(data) <= _HEAD_BYTES else data[:_HEAD_BYTES]
                recs.append(
                    OP2Record(
                        info=info,
                        data=stored,
                        ascii_hint=ascii_hint,
                        probable_table_name=name,
                    )
                )
                if limit_records is not None and len(recs) >= limit_records:
                    break
        return OP2Inventory(
            path=self.path, endian=endian, marker_size=msize, records=recs
        )


def inventory_to_rows(inv: OP2Inventory) -> List[dict]:
    rows = []
    for r in inv.records:
        rows.append(
            {
                "rec_index": r.info.index,
                "file_offset": r.info.offset,
                "length_bytes": r.info.length,
                "probable_table": r.probable_table_name or "",
                "ascii_hint": r.ascii_hint,
            }
        )
    return rows
