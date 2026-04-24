# op2_native/fortran_io.py
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, Optional, Tuple, Union

import io
import struct


@dataclass(frozen=True)
class RecordInfo:
    """Metadata for a single Fortran unformatted record."""

    index: int  # 0-based record index in file
    offset: int  # file offset where the record length field begins
    length: int  # byte length of the record payload (not counting the markers)


class FortranUnformattedReader:
    """
    Minimal reader for compiler-style 'unformatted' binary:
      [marker_len][payload bytes...][marker_len]
    where marker_len is either 4 or 8 bytes, little or big endian.
    This is the foundation for reading Nastran OP2 files.
    """

    def __init__(self, fp: BinaryIO):
        self.fp = fp
        self.endian: str = "<"  # '<' little, '>' big
        self.marker_size: int = 4
        self._detected = False

    @classmethod
    def open(cls, path: Union[str, Path]) -> "FortranUnformattedReader":
        fp = open(path, "rb")
        return cls(fp)

    def close(self) -> None:
        try:
            self.fp.close()
        except Exception:
            pass

    # --- detection ---------------------------------------------------------
    def _try_probe(self, endian: str, marker_size: int) -> Optional[int]:
        """Return first record length if trailer matches; else None. Keeps file position at 0."""
        self.fp.seek(0, io.SEEK_SET)
        # 8-byte markers: some runtimes write a signed int64; strip the high
        # continuation bit (0x80000000_00000000) used by gfortran subrecords.
        len_fmt = endian + ("I" if marker_size == 4 else "q")
        head = self.fp.read(marker_size)
        if len(head) != marker_size:
            return None
        (raw,) = struct.unpack(len_fmt, head)
        nbytes = int(raw) & 0x7FFF_FFFF_FFFF_FFFF if marker_size == 8 else int(raw)
        # nbytes must be positive and "reasonable" (avoid gigabytes at rec #0)
        if nbytes <= 0 or nbytes > (1 << 30):
            return None
        # seek payload + trailer and verify symmetry
        self.fp.seek(nbytes, io.SEEK_CUR)
        tail = self.fp.read(marker_size)
        if len(tail) != marker_size:
            return None
        (raw2,) = struct.unpack(len_fmt, tail)
        nbytes2 = int(raw2) & 0x7FFF_FFFF_FFFF_FFFF if marker_size == 8 else int(raw2)
        if nbytes != nbytes2:
            return None
        return nbytes

    def detect(self) -> Tuple[str, int]:
        """Detect endianness and marker size. Idempotent."""
        if self._detected:
            return (self.endian, self.marker_size)
        for endian in ("<", ">"):
            for msize in (4, 8):
                n = self._try_probe(endian, msize)
                if n is not None:
                    self.endian, self.marker_size = endian, msize
                    self._detected = True
                    self.fp.seek(0, io.SEEK_SET)
                    return (self.endian, self.marker_size)
        raise ValueError(
            "Could not detect Fortran unformatted record markers (endianness/size)."
        )

    # --- iteration ---------------------------------------------------------
    def __iter__(self) -> Iterator[Tuple[RecordInfo, bytes]]:
        """Iterate all records as (RecordInfo, payload_bytes)."""
        self.detect()
        length_fmt = self.endian + ("I" if self.marker_size == 4 else "q")
        _8byte = self.marker_size == 8
        rec_index = 0
        while True:
            offset = self.fp.tell()
            head = self.fp.read(self.marker_size)
            if not head:
                break  # EOF
            if len(head) != self.marker_size:
                raise IOError(f"Truncated marker at offset {offset}")
            (raw,) = struct.unpack(length_fmt, head)
            nbytes = (int(raw) & 0x7FFF_FFFF_FFFF_FFFF) if _8byte else int(raw)
            payload = self.fp.read(nbytes)
            if len(payload) != nbytes:
                raise IOError(
                    f"Truncated payload at record {rec_index}, offset {offset}"
                )
            tail = self.fp.read(self.marker_size)
            if len(tail) != self.marker_size:
                raise IOError(f"Missing trailer at record {rec_index}, offset {offset}")
            (raw2,) = struct.unpack(length_fmt, tail)
            nbytes2 = (int(raw2) & 0x7FFF_FFFF_FFFF_FFFF) if _8byte else int(raw2)
            if nbytes != nbytes2:
                raise IOError(
                    f"Marker mismatch at record {rec_index}, offset {offset}: {nbytes} != {nbytes2}"
                )
            yield RecordInfo(rec_index, offset, nbytes), payload
            rec_index += 1

    # --- context manager support -------------------------------------------
    def __enter__(self) -> "FortranUnformattedReader":
        # Ensure detection runs once you enter, so callers can immediately read .endian/.marker_size
        self.detect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self.close()
        except Exception:
            pass
