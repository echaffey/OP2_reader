# op2_native/reader.py
"""
Central OP2 class.  Open a file once; access any result type as a property
or method.  All result methods return ``{subcase_id: DataFrame}``.
"""
from __future__ import annotations

import struct
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .fortran_io import FortranUnformattedReader
from .models import OP2Inventory, OP2Record, SubcaseMeta
from .op2_reader import OP2Reader

# decoder imports
from .decoders.ougv1 import find_ougv1_headers, classify_ougv1_headers, decode_ougv1
from .decoders.oes_search import (
    find_oes_tables,
    find_oef_tables,
    classify_ostr_headers,
    classify_ostr_el_headers,
    classify_oes_headers,
)
from .decoders.oes1x1_shell import (
    decode_oes1x1_shell,
    decode_oes1x1_shell_corners,
    decode_oes1x1_tria3_payload,
)
from .decoders.oes_solid import decode_oes_solid
from .decoders.oes_bar import decode_oes_bar
from .decoders.oes_cbush import decode_oes_cbush
from .decoders.oef1 import decode_oef1, classify_oef_headers
from .decoders.oqg1 import (
    decode_oqg1,
    classify_oqg_headers,
    classify_oqgcf1_headers,
    classify_separation_headers,
    decode_separation,
)
from .decoders.opg1 import decode_opg1, classify_opg_headers
from .decoders.ostr1 import decode_ostr1
from .decoders.ogpwg import decode_ogpwg
from .decoders.lama import decode_lama
from .decoders.oesnlxr import (
    classify_oesnlxr_headers,
    decode_oesnlxr_cbeam,
    decode_oesnlxr_cbush,
    decode_oesnlxr_ctetra,
    decode_oesnlxr_shell,
    SHELL_NL_STRESS_COLS,
    SHELL_NL_STRAIN_COLS,
)
from .decoders.geom_dat import parse_dat, GeomData


class OP2:
    """
    Lightweight Nastran OP2 reader.

    Parameters
    ----------
    path : str or Path
        Path to the .op2 file.
    geometry : bool or str or Path, optional
        Controls whether geometry (grid coordinates + element connectivity)
        is loaded from the companion Nastran bulk-data file.

        ``False`` (default)
            No geometry is loaded.
        ``True``
            Auto-detect the companion ``.dat`` / ``.bdf`` file: looks for a
            file with the same stem as *path* and extensions ``.dat``,
            ``.bdf``, ``.nas`` (checked in that order).
        str or Path
            Explicit path to a ``.dat`` / ``.bdf`` file.

        When geometry is available, use :meth:`grid_coordinates` and
        :meth:`element_connectivity` to access the data.

    Examples
    --------
    >>> op2 = OP2("model.op2")
    >>> disp = op2.displacements()   # {subcase_id: DataFrame}
    >>> stress = op2.stresses()      # {subcase_id: DataFrame}

    >>> op2 = OP2("model.op2", geometry=True)   # auto-find model.dat
    >>> grids = op2.grid_coordinates()           # DataFrame(GID, X, Y, Z)
    >>> conn  = op2.element_connectivity()       # {'CTETRA': df, ...}
    """

    def __init__(
        self,
        path: str | Path,
        geometry: Union[bool, str, Path] = False,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._inv: Optional[OP2Inventory] = None
        # Result cache: keyed by method name so each decode runs at most once.
        self._cache: Dict[str, object] = {}
        # Geometry: resolve the companion dat/bdf path (or None)
        self._geom_path: Optional[Path] = self._resolve_geom_path(geometry)
        self._geom: Optional[GeomData] = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    _GEOM_EXTENSIONS = (".dat", ".bdf", ".nas")

    def _resolve_geom_path(self, geometry) -> Optional[Path]:
        """Return the resolved bulk-data path, or None if geometry=False."""
        if geometry is False or geometry is None:
            return None
        if geometry is True:
            stem = self.path.stem
            parent = self.path.parent
            for ext in self._GEOM_EXTENSIONS:
                candidate = parent / (stem + ext)
                if candidate.exists():
                    return candidate
            warnings.warn(
                f"geometry=True but no .dat/.bdf/.nas found alongside {self.path.name}."
                " Set geometry=False or pass an explicit path.",
                UserWarning,
            )
            return None
        p = Path(geometry)
        if not p.exists():
            raise FileNotFoundError(f"Geometry file not found: {p}")
        return p

    @property
    def _geometry(self) -> GeomData:
        """Lazy-load and cache the parsed geometry."""
        if self._geom_path is None:
            raise RuntimeError(
                "Geometry was not loaded for this OP2 instance.  "
                "Re-open with geometry=True (or pass the .dat path)."
            )
        if self._geom is None:
            self._geom = parse_dat(self._geom_path)
        return self._geom

    def clear_cache(self) -> None:
        """Discard all cached results (e.g. after swapping the file)."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @property
    def inventory(self) -> OP2Inventory:
        """Lazily scan the file and cache the record inventory."""
        if self._inv is None:
            self._inv = OP2Reader(self.path).peek_inventory()
        return self._inv

    def _first_hit(self, tables: Dict[str, list]) -> Optional[int]:
        """Return the record index of the very first header across all token hits."""
        if not tables:
            return None
        return min(idx for hits in tables.values() for idx in hits)

    def _all_hits(self, tables: Dict[str, list]) -> list:
        """Flatten all record indices from a find_*_tables result, sorted."""
        return sorted(idx for hits in tables.values() for idx in hits)

    def _read_subcase_id(self, inv: OP2Inventory, header_index: int) -> int:
        """
        Read the actual ISUBCASE from the 28-byte IDENT record that follows
        an 8-byte table-name record.

        The OP2 header sequence is:
          rec+0  (8 bytes)  table name
          rec+1  (4 bytes)  -1
          rec+2  (4 bytes)  7  (word count)
          rec+3  (28 bytes) first IDENT: [ACODE, TCODE, ?, NUMWDE, NUMWDE, LSDVMN, ISUBCASE]

        ISUBCASE is reliably at word index 6 across all table types.
        """
        for i in range(header_index + 1, min(len(inv.records), header_index + 10)):
            rec = inv.records[i]
            if rec.info.length == 28:
                words = struct.unpack(f"{inv.endian}7i", rec.data)
                isubcase = words[6]
                return max(1, isubcase)
        return 1  # fallback

    @staticmethod
    def _decode_text_record(data: bytes) -> str:
        """Decode a Nastran text record (Hollerith 4-byte words) to a stripped string."""
        try:
            return data.decode("latin-1", errors="replace").strip()
        except Exception:
            return ""

    @staticmethod
    def _is_text_record(data: bytes, min_printable: float = 0.70) -> bool:
        """Return True if *data* is mostly printable ASCII (space through ~)."""
        if not data:
            return False
        printable = sum(0x20 <= b <= 0x7E for b in data)
        return printable / len(data) >= min_printable

    def _read_header_meta(self, inv: OP2Inventory, header_index: int) -> SubcaseMeta:
        """
        Extract ACODE, TCODE, ISUBCASE, and optional title/subtitle/label
        strings from the records immediately following an 8-byte table-name
        record.

        OP2 header sequence (0-based from the table-name record):
          +0  (8 bytes)  table name (padded with spaces)
          +1  (4 bytes)  -1 marker
          +2  (4 bytes)  7  (word count)
          +3  (28 bytes) IDENT: [ACODE, TCODE, ?, ?, ?, LSDVMN, ISUBCASE]
          +4  (4 bytes)  -1 marker
          ...
          subsequent records of 128 bytes each: TITLE, SUBTITLE, LABEL
        """
        table_name = ""
        name_rec = inv.records[header_index]
        if name_rec.info.length == 8:
            table_name = name_rec.data.decode("latin-1", errors="replace").strip()

        acode = 0
        tcode = 0
        isubcase = 1

        # Parse the IDENT record
        for i in range(header_index + 1, min(len(inv.records), header_index + 12)):
            rec = inv.records[i]
            if rec.info.length == 28:
                try:
                    words = struct.unpack(f"{inv.endian}7i", rec.data)
                    acode = words[0]
                    tcode = words[1]
                    isubcase = max(1, words[6])
                except struct.error:
                    pass
                break

        # Scan for title / subtitle / label text records (128 bytes each)
        # They appear within the first ~25 records after the table name.
        text_records: List[str] = []
        for i in range(header_index + 1, min(len(inv.records), header_index + 25)):
            rec = inv.records[i]
            if rec.info.length == 8:
                break  # next table starts
            if rec.info.length == 128 and self._is_text_record(rec.data):
                text_records.append(self._decode_text_record(rec.data))
            if len(text_records) == 3:
                break

        title = text_records[0] if len(text_records) > 0 else ""
        subtitle = text_records[1] if len(text_records) > 1 else ""
        label = text_records[2] if len(text_records) > 2 else ""

        return SubcaseMeta(
            subcase=isubcase,
            acode=acode,
            tcode=tcode,
            table_name=table_name,
            title=title,
            subtitle=subtitle,
            label=label,
        )

    def _decode_all(
        self,
        headers: list,
        decode_fn: Callable,
        label: str,
    ) -> Dict[int, pd.DataFrame]:
        """
        Decode every header block with *decode_fn*, keying results by the
        real ISUBCASE read from the IDENT record.  Multiple blocks with the
        same subcase ID are concatenated.

        Items in *headers* may be:
          - plain int   → just a header record index
          - 2-tuple     → (header_idx, ekey_idx)
          - 3-tuple     → (header_idx, ekey_idx, sc_offset)

        ``sc_offset`` is added to the base ISUBCASE so that multi-subcase
        tables that omit a new IDENT record for later subcases are handled
        correctly.
        """
        inv = self.inventory
        result: Dict[int, pd.DataFrame] = {}
        for item in headers:
            sc_offset = 0
            if isinstance(item, tuple):
                if len(item) == 3:
                    hdr, ekey_idx, sc_offset = item
                else:
                    hdr, ekey_idx = item
            else:
                hdr, ekey_idx = item, None
            sc = self._read_subcase_id(inv, hdr) + sc_offset
            try:
                df = (
                    decode_fn(inv, hdr)
                    if ekey_idx is None
                    else decode_fn(inv, hdr, ekey_idx)
                )
            except Exception as exc:
                warnings.warn(f"{label} header rec {hdr}: {exc}", RuntimeWarning)
                continue
            if sc in result:
                result[sc] = pd.concat(
                    [result[sc], df.dropna(axis=1, how="all")],
                    ignore_index=True,
                )
            else:
                result[sc] = df
        return result

    def _cached(self, key: str, compute):
        """Return cached result for *key*, computing it via *compute()* on first call."""
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    # ------------------------------------------------------------------
    # Public result accessors
    # ------------------------------------------------------------------

    def displacements(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OUGV1 displacement/velocity/acceleration blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``GRID, TX, TY, TZ, RX, RY, RZ``.
        """
        return self._cached(
            "displacements",
            lambda: self._decode_all(
                classify_ougv1_headers(self.inventory), decode_ougv1, "OUGV1"
            ),
        )

    def stresses(
        self,
        location: str = "max",
    ) -> Dict[int, pd.DataFrame]:
        """
        Decode all OES1X1 **shell** stress blocks.

        Parameters
        ----------
        location : {"max", "centroid"}, default "max"
            Controls which stress values are reported when the OP2 contains
            corner-node output (``STRESS(CORNER)``, NUMWDE=87).

            ``"max"``
                One row per element containing the **maximum** value of each
                stress column across all four corner nodes.  This represents
                the worst-case stress anywhere in the element and is the
                default because corner stresses are generally more accurate
                than centroid extrapolations.
            ``"centroid"``
                One row per element containing the centroid (average)
                stresses as written by the solver.

            When the OP2 was written centroid-only (NUMWDE=17 or 19) this
            parameter has no effect and the centroid values are always returned.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, FD1, SX1, SY1, TXY1, ANG1, MAX_PRIN1, MIN_PRIN1, VON_MISES1,
            FD2, SX2, SY2, TXY2, ANG2, MAX_PRIN2, MIN_PRIN2, VON_MISES2``.
        """
        if location not in ("max", "centroid"):
            raise ValueError(f"location must be 'max' or 'centroid', got {location!r}")

        cache_key = f"stresses_{location}"

        def _compute():
            import struct as _struct

            inv = self.inventory
            shell_blocks = classify_oes_headers(inv)[0]
            result: Dict[int, pd.DataFrame] = {}
            for item in shell_blocks:
                hdr, ekey_idx, sc_offset = item
                sc = self._read_subcase_id(inv, hdr) + sc_offset
                # Read NUMWDE from the EKEY record to choose the decoder
                numwde = None
                if ekey_idx is not None:
                    rec_e = inv.records[ekey_idx]
                    if rec_e.info.length == 584:
                        numwde = _struct.unpack(f"{inv.endian}146i", rec_e.data)[9]
                try:
                    if numwde == 87:
                        df_full = decode_oes1x1_shell_corners(inv, hdr, ekey_idx)
                        stress_cols = [
                            c for c in df_full.columns if c not in ("EID", "GRID")
                        ]
                        if location == "max":
                            # Max of each stress column across the 4 corner nodes
                            corners = df_full[df_full["GRID"] != 0]
                            df = (
                                corners.groupby("EID", sort=False)[stress_cols]
                                .max()
                                .reset_index()
                            )
                        else:  # "centroid"
                            df = (
                                df_full[df_full["GRID"] == 0]
                                .drop(columns=["GRID"])
                                .reset_index(drop=True)
                            )
                        df.attrs.update(df_full.attrs)
                    elif numwde == 17:
                        from .decoders.oes_peek import load_data_bytes

                        payload, data_idx, _all_recs = load_data_bytes(
                            inv, ekey_idx if ekey_idx is not None else hdr
                        )
                        df = decode_oes1x1_tria3_payload(payload, endian=inv.endian)
                        df.attrs["header_record"] = hdr
                        df.attrs["data_record"] = data_idx
                        df.attrs["all_data_records"] = _all_recs
                    else:
                        df = decode_oes1x1_shell(inv, hdr, ekey_idx)
                except Exception as exc:
                    import warnings as _w

                    _w.warn(f"OES1X1-shell header rec {hdr}: {exc}", RuntimeWarning)
                    continue
                if sc in result:
                    result[sc] = pd.concat(
                        [result[sc], df.dropna(axis=1, how="all")],
                        ignore_index=True,
                    )
                else:
                    result[sc] = df
            return result

        return self._cached(cache_key, _compute)

    def stress_tensors(
        self,
        location: str = "max",
    ) -> Dict[int, pd.DataFrame]:
        """
        Extract the normal and shear stress components needed to assemble
        2D in-plane stress tensors.

        For each element and fiber layer the in-plane stress tensor is::

            ⎡ SX   TXY ⎤
            ⎣ TXY   SY ⎦

        Components are returned for both fiber layers (bottom Z1 and top Z2).

        Parameters
        ----------
        location : {"max", "centroid"}, default "max"
            Same meaning as in :meth:`stresses`.  Passed through so the
            tensor components come from the same source as the full stress
            table.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, SX1, SY1, TXY1, SX2, SY2, TXY2``.

            To build a numpy tensor for a single element in subcase 1::

                row = df[df["EID"] == eid].iloc[0]
                T1 = np.array([[row.SX1, row.TXY1],
                               [row.TXY1, row.SY1]])
                T2 = np.array([[row.SX2, row.TXY2],
                               [row.TXY2, row.SY2]])
        """
        _TENSOR_COLS = ["EID", "SX1", "SY1", "TXY1", "SX2", "SY2", "TXY2"]

        def _compute():
            raw = self.stresses(location=location)
            out: Dict[int, pd.DataFrame] = {}
            for sc, df in raw.items():
                present = [c for c in _TENSOR_COLS if c in df.columns]
                out[sc] = df[present].reset_index(drop=True)
            return out

        cache_key = f"stress_tensors_{location}"
        return self._cached(cache_key, _compute)

    def solid_stresses(
        self,
        location: str = "max",
    ) -> Dict[int, pd.DataFrame]:
        """
        Decode all OES1X1 **solid** element stress blocks
        (CHEXA / CPENTA / CTETRA).

        Parameters
        ----------
        location : {"max", "centroid"}, default "max"
            Controls which values are returned when the OP2 contains
            corner-node output (multiple GRID rows per element).

            ``"max"``
                One row per element containing the **maximum** value of each
                stress component across all corner nodes.  The ``GRID``
                column is omitted.
            ``"centroid"``
                One row per element using the centroid row (``GRID == 0``).
                The ``GRID`` column is omitted.
            ``"all"``
                All rows (centroid + every corner node) with the ``GRID``
                column included.  This is the raw decoded output.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, SX, SY, SZ, SXY, SYZ, SZX, VON_MISES``
            (plus ``GRID`` when ``location='all'``).
        """
        if location not in ("max", "centroid", "all"):
            raise ValueError(
                f"location must be 'max', 'centroid', or 'all', got {location!r}"
            )

        def _compute_all():
            return self._decode_all(
                classify_oes_headers(self.inventory)[1],
                decode_oes_solid,
                "OES1X1-solid",
            )

        all_data = self._cached("solid_stresses_all", _compute_all)

        if location == "all":
            return all_data

        cache_key = f"solid_stresses_{location}"

        def _compute():
            stress_cols = ["SX", "SY", "SZ", "SXY", "SYZ", "SZX", "VON_MISES"]
            out: Dict[int, pd.DataFrame] = {}
            for sc, df in all_data.items():
                present = [c for c in stress_cols if c in df.columns]
                if location == "max":
                    corners = df[df["GRID"] != 0]
                    out[sc] = (
                        corners.groupby("EID", sort=False)[present].max().reset_index()
                    )
                else:  # "centroid"
                    out[sc] = df[df["GRID"] == 0][["EID"] + present].reset_index(
                        drop=True
                    )
            return out

        return self._cached(cache_key, _compute)

    def solid_stress_tensors(
        self,
        location: str = "max",
    ) -> Dict[int, pd.DataFrame]:
        """
        Extract the normal and shear stress components for solid elements
        (CHEXA / CPENTA / CTETRA) needed to assemble 3D stress tensors.

        For each element the 3D symmetric stress tensor is::

            ⎡ SX   SXY  SZX ⎤
            ⎢ SXY   SY  SYZ ⎥
            ⎣ SZX  SYZ   SZ ⎦

        Parameters
        ----------
        location : {"max", "centroid"}, default "max"
            Controls which values are returned when the OP2 contains
            corner-node output (multiple GRID rows per element).

            ``"max"``
                One row per element containing the **maximum** value of each
                stress component across all corner nodes.
            ``"centroid"``
                One row per element using the centroid row (``GRID == 0``).

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, SX, SY, SZ, SXY, SYZ, SZX``.

            To build a numpy tensor for a single element in subcase 1::

                row = df[df["EID"] == eid].iloc[0]
                T = np.array([[row.SX,  row.SXY, row.SZX],
                              [row.SXY, row.SY,  row.SYZ],
                              [row.SZX, row.SYZ, row.SZ ]])
        """
        if location not in ("max", "centroid"):
            raise ValueError(f"location must be 'max' or 'centroid', got {location!r}")

        _TENSOR_COLS = ["EID", "SX", "SY", "SZ", "SXY", "SYZ", "SZX"]

        def _compute():
            stress_cols = ["SX", "SY", "SZ", "SXY", "SYZ", "SZX"]
            out: Dict[int, pd.DataFrame] = {}
            for sc, df in self.solid_stresses(location=location).items():
                present_stress = [c for c in stress_cols if c in df.columns]
                present = [c for c in _TENSOR_COLS if c in df.columns]
                out[sc] = df[["EID"] + present_stress].reset_index(drop=True)
            return out

        cache_key = f"solid_stress_tensors_{location}"
        return self._cached(cache_key, _compute)

    def bar_stresses(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OES1X1 **bar/beam** element stress blocks (CBAR / CBEAM).

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}``.  Columns depend on element type —
            see :mod:`op2_native.decoders.oes_bar`.
        """
        return self._cached(
            "bar_stresses",
            lambda: self._decode_all(
                classify_oes_headers(self.inventory)[2], decode_oes_bar, "OES1X1-bar"
            ),
        )

    def bush_stresses(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OES1X1 **CBUSH** spring element deformation blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, EX, EY, EZ, ETX, ETY, ETZ``.
        """
        return self._cached(
            "bush_stresses",
            lambda: self._decode_all(
                classify_oes_headers(self.inventory)[3],
                decode_oes_cbush,
                "OES1X1-cbush",
            ),
        )

    def element_forces(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OEF1 element force blocks for shell and bar/beam elements.

        When the OP2 was written with ``FORCE(CORNER)`` the result includes
        corner rows per element (``GRID`` column: 0=centroid, >0=corner grid ID).
        Centroid-only output and mixed models with both CQUAD4 and CTRIA3
        are handled automatically via the ``NUMWDE`` field.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}``.
        """

        def _compute():
            shells, bars, _bushes, _gaps, _others = classify_oef_headers(self.inventory)
            return self._decode_all(shells + bars, decode_oef1, "OEF1")

        return self._cached("element_forces", _compute)

    def bush_forces(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OEF1 element force blocks for CBUSH spring elements.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, FX, FY, FZ, MX, MY, MZ``.
        """

        def _compute():
            _shells, _bars, bushes, _gaps, _others = classify_oef_headers(
                self.inventory
            )
            return self._decode_all(bushes, decode_oef1, "OEF1-cbush")

        return self._cached("bush_forces", _compute)

    def gap_forces(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OEF1 element force blocks for CGAP gap elements.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, COMP_X, SHEAR_Y, SHEAR_Z, AXIAL_U, TOTAL_V, TOTAL_W, SLIP_V, SLIP_W``.
        """

        def _compute():
            _shells, _bars, _bushes, gaps, _others = classify_oef_headers(
                self.inventory
            )
            return self._decode_all(gaps, decode_oef1, "OEF1-cgap")

        return self._cached("gap_forces", _compute)

    def contact_forces(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OQGCF1 contact force blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``GRID, FX, FY, FZ, MX, MY, MZ``.
        """

        def _compute():
            return self._decode_all(
                classify_oqgcf1_headers(self.inventory), decode_oqg1, "OQGCF1"
            )

        return self._cached("contact_forces", _compute)

    def initial_separation(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OSPDSI1 initial contact separation distance blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns ``GRID, DISTANCE``.
        """

        def _compute():
            return self._decode_all(
                classify_separation_headers(self.inventory, "OSPDSI1"),
                decode_separation,
                "OSPDSI1",
            )

        return self._cached("initial_separation", _compute)

    def deformed_separation(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OSPDS1 deformed contact separation distance blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns ``GRID, DISTANCE``.
        """

        def _compute():
            return self._decode_all(
                classify_separation_headers(self.inventory, "OSPDS1"),
                decode_separation,
                "OSPDS1",
            )

        return self._cached("deformed_separation", _compute)

    def spc_forces(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OQG1 SPC/MPC constraint force blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``GRID, FX, FY, FZ, MX, MY, MZ``.
        """
        return self._cached(
            "spc_forces",
            lambda: self._decode_all(
                classify_oqg_headers(self.inventory), decode_oqg1, "OQG1"
            ),
        )

    def applied_loads(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OPG1 applied load blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``GRID, FX, FY, FZ, MX, MY, MZ``.
        """
        return self._cached(
            "applied_loads",
            lambda: self._decode_all(
                classify_opg_headers(self.inventory), decode_opg1, "OPG1"
            ),
        )

    def strains(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OSTR1 shell strain blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, FD1, EX1, EY1, EXY1, EANG1, EMAX_PRIN1, EMIN_PRIN1, EVON_MISES1,
            FD2, EX2, EY2, EXY2, EANG2, EMAX_PRIN2, EMIN_PRIN2, EVON_MISES2``.
        """
        return self._cached(
            "strains",
            lambda: self._decode_all(
                classify_ostr_headers(self.inventory)[0], decode_ostr1, "OSTR1"
            ),
        )

    def bar_strains(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OSTR1 bar/beam strain blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, GRID, SD, SXC, SXD, SXE, SXF, SMAX, SMIN, MS_T, MS_C``.
        """
        return self._cached(
            "bar_strains",
            lambda: self._decode_all(
                classify_ostr_headers(self.inventory)[2], decode_oes_bar, "OSTR1-bar"
            ),
        )

    def solid_strains(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OSTR1 solid element strain blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, GRID, SX, SY, SZ, SXY, SYZ, SZX, VM``.
        """
        return self._cached(
            "solid_strains",
            lambda: self._decode_all(
                classify_ostr_headers(self.inventory)[1],
                decode_oes_solid,
                "OSTR1-solid",
            ),
        )

    def bush_strains(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OSTR1 bush element strain blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, EX, EY, EZ, ETX, ETY, ETZ``.
        """
        return self._cached(
            "bush_strains",
            lambda: self._decode_all(
                classify_ostr_headers(self.inventory)[3], decode_oes_cbush, "OSTR1-bush"
            ),
        )

    def bar_strains_el(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OSTR1EL bar/beam strain blocks (element coordinate system).

        These are the same physical strains as :meth:`bar_strains` but expressed
        in the element (local) coordinate system rather than the basic system.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, GRID, SD, SXC, SXD, SXE, SXF, SMAX, SMIN, MS_T, MS_C``.
        """
        return self._cached(
            "bar_strains_el",
            lambda: self._decode_all(
                classify_ostr_el_headers(self.inventory)[2],
                decode_oes_bar,
                "OSTR1EL-bar",
            ),
        )

    def solid_strains_el(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OSTR1EL solid element strain blocks (element coordinate system).

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, GRID, SX, SY, SZ, SXY, SYZ, SZX, VM``.
        """
        return self._cached(
            "solid_strains_el",
            lambda: self._decode_all(
                classify_ostr_el_headers(self.inventory)[1],
                decode_oes_solid,
                "OSTR1EL-solid",
            ),
        )

    def bush_strains_el(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OSTR1EL bush element strain blocks (element coordinate system).

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, EX, EY, EZ, ETX, ETY, ETZ``.
        """
        return self._cached(
            "bush_strains_el",
            lambda: self._decode_all(
                classify_ostr_el_headers(self.inventory)[3],
                decode_oes_cbush,
                "OSTR1EL-bush",
            ),
        )

    def grid_weight(self) -> Optional[dict]:
        """
        Decode the OGPWG (Grid Point Weight Generator) table.

        Returns
        -------
        dict or None
            ``None`` if no OGPWG table is present.  Otherwise a dict with:

            ``mass`` : float
                Total model mass.
            ``cg`` : list[float, float, float]
                Centre of gravity [X, Y, Z].
            ``S`` : list[list[float]]
                6x6 mass/inertia matrix in the reference CS.
            ``IQ``, ``S1``, ``Q`` : list[list[float]]
                Direction-cosine and principal-frame matrices.
            ``summary`` : pd.DataFrame
                One-row DataFrame with
                ``mass, CG_X, CG_Y, CG_Z, IXX, IYY, IZZ, IXY, IXZ, IYZ``.
        """
        return decode_ogpwg(self.inventory)

    def eigenvalues(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all LAMA (real eigenvalue) tables.

        Available for normal-modes (SOL 103) and buckling (SOL 105) analyses.
        Returns an empty dict for static analysis files.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``MODE, ORDER, EIGENVALUE, RADIANS, CYCLES, GENM, GENSTIF``.
        """
        return self._cached(
            "eigenvalues",
            lambda: decode_lama(self.inventory),
        )

    def metadata(self) -> Dict[int, SubcaseMeta]:
        """
        Extract case-control metadata from every result table header in the file.

        Reads ACODE (analysis type), TCODE (table code), ISUBCASE, and the
        TITLE / SUBTITLE / LABEL strings from each table's IDENT block.

        When multiple tables share the same subcase ID (e.g. OES + OEF for the
        same subcase), the *last* table's strings win for that subcase.  In
        practice all tables in a subcase share the same case-control strings.

        Returns
        -------
        dict
            ``{subcase_id: SubcaseMeta}``

        Examples
        --------
        >>> op2 = OP2("model.op2")
        >>> for sc, m in op2.metadata().items():
        ...     print(sc, m.acode, m.title)
        """

        def _compute() -> Dict[int, SubcaseMeta]:
            inv = self.inventory
            # All 8-byte name records are table-header boundaries
            headers = [r.info.index for r in inv.records if r.info.length == 8]
            result: Dict[int, SubcaseMeta] = {}
            for hdr in headers:
                try:
                    meta = self._read_header_meta(inv, hdr)
                    # Only overwrite if we got a real acode (skip pure boundary markers)
                    if meta.acode > 0 or meta.tcode > 0:
                        result[meta.subcase] = meta
                except Exception:
                    pass
            return result

        return self._cached("metadata", _compute)

    # ------------------------------------------------------------------
    # Geometry (from companion .dat / .bdf file)
    # ------------------------------------------------------------------

    def grid_coordinates(self) -> pd.DataFrame:
        """
        Return grid-point coordinates parsed from the companion bulk-data file.

        Requires the OP2 instance to have been opened with ``geometry=True``
        (or an explicit path), e.g.::

            op2 = OP2("model.op2", geometry=True)
            grids = op2.grid_coordinates()

        Returns
        -------
        pd.DataFrame
            Columns: ``GID`` (int32), ``X`` (float64), ``Y`` (float64),
            ``Z`` (float64), ``CP`` (int16, input coord system),
            ``CD`` (int16, output coord system).  Sorted by ``GID``.

        Raises
        ------
        RuntimeError
            If geometry was not loaded (``geometry=False``).
        """
        return self._geometry.grids

    def element_connectivity(
        self, etype: Optional[str] = None
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Return element-to-node connectivity parsed from the companion bulk-data
        file.

        Requires the OP2 instance to have been opened with ``geometry=True``
        (or an explicit path).

        Parameters
        ----------
        etype : str, optional
            Element type to return, e.g. ``'CTETRA'``, ``'CBEAM'``,
            ``'CBUSH'``.  Case-insensitive.  If ``None`` (default), returns
            a dict of all element types found.

        Returns
        -------
        dict or pd.DataFrame
            If *etype* is ``None``: ``{etype_str: DataFrame}`` for every
            element type present in the bulk data.

            If *etype* is given: the DataFrame for that element type.
            Solid-element DataFrames have columns
            ``EID, PID, G1, G2, G3, G4`` (CTETRA) etc.;
            line-element DataFrames have ``EID, PID, GA, GB``.

        Raises
        ------
        RuntimeError
            If geometry was not loaded.
        KeyError
            If the requested *etype* is not present in the bulk data.
        """
        elems = self._geometry.elements
        if etype is None:
            return elems
        key = etype.upper()
        if key not in elems:
            available = list(elems.keys())
            raise KeyError(f"Element type {key!r} not found.  Available: {available}")
        return elems[key]

    def element_centroids(
        self, etype: Optional[str] = None
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Compute element centroids (average of corner-node coordinates).

        Requires geometry to have been loaded.

        Parameters
        ----------
        etype : str, optional
            Restrict to a single element type.  If ``None``, returns a dict
            for every element type.

        Returns
        -------
        dict or pd.DataFrame
            DataFrame(s) with columns ``EID``, ``X``, ``Y``, ``Z``.
        """
        geom = self._geometry
        grids = geom.grids.set_index("GID")[["X", "Y", "Z"]]

        def _centroid_for(df: pd.DataFrame) -> pd.DataFrame:
            node_cols = [c for c in df.columns if c not in ("EID", "PID")]
            xyz_sum = np.zeros((len(df), 3))
            n_valid = np.zeros(len(df), dtype=int)
            for col in node_cols:
                gids = df[col].values
                mask = gids > 0
                if not mask.any():
                    continue
                valid_gids = gids[mask]
                present = np.isin(valid_gids, grids.index)
                if not present.any():
                    continue
                xyz = grids.loc[valid_gids[present]].values
                # Accumulate only for rows that have this node
                row_idx = np.where(mask)[0][present]
                xyz_sum[row_idx] += xyz
                n_valid[row_idx] += 1
            n_valid = np.maximum(n_valid, 1)
            result = pd.DataFrame(
                {
                    "EID": df["EID"].values,
                    "X": xyz_sum[:, 0] / n_valid,
                    "Y": xyz_sum[:, 1] / n_valid,
                    "Z": xyz_sum[:, 2] / n_valid,
                }
            )
            return result

        if etype is not None:
            key = etype.upper()
            if key not in geom.elements:
                raise KeyError(f"Element type {key!r} not found.")
            return _centroid_for(geom.elements[key])

        return {et: _centroid_for(df) for et, df in geom.elements.items()}

    def stresses_with_corners(self) -> Dict[int, pd.DataFrame]:
        """
        Decode shell stresses including all four corner nodes.

        Returns the centroid row (GRID=0) plus one row per corner grid for
        every element.  This gives 5 rows per CQUAD4 element (1 centroid +
        4 corners).

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, GRID, FD1, SX1, SY1, TXY1, ANG1, MAX_PRIN1, MIN_PRIN1, VON_MISES1,
            FD2, SX2, SY2, TXY2, ANG2, MAX_PRIN2, MIN_PRIN2, VON_MISES2``.
        """
        return self._cached(
            "stresses_corners",
            lambda: self._decode_all(
                classify_oes_headers(self.inventory)[0],
                decode_oes1x1_shell_corners,
                "OES1X1-shell-corners",
            ),
        )

    def nl_bar_stresses(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OESNLXR nonlinear **CBEAM** stress blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, GRID, FIBER, STRESS, EQ_STRESS, TOTAL_STRAIN,
            EFF_STRAIN_PLAS, EFF_CREEP``.
            One row per fiber (C/D/E/F) per station per element.
        """

        def _compute():
            cbeam_blocks, _cbush, _ctetra, _shell = classify_oesnlxr_headers(
                self.inventory
            )
            return self._decode_all(cbeam_blocks, decode_oesnlxr_cbeam, "OESNLXR-cbeam")

        return self._cached("nl_bar_stresses", _compute)

    def nl_bush_stresses(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OESNLXR nonlinear **CBUSH** force/stress/strain blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, FORCE_X, FORCE_Y, FORCE_Z, STRESS_TX, STRESS_TY, STRESS_TZ,
            STRAIN_TX, STRAIN_TY, STRAIN_TZ, MOMENT_X, MOMENT_Y, MOMENT_Z,
            STRESS_RX, STRESS_RY, STRESS_RZ, STRAIN_RX, STRAIN_RY, STRAIN_RZ``.
        """

        def _compute():
            _cbeam, cbush_blocks, _ctetra, _shell = classify_oesnlxr_headers(
                self.inventory
            )
            return self._decode_all(cbush_blocks, decode_oesnlxr_cbush, "OESNLXR-cbush")

        return self._cached("nl_bush_stresses", _compute)

    def nl_solid_stresses(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OESNLXR nonlinear **CTETRA** stress blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, GRID, SX, SY, SZ, SXY, SYZ, SZX, VON_MISES,
            EFF_STRAIN_PLAS, EFF_CREEP, EX, EY, EZ, EXY, EYZ, EZX``.
            ``GRID == 0`` rows are the element centroid.
        """

        def _compute():
            _cbeam, _cbush, ctetra_blocks, _shell = classify_oesnlxr_headers(
                self.inventory
            )
            return self._decode_all(
                ctetra_blocks, decode_oesnlxr_ctetra, "OESNLXR-ctetra"
            )

        return self._cached("nl_solid_stresses", _compute)

    def _nl_shell_raw(self) -> Dict[int, pd.DataFrame]:
        """Decode all OESNLXR shell blocks; cached with all columns."""

        def _compute():
            _cbeam, _cbush, _ctetra, shell_blocks = classify_oesnlxr_headers(
                self.inventory
            )
            return self._decode_all(shell_blocks, decode_oesnlxr_shell, "OESNLXR-shell")

        return self._cached("_nl_shell_raw", _compute)

    def nl_shell_stresses(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OESNLXR nonlinear **CQUAD4/CTRIA3** stress blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, FIBER, FD, SX, SY, TXY, VON_MISES``.
            Two rows per element (FIBER=1 bottom, FIBER=2 top).
        """
        return {sc: df[SHELL_NL_STRESS_COLS] for sc, df in self._nl_shell_raw().items()}

    def nl_shell_strains(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OESNLXR nonlinear **CQUAD4/CTRIA3** strain blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, FIBER, FD, EX, EY, EXY, EFF_STRAIN_PLAS, EFF_CREEP``.
            Two rows per element (FIBER=1 bottom, FIBER=2 top).
        """
        return {sc: df[SHELL_NL_STRAIN_COLS] for sc, df in self._nl_shell_raw().items()}

    def results(self, subcase: int = 1) -> "Results":
        """
        Return a :class:`Results` object for a single subcase.

        Provides attribute-style access to all decoded tables::

            r = op2.results(1)
            r.stresses          # centroid stresses DataFrame
            r.element_forces    # element forces DataFrame
            r.displacements     # displacements DataFrame
            ...

        Parameters
        ----------
        subcase : int
            Subcase ID to extract.  Defaults to ``1``.

        Returns
        -------
        Results
        """
        return Results(self, subcase)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def envelope(
        self,
        result: str = "stresses",
        column: str = "VON_MISES1",
        mode: str = "max",
    ) -> pd.DataFrame:
        """
        Compute the element-wise extreme value of *column* across **all
        subcases**, returning a single "worst case" DataFrame.

        This is most useful for multi-subcase load-case sets where you want
        the governing stress (or force) regardless of which subcase caused it.
        For a single-subcase file the result equals that subcase's table.

        Parameters
        ----------
        result : str
            Which result group to scan.  Any attribute name on this class
            that returns ``{subcase: DataFrame}`` is valid — e.g.
            ``"stresses"``, ``"element_forces"``, ``"strains"``,
            ``"spc_forces"``.  Default ``"stresses"``.
        column : str
            The numeric column to take the extreme over.  Default ``"VON_MISES1"``.
        mode : {"max", "min", "absmax"}
            * ``"max"``    — highest (most positive) value
            * ``"min"``    — lowest (most negative) value
            * ``"absmax"`` — largest absolute value (sign preserved)

        Returns
        -------
        pd.DataFrame
            One row per element / node.  Columns are the union of the input
            columns.  A ``SUBCASE`` column records which subcase provided the
            governing value for each row.  Sorted by the governing column in
            descending order (ascending for ``"min"``).

        Examples
        --------
        >>> worst = op2.envelope("stresses", "VON_MISES1", "absmax")
        >>> worst.head()
        """
        method = getattr(self, result, None)
        if method is None or not callable(method):
            raise ValueError(
                f"{result!r} is not a recognised result method on OP2. "
                f"Try 'stresses', 'element_forces', 'strains', etc."
            )
        subcase_dict: Dict[int, pd.DataFrame] = method()
        if not subcase_dict:
            return pd.DataFrame()

        # Determine the ID column (EID for element results, GRID for nodal)
        id_col = None
        for df in subcase_dict.values():
            if not df.empty:
                id_col = "EID" if "EID" in df.columns else "GRID"
                break
        if id_col is None:
            return pd.DataFrame()

        if mode not in ("max", "min", "absmax"):
            raise ValueError(f"mode must be 'max', 'min', or 'absmax'; got {mode!r}")

        # Tag each subcase frame and stack them
        frames = []
        for sc, df in subcase_dict.items():
            if column not in df.columns:
                continue
            tmp = df.copy()
            tmp["SUBCASE"] = sc
            frames.append(tmp)

        if not frames:
            raise KeyError(f"Column {column!r} not found in any subcase of {result!r}.")

        combined = pd.concat(frames, ignore_index=True)

        # For each ID pick the row with the extreme value
        if mode == "max":
            idx = combined.groupby(id_col)[column].idxmax()
        elif mode == "min":
            idx = combined.groupby(id_col)[column].idxmin()
        else:  # absmax
            idx = combined.groupby(id_col)[column].apply(lambda s: s.abs().idxmax())

        result_df = combined.loc[idx.values].copy().reset_index(drop=True)

        # Sort: ascending for min, descending otherwise
        ascending = mode == "min"
        result_df = result_df.sort_values(column, ascending=ascending).reset_index(
            drop=True
        )
        return result_df

    # Prefixes that identify result/output data-block tables.  All other
    # tables (NX2412, PVT0, CASECC, EQEXIN, …) are metadata or admin blocks.
    _RESULT_TABLE_PREFIXES = (
        "OES",
        "OUG",
        "OUGV",
        "OEF",
        "OQG",
        "OPG",
        "OGP",
        "OGS",
        "OSTR",
        "OGPWG",
        "LAMA",
        "OESNL",
    )

    def table_names(self, results_only: bool = True) -> List[str]:
        """
        Return the OP2 data-block (table) names present in the file,
        in the order they appear.

        Table names are read directly from the 8-byte header records that
        Nastran writes at the start of every data block.  Duplicate entries
        mean the same table appears more than once (one per subcase group).

        Note: the list includes *all* OP2 tables by default, including
        metadata/admin blocks such as ``NX2412`` (NX Nastran version header),
        ``PVT0``/``PVT`` (private version data), ``CASECC`` (case control
        echo), and ``EQEXIN`` (DOF mapping).  Pass ``results_only=True`` to
        keep only tables that contain FEM result data.

        Parameters
        ----------
        results_only : bool, optional
            If ``True``, return only result/output tables (those whose names
            begin with ``OES``, ``OUG``, ``OEF``, ``OQG``, ``OPG``,
            ``OSTR``, ``OGPWG``, ``LAMA``, etc.).  Default ``True``.

        Returns
        -------
        list of str
            Table names in file order.

        Examples
        --------
        >>> op2.table_names()
        ['NX2412', 'PVT0', 'PVT', 'CASECC', 'CASECC1', 'EQEXINS',
         'EQEXIN', 'OGPWG', 'OGPWG', 'OUGV1', 'OES1X1', 'OSTR1X']
        >>> op2.table_names(results_only=True)
        ['OGPWG', 'OGPWG', 'OUGV1', 'OES1X1', 'OSTR1X']
        """
        import re as _re

        _NAME_RE = _re.compile(rb"^[A-Z0-9_]{2,8}[ ]{0,6}$")
        names = [
            r.data.decode("ascii", "replace").rstrip()
            for r in self.inventory.records
            if r.info.length == 8 and _NAME_RE.match(r.data)
        ]
        if results_only:
            names = [n for n in names if n.startswith(self._RESULT_TABLE_PREFIXES)]
        return names

    def describe(self) -> pd.DataFrame:
        """
        Return a statistical summary table for all non-empty result tables in
        the file.

        Each row in the output corresponds to one numeric column in one
        result table (e.g. ``stresses/VON_MISES1``).  The columns are the standard
        descriptive statistics: count, mean, std, min, 25%, 50%, 75%, max.

        Returns
        -------
        pd.DataFrame
            Multi-index DataFrame with levels ``(result, subcase, column)``
            in the index and descriptive statistics as columns.

        Examples
        --------
        >>> op2.describe()
        """
        result_methods: Dict[str, Callable] = {
            "displacements": self.displacements,
            "stresses": self.stresses,
            "strains": self.strains,
            "bar_strains": self.bar_strains,
            "bar_strains_el": self.bar_strains_el,
            "solid_strains": self.solid_strains,
            "solid_strains_el": self.solid_strains_el,
            "bush_strains": self.bush_strains,
            "bush_strains_el": self.bush_strains_el,
            "solid_stresses": self.solid_stresses,
            "bar_stresses": self.bar_stresses,
            "element_forces": self.element_forces,
            "spc_forces": self.spc_forces,
            "applied_loads": self.applied_loads,
            "eigenvalues": self.eigenvalues,
            "nl_bar_stresses": self.nl_bar_stresses,
            "nl_bush_stresses": self.nl_bush_stresses,
            "nl_solid_stresses": self.nl_solid_stresses,
            "nl_shell_stresses": self.nl_shell_stresses,
            "nl_shell_strains": self.nl_shell_strains,
        }
        rows = []
        for name, method in result_methods.items():
            try:
                subcase_dict = method()
            except Exception:
                continue
            for sc, df in subcase_dict.items():
                if df.empty:
                    continue
                numeric_cols = df.select_dtypes("number").columns.tolist()
                # Drop pure-ID columns for the statistics
                stat_cols = [c for c in numeric_cols if c not in ("EID", "GRID", "CP")]
                if not stat_cols:
                    continue
                stats = df[stat_cols].describe().T  # columns: count mean std min ...
                for col, row_s in stats.iterrows():
                    entry = {"result": name, "subcase": sc, "column": col}
                    entry.update(row_s.to_dict())
                    rows.append(entry)

        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows)
        out = out.set_index(["result", "subcase", "column"]).sort_index()
        return out

    def find_by_eid(self, eid: int) -> Dict[str, pd.DataFrame]:
        """
        Retrieve all element results for a single element ID.

        Searches stresses, strains, element forces, solid stresses, and bar
        stresses and returns any rows that match *eid*.

        Parameters
        ----------
        eid : int
            The element ID to look up.

        Returns
        -------
        dict
            ``{result_name: DataFrame}`` — only entries with matching rows are
            included.  The DataFrames are not subcase-keyed; all subcases are
            concatenated.
        """
        sources: Dict[str, Dict[int, pd.DataFrame]] = {
            "stresses": self.stresses(),
            "stresses_corners": self.stresses_with_corners(),
            "strains": self.strains(),
            "bar_strains": self.bar_strains(),
            "bar_strains_el": self.bar_strains_el(),
            "solid_strains": self.solid_strains(),
            "solid_strains_el": self.solid_strains_el(),
            "bush_strains": self.bush_strains(),
            "bush_strains_el": self.bush_strains_el(),
            "element_forces": self.element_forces(),
            "solid_stresses": self.solid_stresses(),
            "bar_stresses": self.bar_stresses(),
            "nl_bar_stresses": self.nl_bar_stresses(),
            "nl_bush_stresses": self.nl_bush_stresses(),
            "nl_solid_stresses": self.nl_solid_stresses(),
            "nl_shell_stresses": self.nl_shell_stresses(),
            "nl_shell_strains": self.nl_shell_strains(),
        }
        out: Dict[str, pd.DataFrame] = {}
        for name, subcase_dict in sources.items():
            matched = []
            for df in subcase_dict.values():
                if "EID" in df.columns:
                    hit = df[df["EID"] == eid]
                    if not hit.empty:
                        matched.append(hit)
            if matched:
                out[name] = pd.concat(matched, ignore_index=True)
        return out

    def find_by_grid(self, grid_id: int) -> Dict[str, pd.DataFrame]:
        """
        Retrieve all nodal results for a single grid (node) ID.

        Searches displacements, SPC forces, and applied loads and returns any
        rows that match *grid_id*.

        Parameters
        ----------
        grid_id : int
            The grid (node) ID to look up.

        Returns
        -------
        dict
            ``{result_name: DataFrame}`` — only entries with matching rows are
            included.  The DataFrames concatenate all subcases.
        """
        sources: Dict[str, Dict[int, pd.DataFrame]] = {
            "displacements": self.displacements(),
            "spc_forces": self.spc_forces(),
            "applied_loads": self.applied_loads(),
        }
        out: Dict[str, pd.DataFrame] = {}
        for name, subcase_dict in sources.items():
            matched = []
            for df in subcase_dict.values():
                if "GRID" in df.columns:
                    hit = df[df["GRID"] == grid_id]
                    if not hit.empty:
                        matched.append(hit)
            if matched:
                out[name] = pd.concat(matched, ignore_index=True)
        return out

    def to_csv(self, output_dir: str = ".") -> Dict[str, str]:
        """
        Export all result tables to CSV files.

        One CSV is written per result type per subcase.  File names follow the
        pattern ``{result}_{subcase}.csv`` (e.g. ``stresses_1.csv``).
        Results with no data are skipped.  The OGPWG summary is written as
        ``grid_weight.csv``.

        Parameters
        ----------
        output_dir : str
            Directory to write the CSV files into.  Created if it does not
            exist.  Defaults to the current working directory.

        Returns
        -------
        dict
            ``{filename: absolute_path}`` for every file written.
        """
        from pathlib import Path as _Path

        out_dir = _Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_results: Dict[str, Dict[int, pd.DataFrame]] = {
            "displacements": self.displacements(),
            "stresses": self.stresses(),
            "stresses_corners": self.stresses_with_corners(),
            "strains": self.strains(),
            "bar_strains": self.bar_strains(),
            "bar_strains_el": self.bar_strains_el(),
            "solid_strains": self.solid_strains(),
            "solid_strains_el": self.solid_strains_el(),
            "bush_strains": self.bush_strains(),
            "bush_strains_el": self.bush_strains_el(),
            "solid_stresses": self.solid_stresses(),
            "bar_stresses": self.bar_stresses(),
            "element_forces": self.element_forces(),
            "spc_forces": self.spc_forces(),
            "applied_loads": self.applied_loads(),
            "nl_bar_stresses": self.nl_bar_stresses(),
            "nl_bush_stresses": self.nl_bush_stresses(),
            "nl_solid_stresses": self.nl_solid_stresses(),
            "nl_shell_stresses": self.nl_shell_stresses(),
            "nl_shell_strains": self.nl_shell_strains(),
            "eigenvalues": self.eigenvalues(),
        }

        written: Dict[str, str] = {}
        for result_name, subcase_dict in all_results.items():
            for sc, df in subcase_dict.items():
                if df.empty:
                    continue
                fname = f"{result_name}_{sc}.csv"
                fpath = out_dir / fname
                df.to_csv(fpath, index=False)
                written[fname] = str(fpath)

        # OGPWG summary
        gw = self.grid_weight()
        if gw is not None:
            fname = "grid_weight.csv"
            fpath = out_dir / fname
            gw["summary"].to_csv(fpath, index=False)
            written[fname] = str(fpath)

        return written

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    _METADATA_TABLES = frozenset(
        {
            "NX2412",
            "PVT0",
            "PVT",
            "CASECC",
            "CASECC1",
            "EQEXIN",
            "EQEXINS",
        }
    )

    def file_info(self) -> dict:
        """
        Return a dictionary of file-level metadata.

        Parses lightweight header records only — no result data is decoded.

        Returns
        -------
        dict with keys:

        ``path``
            Absolute path to the OP2 file.
        ``product``
            Nastran product string, e.g. ``'NX Nastran'``.
        ``endian``
            Byte order: ``'little'`` or ``'big'``.
        ``result_tables``
            Sorted list of result table names present in the file
            (metadata tables are excluded).  Examples: ``['OES1X1',
            'OESNLXR', 'OUGV1']``.
        ``solution_type``
            Human-readable description of the analysis type, inferred
            from the result tables present.
        ``n_subcases``
            Number of unique subcase IDs found.
        ``params``
            Dict of solution parameters parsed from the PVT table.
            Keys are parameter names (e.g. ``'AUTOSPC'``, ``'K6ROT'``,
            ``'LGDISP'``).  Values are ``int``, ``float``, or ``str``.

        Examples
        --------
        >>> op2 = OP2("model.op2")
        >>> info = op2.file_info()
        >>> info['solution_type']
        'Nonlinear Static (SOL 106)'
        >>> info['params']['LGDISP']
        1
        """
        inv = self.inventory
        bo = inv.endian  # '<' or '>'

        # ── 1. All unique 8-byte table names ──────────────────────────
        all_tables: set = {
            rec.data.rstrip(b"\x00 ").decode("latin-1", errors="replace")
            for rec in inv.records
            if rec.info.length == 8
        }
        result_tables = sorted(all_tables - self._METADATA_TABLES)

        # ── 2. Product ────────────────────────────────────────────────
        product = "NX Nastran" if "NX2412" in all_tables else "MSC Nastran"

        # ── 3. Solution type (inferred from tables present) ───────────
        has_lama = "LAMA" in all_tables
        has_nlxr = "OESNLXR" in all_tables
        has_onrgy = any(t.startswith("ONRGY") for t in all_tables)
        has_freq = "OESATO1" in all_tables or "OUPV1" in all_tables

        if has_lama and has_onrgy:
            sol_type = "Normal Modes (SOL 103)"
        elif has_lama:
            sol_type = "Buckling (SOL 105)"
        elif has_nlxr:
            sol_type = "Nonlinear Static (SOL 106)"
        elif has_freq:
            sol_type = "Frequency Response (SOL 108/111)"
        else:
            sol_type = "Linear Static (SOL 101)"

        # ── 4. Subcases ───────────────────────────────────────────────
        sc_df = self.subcases()
        n_subcases = len(sc_df) if not sc_df.empty else 0

        # ── 5. PVT parameters ─────────────────────────────────────────
        params = self._parse_pvt_params(bo)

        return {
            "path": str(self.path.resolve()),
            "product": product,
            "endian": "little" if bo == "<" else "big",
            "result_tables": result_tables,
            "solution_type": sol_type,
            "n_subcases": n_subcases,
            "params": params,
        }

    def _parse_pvt_params(self, endian: str = "<") -> dict:
        """
        Parse the PVT solution-parameter block into a plain dict.

        Each entry in the block has the form::

            [8-char keyword][4-byte type][value bytes]

        where type=1 → int (4 bytes), type=2 → float32 (4 bytes),
        type=3 → char8 (8 bytes).
        """
        import struct as _struct

        bo = endian  # '<' or '>'
        inv = self.inventory
        current_table: Optional[str] = None
        pvt_data: Optional[bytes] = None

        for rec in inv.records:
            if rec.info.length == 8:
                current_table = rec.data.rstrip(b"\x00 ").decode(
                    "latin-1", errors="replace"
                )
            elif current_table == "PVT" and rec.info.length > 8:
                pvt_data = rec.data[: rec.info.length]
                break

        if pvt_data is None:
            return {}

        params: dict = {}
        pos = 0
        n = len(pvt_data)
        while pos + 12 <= n:  # minimum: 8 (keyword) + 4 (type)
            keyword = (
                pvt_data[pos : pos + 8].decode("latin-1", errors="replace").rstrip()
            )
            pos += 8
            type_code = _struct.unpack_from(bo + "i", pvt_data, pos)[0]
            pos += 4

            if type_code == 1:  # integer
                if pos + 4 > n:
                    break
                val: object = _struct.unpack_from(bo + "i", pvt_data, pos)[0]
                pos += 4
            elif type_code == 2:  # real (float32)
                if pos + 4 > n:
                    break
                val = float(_struct.unpack_from(bo + "f", pvt_data, pos)[0])
                pos += 4
            elif type_code == 3:  # char8
                if pos + 8 > n:
                    break
                val = (
                    pvt_data[pos : pos + 8].decode("latin-1", errors="replace").rstrip()
                )
                pos += 8
            else:
                break  # unknown type — stop parsing

            params[keyword] = val

        return params

    def subcases(self) -> pd.DataFrame:
        """
        Return a summary of which result types are available for each subcase.

        Scans all decoded result tables and produces a cross-tabulation of
        ``(subcase_id x result_type)`` showing the row count for each
        combination that has data, and 0 otherwise.

        Returns
        -------
        pd.DataFrame
            Index = subcase IDs (int).  Columns = result type names.
            Values = number of rows decoded for that (subcase, result) pair.

        Examples
        --------
        >>> op2.subcases()
                displacements  stresses  element_forces  ...
        subcase
        1                3860       754            1395  ...
        """
        result_methods: Dict[str, Callable] = {
            "displacements": self.displacements,
            "stresses": self.stresses,
            "stresses_corners": self.stresses_with_corners,
            "strains": self.strains,
            "bar_strains": self.bar_strains,
            "bar_strains_el": self.bar_strains_el,
            "solid_strains": self.solid_strains,
            "solid_strains_el": self.solid_strains_el,
            "bush_strains": self.bush_strains,
            "bush_strains_el": self.bush_strains_el,
            "solid_stresses": self.solid_stresses,
            "bar_stresses": self.bar_stresses,
            "element_forces": self.element_forces,
            "spc_forces": self.spc_forces,
            "applied_loads": self.applied_loads,
            "eigenvalues": self.eigenvalues,
            "nl_bar_stresses": self.nl_bar_stresses,
            "nl_bush_stresses": self.nl_bush_stresses,
            "nl_solid_stresses": self.nl_solid_stresses,
            "nl_shell_stresses": self.nl_shell_stresses,
            "nl_shell_strains": self.nl_shell_strains,
        }
        # Collect (subcase, result_name, n_rows) triples
        rows: list = []
        all_subcases: set = set()
        counts: Dict[tuple, int] = {}
        for name, method in result_methods.items():
            try:
                d = method()
            except Exception:
                d = {}
            for sc, df in d.items():
                all_subcases.add(sc)
                counts[(sc, name)] = len(df)

        if not all_subcases:
            return pd.DataFrame()

        sc_list = sorted(all_subcases)
        col_list = list(result_methods.keys())
        data = {col: [counts.get((sc, col), 0) for sc in sc_list] for col in col_list}
        out = pd.DataFrame(data, index=sc_list)
        out.index.name = "subcase"
        # Drop columns that are all-zero (result type absent from file)
        out = out.loc[:, (out > 0).any(axis=0)]
        return out

    def eqexin(self) -> pd.DataFrame:
        """
        Decode the EQEXIN table — the mapping from grid ID to its
        degree-of-freedom (DOF) pointer in the equation set.

        The EQEXIN table is always present in Nastran OP2 files.  It records
        the internal equation number for each unconstrained DOF of every
        grid point.  The ``EQTYPE`` column is the packed DOF-set pointer
        (not a simple bitmask; a value of 0 means the grid is fully
        constrained / has no free DOFs in the solution).

        Returns
        -------
        pd.DataFrame
            Columns ``GRID`` (int) and ``EQTYPE`` (int), one row per grid.
            Returns an empty DataFrame if no EQEXIN table is found.
        """
        inv = self.inventory
        # Find the EQEXIN table-name record (8 bytes), then walk forward to
        # find the length-announcement 4-byte record followed by the data block.
        # Pattern: [name(8)] ... [ctrl(4)]* [n_words(4)] [data(n_words*4)]
        for i, rec in enumerate(inv.records):
            if rec.ascii_hint.startswith("EQEXIN") and rec.info.length == 8:
                for j in range(i + 1, min(i + 15, len(inv.records))):
                    jr = inv.records[j]
                    if jr.info.length != 4:
                        continue
                    n_words = struct.unpack("<i", jr.data)[0]
                    if n_words <= 0 or n_words % 2 != 0:
                        continue
                    # The next record should be exactly the data block
                    k = j + 1
                    if k >= len(inv.records):
                        break
                    kr = inv.records[k]
                    if kr.info.length != n_words * 4:
                        continue
                    raw = np.frombuffer(inv.get_record_data(k), dtype="<i4").reshape(
                        -1, 2
                    )
                    grids = raw[:, 0]
                    # Sanity: grid IDs should be small positive integers
                    if int(grids.min()) >= 1 and int(grids.max()) < 10_000_000:
                        return pd.DataFrame({"GRID": raw[:, 0], "EQTYPE": raw[:, 1]})
        return pd.DataFrame(columns=["GRID", "EQTYPE"])

    def summary(self) -> pd.DataFrame:
        """
        Return a DataFrame listing every record in the file with its
        byte offset, length, and probable table name.
        """
        inv = self.inventory
        rows = [
            {
                "rec": r.info.index,
                "offset": r.info.offset,
                "length": r.info.length,
                "table": r.probable_table_name or "",
                "hint": r.ascii_hint[:48],
            }
            for r in inv.records
        ]
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        inv = self.inventory
        n_recs = len(inv.records)
        # Collect non-empty result names for a quick overview
        result_methods = [
            ("displ", self.displacements),
            ("stress", self.stresses),
            ("forces", self.element_forces),
            ("spc", self.spc_forces),
            ("loads", self.applied_loads),
            ("strains", self.strains),
            ("solid", self.solid_stresses),
            ("bar", self.bar_stresses),
            ("eigs", self.eigenvalues),
        ]
        parts = []
        for label, method in result_methods:
            try:
                d = method()
                if d:
                    subcases = sorted(d.keys())
                    total = sum(len(v) for v in d.values())
                    sc_str = (
                        f"sc={subcases[0]}"
                        if len(subcases) == 1
                        else f"{len(subcases)} subcases"
                    )
                    parts.append(f"{label}({sc_str}, {total} rows)")
            except Exception:
                pass
        content = ", ".join(parts) if parts else "no results decoded"
        return f"OP2({self.path.name!r}, {n_recs} records, {content})"


# ---------------------------------------------------------------------------
# Results convenience wrapper
# ---------------------------------------------------------------------------


class Results:
    """
    Flat attribute-style view of all decoded result tables for a single subcase.

    Obtained via :meth:`OP2.results`:

    .. code-block:: python

        r = op2.results(subcase=1)
        r.stresses          # DataFrame  (centroid stresses)
        r.stresses_corners  # DataFrame  (centroid + 4 corner rows per element)
        r.element_forces    # DataFrame  (shell/bar element forces)
        r.bush_stresses     # DataFrame  (CBUSH spring deformations)
        r.bush_forces       # DataFrame  (CBUSH spring forces)
        r.displacements     # DataFrame
        r.spc_forces        # DataFrame
        r.applied_loads     # DataFrame
        r.strains           # DataFrame
        r.solid_stresses    # DataFrame
        r.bar_stresses      # DataFrame
        r.eigenvalues       # DataFrame  (empty for static analyses)

    Missing result types (no data in the file for this subcase) are represented
    as empty DataFrames rather than raising ``KeyError``.
    """

    def __init__(self, op2: "OP2", subcase: int) -> None:
        self._op2 = op2
        self.subcase = subcase
        self._load()

    def _get(self, d: Dict[int, pd.DataFrame]) -> pd.DataFrame:
        """Return the DataFrame for this subcase, or an empty DataFrame."""
        return d.get(self.subcase, pd.DataFrame())

    def _load(self) -> None:
        op2 = self._op2
        self.displacements = self._get(op2.displacements())
        self.stresses = self._get(op2.stresses())
        self.stresses_corners = self._get(op2.stresses_with_corners())
        self.strains = self._get(op2.strains())
        self.solid_stresses = self._get(op2.solid_stresses())
        self.bar_stresses = self._get(op2.bar_stresses())
        self.bush_stresses = self._get(op2.bush_stresses())
        self.element_forces = self._get(op2.element_forces())
        self.bush_forces = self._get(op2.bush_forces())
        self.spc_forces = self._get(op2.spc_forces())
        self.applied_loads = self._get(op2.applied_loads())
        self.eigenvalues = self._get(op2.eigenvalues())
        self.bar_strains = self._get(op2.bar_strains())
        self.bar_strains_el = self._get(op2.bar_strains_el())
        self.solid_strains = self._get(op2.solid_strains())
        self.solid_strains_el = self._get(op2.solid_strains_el())
        self.bush_strains = self._get(op2.bush_strains())
        self.bush_strains_el = self._get(op2.bush_strains_el())
        self.nl_bar_stresses = self._get(op2.nl_bar_stresses())
        self.nl_bush_stresses = self._get(op2.nl_bush_stresses())
        self.nl_solid_stresses = self._get(op2.nl_solid_stresses())
        self.nl_shell_stresses = self._get(op2.nl_shell_stresses())
        self.nl_shell_strains = self._get(op2.nl_shell_strains())

    def __repr__(self) -> str:
        parts = []
        for attr in (
            "displacements",
            "stresses",
            "stresses_corners",
            "strains",
            "solid_stresses",
            "bar_stresses",
            "bush_stresses",
            "element_forces",
            "bush_forces",
            "spc_forces",
            "applied_loads",
            "eigenvalues",
            "bar_strains",
            "bar_strains_el",
            "solid_strains",
            "solid_strains_el",
            "bush_strains",
            "bush_strains_el",
            "nl_bar_stresses",
            "nl_bush_stresses",
            "nl_solid_stresses",
            "nl_shell_stresses",
            "nl_shell_strains",
        ):
            df = getattr(self, attr)
            if not df.empty:
                parts.append(f"{attr}({len(df)} rows)")
        body = ", ".join(parts) if parts else "no data"
        return f"Results(subcase={self.subcase}, {body})"
