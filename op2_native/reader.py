# op2_native/reader.py
"""
Central OP2 class.  Open a file once; access any result type as a property
or method.  All result methods return ``{subcase_id: DataFrame}``.
"""
from __future__ import annotations

import struct
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional

import pandas as pd

from .fortran_io import FortranUnformattedReader
from .models import OP2Inventory, OP2Record
from .op2_reader import OP2Reader

# decoder imports
from .decoders.ougv1 import find_ougv1_headers, classify_ougv1_headers, decode_ougv1
from .decoders.oes_search import (
    find_oes_tables,
    find_oef_tables,
    classify_ostr_headers,
    classify_oes_headers,
)
from .decoders.oes1x1_shell import decode_oes1x1_shell, decode_oes1x1_shell_corners
from .decoders.oes_solid import decode_oes_solid
from .decoders.oes_bar import decode_oes_bar
from .decoders.oes_cbush import decode_oes_cbush
from .decoders.oef1 import decode_oef1, classify_oef_headers
from .decoders.oqg1 import decode_oqg1, classify_oqg_headers
from .decoders.opg1 import decode_opg1, classify_opg_headers
from .decoders.ostr1 import decode_ostr1
from .decoders.ogpwg import decode_ogpwg
from .decoders.lama import decode_lama


class OP2:
    """
    Lightweight Nastran OP2 reader.

    Parameters
    ----------
    path : str or Path
        Path to the .op2 file.

    Examples
    --------
    >>> op2 = OP2("model.op2")
    >>> disp = op2.displacements()   # {subcase_id: DataFrame}
    >>> stress = op2.stresses()      # {subcase_id: DataFrame}
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._inv: Optional[OP2Inventory] = None
        # Result cache: keyed by method name so each decode runs at most once.
        self._cache: Dict[str, object] = {}

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
                words = struct.unpack("<7i", rec.data)
                isubcase = words[6]
                return max(1, isubcase)
        return 1  # fallback

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
                df = decode_fn(inv, hdr) if ekey_idx is None else decode_fn(inv, hdr, ekey_idx)
            except Exception as exc:
                warnings.warn(f"{label} header rec {hdr}: {exc}", RuntimeWarning)
                continue
            if sc in result:
                result[sc] = pd.concat([result[sc], df], ignore_index=True)
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
            ``GRID, COMP, CP, DX, DY, DZ, RX, RY, RZ``.
        """
        return self._cached(
            "displacements",
            lambda: self._decode_all(
                classify_ougv1_headers(self.inventory), decode_ougv1, "OUGV1"
            ),
        )

    def stresses(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OES1X1 **shell** stress blocks.

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, FD1, SX1, SY1, TXY1, ANG1, MAJOR1, MINOR1, VM1,
            FD2, SX2, SY2, TXY2, ANG2, MAJOR2, MINOR2, VM2``.
        """
        return self._cached(
            "stresses",
            lambda: self._decode_all(
                classify_oes_headers(self.inventory)[0],
                decode_oes1x1_shell,
                "OES1X1-shell",
            ),
        )

    def solid_stresses(self) -> Dict[int, pd.DataFrame]:
        """
        Decode all OES1X1 **solid** element stress blocks
        (CHEXA / CPENTA / CTETRA).

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}`` with columns
            ``EID, GRID, SX, SY, SZ, SXY, SYZ, SZX, VM``.
            Rows with ``GRID == 0`` are the centroid average.
        """
        return self._cached(
            "solid_stresses",
            lambda: self._decode_all(
                classify_oes_headers(self.inventory)[1],
                decode_oes_solid,
                "OES1X1-solid",
            ),
        )

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

        Returns
        -------
        dict
            ``{subcase_id: DataFrame}``.
        """
        def _compute():
            shells, bars, _bushes, _others = classify_oef_headers(self.inventory)
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
            _shells, _bars, bushes, _others = classify_oef_headers(self.inventory)
            return self._decode_all(bushes, decode_oef1, "OEF1-cbush")

        return self._cached("bush_forces", _compute)

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
            ``EID, LOC, FD1, EX1, EY1, EXY1, ET1Z1, ET2Z1, EANG1, EMAJOR1,
            FD2, EX2, EY2, EXY2, ET1Z2, ET2Z2, EANG2, EMAJOR2``.
        """
        return self._cached(
            "strains",
            lambda: self._decode_all(
                classify_ostr_headers(self.inventory)[0], decode_ostr1, "OSTR1"
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
            ``EID, GRID, FD1, SX1, SY1, TXY1, ANG1, MAJOR1, MINOR1, VM1,
            FD2, SX2, SY2, TXY2, ANG2, MAJOR2, MINOR2, VM2``.
        """
        return self._cached(
            "stresses_corners",
            lambda: self._decode_all(
                classify_oes_headers(self.inventory)[0],
                decode_oes1x1_shell_corners,
                "OES1X1-shell-corners",
            ),
        )

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
        column: str = "VM1",
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
            The numeric column to take the extreme over.  Default ``"VM1"``.
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
        >>> worst = op2.envelope("stresses", "VM1", "absmax")
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

    def describe(self) -> pd.DataFrame:
        """
        Return a statistical summary table for all non-empty result tables in
        the file.

        Each row in the output corresponds to one numeric column in one
        result table (e.g. ``stresses/VM1``).  The columns are the standard
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
            "solid_stresses": self.solid_stresses,
            "bar_stresses": self.bar_stresses,
            "element_forces": self.element_forces,
            "spc_forces": self.spc_forces,
            "applied_loads": self.applied_loads,
            "eigenvalues": self.eigenvalues,
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
            "element_forces": self.element_forces(),
            "solid_stresses": self.solid_stresses(),
            "bar_stresses": self.bar_stresses(),
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
            "solid_stresses": self.solid_stresses(),
            "bar_stresses": self.bar_stresses(),
            "element_forces": self.element_forces(),
            "spc_forces": self.spc_forces(),
            "applied_loads": self.applied_loads(),
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

    def subcases(self) -> pd.DataFrame:
        """
        Return a summary of which result types are available for each subcase.

        Scans all decoded result tables and produces a cross-tabulation of
        ``(subcase_id × result_type)`` showing the row count for each
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
            "solid_stresses": self.solid_stresses,
            "bar_stresses": self.bar_stresses,
            "element_forces": self.element_forces,
            "spc_forces": self.spc_forces,
            "applied_loads": self.applied_loads,
            "eigenvalues": self.eigenvalues,
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
                import numpy as _np
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
                    raw = _np.frombuffer(kr.data, dtype="<i4").reshape(-1, 2)
                    grids = raw[:, 0]
                    # Sanity: grid IDs should be small positive integers
                    if int(grids.min()) >= 1 and int(grids.max()) < 10_000_000:
                        return pd.DataFrame(
                            {"GRID": raw[:, 0], "EQTYPE": raw[:, 1]}
                        )
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
        ):
            df = getattr(self, attr)
            if not df.empty:
                parts.append(f"{attr}({len(df)} rows)")
        body = ", ".join(parts) if parts else "no data"
        return f"Results(subcase={self.subcase}, {body})"
