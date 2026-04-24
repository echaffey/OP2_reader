"""
femdata.result_array
~~~~~~~~~~~~~~~~~~~~

ResultArray: a 2-D, index-aligned array of FEM results.

Memory layout
-------------
    _index : int32[n]       entity IDs, always sorted ascending
    _data  : float32[n, m]  result values (rows track _index order)
    _cols  : tuple[str]     column names (length m)

Arithmetic
----------
Binary operations between two ResultArrays automatically align on their
shared entity IDs before applying the operation, so you can safely add,
subtract, scale, etc. results from different subcases or element subsets
without pre-sorting.

    a + b   element-wise add on common index
    a - b   element-wise subtract
    a * 2.0 scalar scale every column
    etc.

All arithmetic returns a new ResultArray; inputs are never mutated.
"""

import numpy as np
from typing import Dict, List, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

_Index = Union[int, np.integer, Sequence[int], np.ndarray]
_Operand = Union[int, float, np.ndarray, "ResultArray"]


# ---------------------------------------------------------------------------
# Indexer helpers
# ---------------------------------------------------------------------------


class _LocIndexer:
    """
    Enables ``ra.loc[eid]`` and ``ra.loc[[e1, e2, ...]]`` retrieval by
    entity-ID value (not by integer position).
    """

    __slots__ = ("_ra",)

    def __init__(self, ra: ResultArray) -> None:
        self._ra = ra

    def __getitem__(self, key: _Index) -> Union[np.ndarray, "ResultArray"]:
        ra = self._ra
        scalar = isinstance(key, (int, np.integer))
        ids = np.atleast_1d(np.asarray(key, dtype=np.int32))

        pos = np.searchsorted(ra._index, ids)
        valid = (pos < len(ra._index)) & (ra._index[pos] == ids)
        if not valid.all():
            raise KeyError(f"entity IDs not in index: {ids[~valid].tolist()}")

        if scalar:
            # Return a plain 1-D numpy row — no index wrapper needed
            return ra._data[pos[0]]
        return ResultArray._from_sorted(
            ra._index[pos],
            ra._data[pos],
            ra._cols,
            ra._domain,
            ra.label,
            ra._data.dtype,
        )


class _ILocIndexer:
    """
    Enables ``ra.iloc[i]``, ``ra.iloc[0:10]``, ``ra.iloc[bool_mask]``
    retrieval by integer position (not by entity-ID value).
    """

    __slots__ = ("_ra",)

    def __init__(self, ra: ResultArray) -> None:
        self._ra = ra

    def __getitem__(self, key) -> "ResultArray":
        ra = self._ra
        return ResultArray._from_sorted(
            ra._index[key],
            ra._data[key],
            ra._cols,
            ra._domain,
            ra.label,
            ra._data.dtype,
        )


# ---------------------------------------------------------------------------
# ResultArray
# ---------------------------------------------------------------------------


class ResultArray:
    """
    A 2-D, index-aligned array of FEM results.

    Parameters
    ----------
    index : array-like of int
        Entity IDs — EIDs for element results, GRIDs for nodal results.
        Duplicates are not allowed.  The array is sorted on construction and
        ``data`` rows are reordered to match.
    data : array-like, shape (n,) or (n, m)
        Numeric result values.  Stored as *dtype* (default ``float32``, the
        native precision of Nastran OP2 output).
    columns : sequence of str
        Name for each of the *m* result columns.
    domain : {'element', 'grid'}
        Whether *index* contains element or grid IDs.
    label : str
        Human-readable tag, e.g. ``"SC1 von Mises"``.
    dtype : numpy dtype
        Storage precision.  ``float32`` halves memory vs ``float64`` with no
        meaningful loss for structural post-processing comparisons.

    Notes
    -----
    *Fast path*: when two ResultArrays share an identical index object (or
    identical values), arithmetic is a direct numpy call with no lookup
    overhead.  This is the normal case when both results come from the same
    subcase of the same mesh.

    *Slow path*: when indices differ, ``numpy.searchsorted`` is used to
    compute the shared subset in O(n log n).

    Examples
    --------
    >>> import numpy as np
    >>> from femdata import ResultArray
    >>> a = ResultArray([1, 2, 3], [[100., 10.], [200., 20.], [300., 30.]], ['VM', 'P1'], label='SC1')
    >>> b = ResultArray([2, 3, 4], [[201., 21.], [301., 31.], [401., 41.]], ['VM', 'P1'], label='SC2')
    >>> diff = a - b          # aligned on common index {2, 3}
    >>> diff['VM']            # returns numpy array
    array([-1., -1.], dtype=float32)
    >>> diff.index
    array([2, 3], dtype=int32)
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        index: _Index,
        data: Union[np.ndarray, Sequence],
        columns: Sequence[str],
        *,
        domain: str = "element",
        label: str = "",
        dtype=np.float32,
    ) -> None:
        idx = np.asarray(index, dtype=np.int32)
        dat = np.asarray(data, dtype=dtype)

        if dat.ndim == 1:
            dat = dat[:, np.newaxis]
        if idx.ndim != 1:
            raise ValueError("index must be 1-D")
        if dat.ndim != 2:
            raise ValueError("data must be 1-D or 2-D")
        if idx.shape[0] != dat.shape[0]:
            raise ValueError(
                f"index length ({idx.shape[0]}) != data rows ({dat.shape[0]})"
            )
        cols = tuple(columns)
        if dat.shape[1] != len(cols):
            raise ValueError(
                f"data has {dat.shape[1]} column(s) but {len(cols)} name(s) given"
            )
        if len(set(cols)) != len(cols):
            raise ValueError("duplicate column names are not allowed")

        order = np.argsort(idx, kind="stable")
        self._index: np.ndarray = idx[order]
        self._data: np.ndarray = np.ascontiguousarray(dat[order], dtype=dtype)
        self._cols: tuple = cols
        self._domain: str = domain
        self.label: str = label
        # lazy index map: built only if needed for dict-based lookup
        self._index_map: Union[Dict[int, int], None] = None

    @classmethod
    def _from_sorted(
        cls,
        index: np.ndarray,
        data: np.ndarray,
        cols: tuple,
        domain: str,
        label: str,
        dtype,
    ) -> ResultArray:
        """
        Fast internal constructor — skips sorting and validation.
        *index* must already be sorted ascending with no duplicates.
        """
        ra: ResultArray = object.__new__(cls)
        ra._index = np.asarray(index, dtype=np.int32)
        ra._data = np.ascontiguousarray(data, dtype=dtype)
        ra._cols = cols
        ra._domain = domain
        ra.label = label
        ra._index_map = None
        return ra

    @classmethod
    def from_dataframe(
        cls,
        df,
        *,
        domain: str = "element",
        label: str = "",
        dtype=np.float32,
    ) -> ResultArray:
        """
        Construct from a pandas DataFrame whose integer index contains entity IDs.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame index becomes the ResultArray index; columns become
            result fields.
        """
        return cls(
            df.index.to_numpy(dtype=np.int32),
            df.to_numpy(dtype=float),
            list(df.columns),
            domain=domain,
            label=label,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def index(self) -> np.ndarray:
        """Sorted int32 array of entity IDs (read-only view)."""
        return self._index

    @property
    def data(self) -> np.ndarray:
        """Raw (n, m) float array — rows correspond to ``index`` order."""
        return self._data

    @property
    def columns(self) -> tuple:
        """Tuple of result-field name strings."""
        return self._cols

    @property
    def domain(self) -> str:
        """``'element'`` or ``'grid'``."""
        return self._domain

    @property
    def shape(self) -> Tuple[int, int]:
        """``(n_entities, n_columns)``."""
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # Index map (lazy, for explicit dict-style lookup if ever needed)
    # ------------------------------------------------------------------

    @property
    def index_map(self) -> Dict[int, int]:
        """Dict mapping entity ID → row position (built once, cached)."""
        if self._index_map is None:
            self._index_map = {int(v): i for i, v in enumerate(self._index)}
        return self._index_map

    # ------------------------------------------------------------------
    # Indexers
    # ------------------------------------------------------------------

    @property
    def loc(self) -> _LocIndexer:
        """Select rows by entity ID value: ``ra.loc[eid]`` or ``ra.loc[[e1, e2]]``."""
        return _LocIndexer(self)

    @property
    def iloc(self) -> _ILocIndexer:
        """Select rows by integer position: ``ra.iloc[0]`` or ``ra.iloc[0:10]``."""
        return _ILocIndexer(self)

    # ------------------------------------------------------------------
    # Column / row selection via []
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        """
        Column selection:

            ``ra['VM']``           → 1-D numpy array (no copy)
            ``ra[['VM', 'P1']]``   → new ResultArray (2 columns)

        Row selection (by position — for ID-based access use ``.loc``):

            ``ra[bool_mask]``      → new ResultArray
            ``ra[int_array]``      → new ResultArray
            ``ra[slice]``          → new ResultArray
        """
        if isinstance(key, str):
            try:
                j = self._cols.index(key)
            except ValueError:
                raise KeyError(
                    f"column {key!r} not found; available: {list(self._cols)}"
                )
            return self._data[:, j]

        if isinstance(key, (list, tuple)) and key and isinstance(key[0], str):
            try:
                js = [self._cols.index(k) for k in key]
            except ValueError as exc:
                raise KeyError(str(exc))
            return ResultArray._from_sorted(
                self._index.copy(),
                self._data[:, js],
                tuple(key),
                self._domain,
                self.label,
                self._data.dtype,
            )

        # numpy-style row indexing: bool mask, integer array, or slice
        return ResultArray._from_sorted(
            self._index[key],
            self._data[key],
            self._cols,
            self._domain,
            self.label,
            self._data.dtype,
        )

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    def _align_pair(
        self, other: ResultArray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return ``(data_self, data_other, common_index)`` aligned to the shared
        subset of entity IDs.

        Fast path: if both arrays carry the same index object or identical
        values, no reindexing is performed — just returns the raw data arrays.
        """
        if len(self) == len(other) and (
            self._index is other._index or np.array_equal(self._index, other._index)
        ):
            return self._data, other._data, self._index

        common = np.intersect1d(self._index, other._index)
        ps = np.searchsorted(self._index, common)
        po = np.searchsorted(other._index, common)
        return self._data[ps], other._data[po], common

    def align(self, other: ResultArray) -> Tuple[ResultArray, ResultArray]:
        """
        Return ``(self_aligned, other_aligned)``, both restricted to the
        common entity IDs.

        Useful for inspecting what changes between two partial-mesh results
        before performing arithmetic.
        """
        if not isinstance(other, ResultArray):
            raise TypeError("align() requires another ResultArray")
        ds, do, common = self._align_pair(other)
        return (
            ResultArray._from_sorted(
                common, ds, self._cols, self._domain, self.label, self._data.dtype
            ),
            ResultArray._from_sorted(
                common, do, other._cols, other._domain, other.label, other._data.dtype
            ),
        )

    def reindex(self, new_index: _Index, *, fill_value: float = 0.0) -> ResultArray:
        """
        Project to *new_index*, filling missing rows with *fill_value*.

        Entities present in *self* are copied; entities absent are set to
        *fill_value*.  Useful for broadcasting a partial-mesh result onto
        the full mesh before arithmetic.
        """
        new_idx = np.sort(np.asarray(new_index, dtype=np.int32))
        new_data = np.full(
            (len(new_idx), self._data.shape[1]),
            fill_value,
            dtype=self._data.dtype,
        )
        pos = np.searchsorted(self._index, new_idx)
        mask = pos < len(self._index)
        mask[mask] &= self._index[pos[mask]] == new_idx[mask]
        new_data[mask] = self._data[pos[mask]]
        return ResultArray._from_sorted(
            new_idx, new_data, self._cols, self._domain, self.label, self._data.dtype
        )

    # ------------------------------------------------------------------
    # Arithmetic — binary operations
    # ------------------------------------------------------------------

    def _binop(self, other: _Operand, op, reverse: bool = False) -> ResultArray:
        """Core binary operation dispatcher."""
        if isinstance(other, ResultArray):
            if self._cols != other._cols:
                raise ValueError(
                    f"column mismatch: {list(self._cols)} vs {list(other._cols)}\n"
                    "Tip: use ra[['col1','col2']] to select matching columns first."
                )
            ds, do, common = self._align_pair(other)
            result_data = op(do, ds) if reverse else op(ds, do)
            return ResultArray._from_sorted(
                common, result_data, self._cols, self._domain, "", self._data.dtype
            )
        # scalar or numpy broadcast value
        result_data = op(other, self._data) if reverse else op(self._data, other)
        return ResultArray._from_sorted(
            self._index,
            result_data,
            self._cols,
            self._domain,
            self.label,
            self._data.dtype,
        )

    def __add__(self, other):
        return self._binop(other, np.add)

    def __sub__(self, other):
        return self._binop(other, np.subtract)

    def __mul__(self, other):
        return self._binop(other, np.multiply)

    def __truediv__(self, other):
        return self._binop(other, np.true_divide)

    def __floordiv__(self, other):
        return self._binop(other, np.floor_divide)

    def __pow__(self, other):
        return self._binop(other, np.power)

    def __mod__(self, other):
        return self._binop(other, np.mod)

    def __radd__(self, other):
        return self._binop(other, np.add, reverse=True)

    def __rsub__(self, other):
        return self._binop(other, np.subtract, reverse=True)

    def __rmul__(self, other):
        return self._binop(other, np.multiply, reverse=True)

    def __rtruediv__(self, other):
        return self._binop(other, np.true_divide, reverse=True)

    def __rpow__(self, other):
        return self._binop(other, np.power, reverse=True)

    def __neg__(self) -> ResultArray:
        return ResultArray._from_sorted(
            self._index,
            -self._data,
            self._cols,
            self._domain,
            self.label,
            self._data.dtype,
        )

    def __abs__(self) -> ResultArray:
        return ResultArray._from_sorted(
            self._index,
            np.abs(self._data),
            self._cols,
            self._domain,
            self.label,
            self._data.dtype,
        )

    # ------------------------------------------------------------------
    # Comparison — return numpy bool arrays (aligned to common index)
    # ------------------------------------------------------------------

    def _cmp(self, other: _Operand, op):
        if isinstance(other, ResultArray):
            ds, do, _ = self._align_pair(other)
            return op(ds, do)
        return op(self._data, other)

    def __lt__(self, other):
        return self._cmp(other, np.less)

    def __le__(self, other):
        return self._cmp(other, np.less_equal)

    def __gt__(self, other):
        return self._cmp(other, np.greater)

    def __ge__(self, other):
        return self._cmp(other, np.greater_equal)

    def __eq__(self, other):
        if isinstance(other, ResultArray):
            ds, do, _ = self._align_pair(other)
            return ds == do
        return self._data == other

    def __ne__(self, other):
        if isinstance(other, ResultArray):
            ds, do, _ = self._align_pair(other)
            return ds != do
        return self._data != other

    def __hash__(self):
        # Unhashable — consistent with overriding __eq__
        return None

    # ------------------------------------------------------------------
    # Reductions (return numpy scalars or 1-D arrays)
    # ------------------------------------------------------------------

    def max(self, axis: int = 0) -> np.ndarray:
        """Maximum along *axis* (default: per-column across all entities)."""
        return np.max(self._data, axis=axis)

    def min(self, axis: int = 0) -> np.ndarray:
        return np.min(self._data, axis=axis)

    def mean(self, axis: int = 0) -> np.ndarray:
        return np.mean(self._data, axis=axis)

    def sum(self, axis: int = 0) -> np.ndarray:
        return np.sum(self._data, axis=axis)

    def std(self, axis: int = 0) -> np.ndarray:
        return np.std(self._data, axis=axis)

    def norm(self, axis: int = 1) -> np.ndarray:
        """Euclidean norm along *axis* (default: across columns per entity)."""
        return np.linalg.norm(self._data.astype(np.float64), axis=axis)

    def argmax(self, axis: int = 0) -> np.ndarray:
        """Index of maximum value along *axis*."""
        return np.argmax(self._data, axis=axis)

    def argmin(self, axis: int = 0) -> np.ndarray:
        return np.argmin(self._data, axis=axis)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def copy(self) -> ResultArray:
        """Return a deep copy."""
        return ResultArray._from_sorted(
            self._index.copy(),
            self._data.copy(),
            self._cols,
            self._domain,
            self.label,
            self._data.dtype,
        )

    def astype(self, dtype) -> ResultArray:
        """Return a copy with data cast to *dtype* (e.g. ``np.float64``)."""
        return ResultArray._from_sorted(
            self._index.copy(),
            self._data.astype(dtype),
            self._cols,
            self._domain,
            self.label,
            dtype,
        )

    def rename(self, mapping: dict) -> ResultArray:
        """
        Return a new ResultArray with columns renamed per *mapping*.

        >>> ra.rename({'VM': 'von_mises', 'P1': 'max_principal'})
        """
        new_cols = tuple(mapping.get(c, c) for c in self._cols)
        if len(set(new_cols)) != len(new_cols):
            raise ValueError("rename would produce duplicate column names")
        return ResultArray._from_sorted(
            self._index,
            self._data,
            new_cols,
            self._domain,
            self.label,
            self._data.dtype,
        )

    def describe(self) -> None:
        """Print per-column min / max / mean / std statistics."""
        n, m = self._data.shape
        tag = f"  [{self.label}]" if self.label else ""
        print(f"ResultArray{tag}  {n} {self._domain}(s) × {m} field(s)")
        w = max(len(c) for c in self._cols)
        hdr = f"  {'col':{w}s}  {'min':>14s}  {'max':>14s}  {'mean':>14s}  {'std':>14s}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for j, col in enumerate(self._cols):
            v = self._data[:, j].astype(np.float64)
            print(
                f"  {col:{w}s}  {v.min():14.6g}  {v.max():14.6g}"
                f"  {v.mean():14.6g}  {v.std():14.6g}"
            )

    def to_dataframe(self):
        """
        Convert to a pandas DataFrame with entity IDs as the integer index.

        The index name is ``'element_id'`` or ``'grid_id'`` depending on
        ``self.domain``.
        """
        import pandas as pd

        return pd.DataFrame(
            self._data,
            index=pd.Index(self._index, name=f"{self._domain}_id"),
            columns=list(self._cols),
        )

    # ------------------------------------------------------------------
    # Repr / str
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n, m = self._data.shape
        cols = ", ".join(self._cols)
        label_str = f" [{self.label}]" if self.label else ""
        idx_str = f"[{self._index[0]}…{self._index[-1]}]" if n > 0 else "[]"
        return (
            f"ResultArray{label_str}\n"
            f"  {n} {self._domain}(s) × {m} field(s)\n"
            f"  columns : {cols}\n"
            f"  index   : {idx_str}  dtype={self._data.dtype}"
        )


# ---------------------------------------------------------------------------
# Module-level utilities
# ---------------------------------------------------------------------------


def concat(arrays: List[ResultArray], *, label: str = "") -> ResultArray:
    """
    Concatenate multiple ResultArrays along the entity axis (union of IDs).

    All arrays must have the same *columns* and *domain*.  Duplicate entity
    IDs raise a ``ValueError``.

    Parameters
    ----------
    arrays : list of ResultArray
    label : str, optional
        Label for the returned array.

    Returns
    -------
    ResultArray
    """
    if not arrays:
        raise ValueError("concat requires at least one ResultArray")
    ref = arrays[0]
    for arr in arrays[1:]:
        if arr._cols != ref._cols:
            raise ValueError(f"column mismatch in concat: {ref._cols} vs {arr._cols}")
        if arr._domain != ref._domain:
            raise ValueError(
                f"domain mismatch in concat: {ref._domain!r} vs {arr._domain!r}"
            )

    combined_idx = np.concatenate([a._index for a in arrays])
    combined_dat = np.concatenate([a._data for a in arrays], axis=0)
    order = np.argsort(combined_idx, kind="stable")
    sorted_idx = combined_idx[order]

    if len(sorted_idx) > 1 and (sorted_idx[1:] == sorted_idx[:-1]).any():
        dupes = sorted_idx[np.where(sorted_idx[1:] == sorted_idx[:-1])[0][:5]]
        raise ValueError(f"concat: duplicate entity IDs found (e.g. {dupes.tolist()})")

    return ResultArray._from_sorted(
        sorted_idx,
        combined_dat[order],
        ref._cols,
        ref._domain,
        label or ref.label,
        ref._data.dtype,
    )


def stack(arrays: List[ResultArray], *, label: str = "") -> ResultArray:
    """
    Combine ResultArrays that share the same index but carry different columns.

    All arrays must have the same *domain*.  They are aligned to their common
    index before combining, so partial-mesh arrays are supported.  All column
    names across all arrays must be unique.

    Parameters
    ----------
    arrays : list of ResultArray
    label : str, optional

    Returns
    -------
    ResultArray
        Columns are the ordered union of all input columns.

    Example
    -------
    >>> vm  = ResultArray(eids, vm_vals,  ['VM'])
    >>> p1  = ResultArray(eids, p1_vals,  ['P1'])
    >>> all = stack([vm, p1])
    >>> all.columns
    ('VM', 'P1')
    """
    if not arrays:
        raise ValueError("stack requires at least one ResultArray")

    all_cols: List[str] = []
    for arr in arrays:
        for c in arr._cols:
            if c in all_cols:
                raise ValueError(f"stack: duplicate column name {c!r}")
            all_cols.append(c)

    domain = arrays[0]._domain

    # Align all to common index
    common = arrays[0]._index
    for arr in arrays[1:]:
        common = np.intersect1d(common, arr._index)

    blocks = []
    for arr in arrays:
        ps = np.searchsorted(arr._index, common)
        blocks.append(arr._data[ps])

    return ResultArray._from_sorted(
        common,
        np.concatenate(blocks, axis=1),
        tuple(all_cols),
        domain,
        label,
        arrays[0]._data.dtype,
    )
