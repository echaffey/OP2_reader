# op2_native

A lightweight, dependency-minimal Nastran OP2 reader written in pure Python.  
Decodes results directly from the binary Fortran unformatted records into
**pandas DataFrames**, one per subcase.

---

## Requirements

| Package | Minimum version |
|---------|----------------|
| Python  | 3.10 |
| numpy   | 1.24 |
| pandas  | 2.0 |
| plotly  | 5.0 *(optional — only for `op2_native.plots`)* |

---

## Installation

No packaging metadata is required.  Clone or copy the `op2_native/` directory
next to your script, then import directly:

```python
from op2_native import OP2
```

---

## Quick start

```python
from op2_native import OP2

op2 = OP2("model.op2")

# All result methods return {subcase_id: DataFrame}
disp   = op2.displacements()    # {1: DataFrame, 2: DataFrame, ...}
bstr   = op2.bar_stresses()
sol    = op2.solid_stresses()
forces = op2.element_forces()

# Access a single subcase using the Results helper
r = op2.results(subcase=1)
print(r.displacements)          # plain DataFrame
print(r.bar_stresses)

# Worst-case envelope across all subcases
env = op2.envelope(result="solid_stresses", column="VM", mode="absmax")

# Export everything to CSV
paths = op2.to_csv(output_dir="results/")
```

---

## Result methods

All methods cache on first call; use `op2.clear_cache()` to force a re-read.

### Nodal results

| Method | Columns | Description |
|--------|---------|-------------|
| `displacements()` | `GRID, TX, TY, TZ, RX, RY, RZ` | OUGV1 translations & rotations |
| `spc_forces()` | `GRID, FX, FY, FZ, MX, MY, MZ` | OQG1 single-point constraint forces |
| `applied_loads()` | `GRID, FX, FY, FZ, MX, MY, MZ` | OPG1 applied load vector |
| `grid_weight()` | — | OGPWG mass/CG/inertia (returns a dict, not a dict-of-DataFrames) |

### Shell element results

| Method | Key columns | Description |
|--------|-------------|-------------|
| `stresses()` | `EID, FD1, SX1, SY1, TXY1, ANG1, MAX_PRIN1, MIN_PRIN1, VON_MISES1, …FD2…` | OES1X1 centroid stresses (2 fibers) |
| `stresses_with_corners()` | same + corner-node rows | Shell stresses at element corners |
| `strains()` | similar to stresses | OSTR1 shell strains |
| `element_forces()` | element-type dependent | OEF1 shell + beam forces |

### Beam / bar element results (CBEAM / CBAR)

| Method | Columns | Description |
|--------|---------|-------------|
| `bar_stresses()` | `EID, GRID, SD, SXC, SXD, SXE, SXF, SMAX, SMIN, MS_T, MS_C` | OES1 per-station fiber stresses |
| `element_forces()` | `EID, GRID, SD, BM1, BM2, WS1, WS2, AF, TRQ` | OEF1 per-station forces |

Column meanings:

| Column | Meaning |
|--------|---------|
| `SD` | Station distance along beam (0.0 = end A, 1.0 = end B) |
| `SXC/SXD/SXE/SXF` | Combined axial + bending stress at fiber recovery points C, D, E, F |
| `SMAX / SMIN` | Max / min across the four fiber points |
| `MS_T / MS_C` | Margin of safety in tension / compression |
| `BM1 / BM2` | Bending moment in planes 1 and 2 |
| `WS1 / WS2` | Web shear force in planes 1 and 2 |
| `AF` | Axial force |
| `TRQ` | Torsional moment |

### CBUSH spring element results

| Method | Columns | Description |
|--------|---------|-------------|
| `bush_stresses()` | `EID, EX, EY, EZ, ETX, ETY, ETZ` | OES CBUSH deformations / strains |
| `bush_forces()` | `EID, FX, FY, FZ, MX, MY, MZ` | OEF CBUSH forces and moments |

### Solid element results (CTETRA / CHEXA / CPENTA)

| Method | Columns | Description |
|--------|---------|-------------|
| `solid_stresses()` | `EID, GRID, SX, SY, SZ, SXY, SYZ, SZX, VM` | OES solid stresses; `GRID=0` → centroid, `GRID!=0` → corner node |

### Modal results

| Method | Columns | Description |
|--------|---------|-------------|
| `eigenvalues()` | `MODE, ORDER, EIGENVALUE, RADIANS, CYCLES, GENM, GENSTIF` | LAMA real eigenvalues |

---

## Utility methods

```python
op2.subcases()          # DataFrame: subcase × result-type row-count table
op2.describe()          # Multi-index stats across all result types
op2.find_by_eid(eid)    # All element results for one element ID
op2.find_by_grid(gid)   # All nodal results for one grid ID
op2.summary()           # Low-level record inventory (offset, length, table name)
op2.to_csv("out/")      # Export all decoded tables to CSV files
op2.envelope(result="solid_stresses", column="VM", mode="absmax")
```

---

## Results object

`op2.results(subcase)` returns a `Results` instance with plain DataFrame
attributes — no dict subscripting needed:

```python
r = op2.results(1)
r.displacements       # DataFrame
r.bar_stresses        # DataFrame
r.solid_stresses      # DataFrame
r.bush_stresses       # DataFrame
r.bush_forces         # DataFrame
r.element_forces      # DataFrame
r.spc_forces          # DataFrame
r.applied_loads       # DataFrame
r.eigenvalues         # DataFrame
```

---

## Plotting (optional)

Requires `plotly`.

```python
from op2_native import plots

fig = plots.plot_vm_stress(op2.stresses()[1])
fig = plots.plot_displacement_magnitude(op2.displacements()[1])
fig = plots.plot_element_forces(op2.element_forces()[1], component="BM1")
fig = plots.plot_stress_histogram(op2.stresses()[1], column="VON_MISES1")
fig.show()
```

---

## Package layout

```
op2_native/
├── __init__.py          # Public exports: OP2, Results, plots
├── reader.py            # OP2 and Results classes
├── models.py            # OP2Inventory, OP2Record data classes
├── fortran_io.py        # Fortran unformatted binary reader
├── op2_reader.py        # Low-level record scanner (phase 1 peeker)
├── plots.py             # Optional Plotly visualisation helpers
└── decoders/
    ├── ougv1.py         # Displacements / velocities / accelerations
    ├── oes_search.py    # Table-type classifier (OES / OEF / OQG / OPG / OSTR)
    ├── oes1x1_shell.py  # Shell element stresses
    ├── oes_bar.py       # Bar / beam stresses (CBAR, CBEAM)
    ├── oes_cbush.py     # CBUSH spring stresses
    ├── oes_solid.py     # Solid element stresses (CTETRA, CHEXA, CPENTA)
    ├── oes_peek.py      # EKEY / data-record location utilities
    ├── oef1.py          # Element forces (OEF1)
    ├── oqg1.py          # SPC forces
    ├── opg1.py          # Applied loads
    ├── ostr1.py         # Shell strains
    ├── ogpwg.py         # Grid point weight generator
    └── lama.py          # Real eigenvalues
```

---

## Supported Nastran result table types

| OP2 table | Result | Method |
|-----------|--------|--------|
| OUGV1 | Displacements | `displacements()` |
| OES1X1 | Shell stresses | `stresses()` |
| OES1X1 (corners) | Shell corner stresses | `stresses_with_corners()` |
| OES (CBEAM/CBAR) | Bar/beam stresses | `bar_stresses()` |
| OES (CBUSH) | Bush stresses | `bush_stresses()` |
| OES (solid) | Solid stresses | `solid_stresses()` |
| OSTR1 | Shell strains | `strains()` |
| OEF1 | Element forces | `element_forces()`, `bush_forces()` |
| OQG1 | SPC forces | `spc_forces()` |
| OPG1 | Applied loads | `applied_loads()` |
| OGPWG | Grid point weight | `grid_weight()` |
| LAMA | Eigenvalues | `eigenvalues()` |
