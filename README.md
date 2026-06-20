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
disp   = op2.displacements()
sstr   = op2.solid_stresses(location="centroid")
shstr  = op2.stresses()
forces = op2.element_forces()

# Access a single subcase using the Results helper
r = op2.results(subcase=1)
print(r.displacements)          # plain DataFrame
print(r.solid_stresses)

# Worst-case envelope across all subcases
env = op2.envelope(result="solid_stresses", column="VON_MISES", mode="absmax")

# Export everything to CSV
paths = op2.to_csv(output_dir="results/")
```

---

## Result methods

All methods cache on first call; use `op2.clear_cache()` to force a re-read.

### Nodal results

| Method | Columns | OP2 table | Description |
|--------|---------|-----------|-------------|
| `displacements()` | `GRID, TX, TY, TZ, RX, RY, RZ` | OUGV1 | Translations and rotations |
| `spc_forces()` | `GRID, FX, FY, FZ, MX, MY, MZ` | OQG1 | Single-point constraint reaction forces |
| `applied_loads()` | `GRID, FX, FY, FZ, MX, MY, MZ` | OPG1 | Applied load vector |
| `contact_forces()` | `GRID, FX, FY, FZ, MX, MY, MZ` | OQGCF1 | Contact interface forces |
| `initial_separation()` | `GRID, DISTANCE` | OSPDSI1 | Initial contact gap distance |
| `deformed_separation()` | `GRID, DISTANCE` | OSPDS1 | Deformed contact gap distance |
| `grid_weight()` | *(dict, not DataFrame)* | OGPWG | Total mass, CG, inertia matrix |

---

### Shell element results — linear (CQUAD4 / CTRIA3 / CQUAD8 / CTRIA6 / CQUADR / CTRIAR)

#### Stresses

```python
op2.stresses(location="max")      # default: element-wise max of corner nodes
op2.stresses(location="centroid") # centroid row only
op2.stresses_with_corners()       # centroid + all corner nodes (GRID column included)
```

| Method | `location` | Columns |
|--------|-----------|---------|
| `stresses()` | `"max"` or `"centroid"` | `EID, FD1, SX1, SY1, TXY1, ANG1, MAX_PRIN1, MIN_PRIN1, VON_MISES1, FD2, SX2, SY2, TXY2, ANG2, MAX_PRIN2, MIN_PRIN2, VON_MISES2` |
| `stresses_with_corners()` | — | `EID, GRID` + all stress columns above |
| `stress_tensors(location)` | `"max"` or `"centroid"` | `EID, SX1, SY1, TXY1, SX2, SY2, TXY2` |

Column meanings:

| Column | Meaning |
|--------|---------|
| `FD1 / FD2` | Fiber distance (Z offset) for bottom / top fiber layer |
| `SX1 / SX2` | Normal stress in X, fiber 1 / 2 |
| `SY1 / SY2` | Normal stress in Y, fiber 1 / 2 |
| `TXY1 / TXY2` | In-plane shear stress, fiber 1 / 2 |
| `ANG1 / ANG2` | Angle of principal axis |
| `MAX_PRIN1 / MIN_PRIN1` | Principal stresses (fiber 1) |
| `VON_MISES1 / VON_MISES2` | Von Mises stress per fiber |

When the OP2 was written centroid-only (`NUMWDE=17` or `19`), the `location`
parameter has no effect and centroid values are always returned.

When written with corner output (`NUMWDE=87`), `location="max"` takes the
element-wise maximum of each column across all four corner nodes.

#### Strains

```python
op2.strains()   # OSTR1 shell strains
```

| Method | Columns |
|--------|---------|
| `strains()` | `EID, FD1, EX1, EY1, EXY1, EANG1, EMAX_PRIN1, EMIN_PRIN1, EVON_MISES1, FD2, EX2, EY2, EXY2, EANG2, EMAX_PRIN2, EMIN_PRIN2, EVON_MISES2` |

---

### Solid element results — linear (CTETRA / CPENTA / CHEXA)

Supported element type codes: CTETRA (39, 85), CPENTA (67, 91), CHEXA (68, 93).

#### Location parameter

All solid stress and strain methods accept a `location` argument:

| `location` | Rows returned | `GRID` column |
|-----------|---------------|---------------|
| `"max"` *(default)* | One row per element — element-wise max of corner nodes | omitted |
| `"centroid"` | One row per element — centroid result (`GRID == 0`) | omitted |
| `"all"` | All rows: centroid (GRID=0) + every corner node (GRID>0) | included |

#### Stresses

```python
op2.solid_stresses()                    # default: max of corner nodes
op2.solid_stresses(location="centroid") # centroid only
op2.solid_stresses(location="all")      # centroid + all corner nodes
```

| Method | Columns |
|--------|---------|
| `solid_stresses(location)` | `EID[, GRID], SX, SY, SZ, SXY, SYZ, SZX, VON_MISES` |
| `solid_stress_tensors(location)` | `EID, SX, SY, SZ, SXY, SYZ, SZX` *(derived from `solid_stresses`)* |

The 3D symmetric stress tensor for an element row:

```
⎡ SX   SXY  SZX ⎤
⎢ SXY   SY  SYZ ⎥
⎣ SZX  SYZ   SZ ⎦
```

#### Strains

```python
op2.solid_strains()                          # OSTR1  — basic CS
op2.solid_strains(location="centroid")
op2.solid_strains_el()                       # OSTR1EL — element CS
op2.solid_strains_el(location="all")
```

| Method | OP2 table | Columns |
|--------|-----------|---------|
| `solid_strains(location)` | OSTR1 | `EID[, GRID], SX, SY, SZ, SXY, SYZ, SZX, VON_MISES` |
| `solid_strains_el(location)` | OSTR1EL | same |

---

### Bar / beam element results — linear (CBAR / CBEAM)

#### Stresses

```python
op2.bar_stresses()   # OES1X1 — CBAR one row/element; CBEAM one row/station
```

**CBAR** columns (element type 34, `NUMWDE=16`):

| Column | Meaning |
|--------|---------|
| `EID` | Element ID |
| `BEND1A..BEND4A` | Bending stress at four fiber points, end A |
| `AXIAL` | Axial stress |
| `SMAX_A / SMIN_A` | Max / min combined stress at end A |
| `MS_A` | Margin of safety in tension, end A |
| `BEND1B..BEND4B` | Bending stress at four fiber points, end B |
| `SMAX_B / SMIN_B` | Max / min combined stress at end B |
| `MS_C` | Margin of safety in compression, end B |

**CBEAM** columns (element type 2, `NUMWDE=111`, one row per station):

| Column | Meaning |
|--------|---------|
| `EID` | Element ID |
| `GRID` | Station grid ID |
| `SD` | Station distance (0.0 = end A, 1.0 = end B) |
| `SXC / SXD / SXE / SXF` | Combined axial + bending stress at recovery points C/D/E/F |
| `SMAX / SMIN` | Max / min across the four fiber points |
| `MS_T / MS_C` | Margin of safety in tension / compression |

#### Strains

```python
op2.bar_strains()      # OSTR1  — basic CS; same columns as CBEAM stresses
op2.bar_strains_el()   # OSTR1EL — element CS; same columns
```

---

### CBUSH spring element results

| Method | OP2 table | Columns |
|--------|-----------|---------|
| `bush_stresses()` | OES1X1 | `EID, EX, EY, EZ, ETX, ETY, ETZ` |
| `bush_strains()` | OSTR1 | `EID, EX, EY, EZ, ETX, ETY, ETZ` |
| `bush_strains_el()` | OSTR1EL | `EID, EX, EY, EZ, ETX, ETY, ETZ` |
| `bush_forces()` | OEF1 | `EID, FX, FY, FZ, MX, MY, MZ` |

---

### Shell and bar/beam element forces

```python
op2.element_forces()   # OEF1 — shells and bars combined
```

**CQUAD4 / CTRIA3** — one row per element for centroid-only output; centroid +
corner rows when `FORCE(CORNER)` was requested:

| Column | Meaning |
|--------|---------|
| `EID` | Element ID |
| `GRID` | 0 = centroid, >0 = corner grid ID *(corner output only)* |
| `NX / NY` | Membrane normal forces (force/length) |
| `NXY` | Membrane shear force (force/length) |
| `MX / MY` | Bending moments (force·length/length) |
| `MXY` | Twisting moment (force·length/length) |
| `QX / QY` | Transverse shear forces (force/length) |

**CBAR / CBEAM** — one row per element:

| Column | Meaning |
|--------|---------|
| `EID` | Element ID |
| `BM1A / BM2A` | Bending moment in planes 1 and 2, end A |
| `BM1B / BM2B` | Bending moment in planes 1 and 2, end B |
| `TS1 / TS2` | Transverse shear in planes 1 and 2 |
| `AF` | Axial force |
| `TRQ` | Torsional moment |

---

### CGAP element forces

```python
op2.gap_forces()   # OEF1 CGAP
```

| Column | Meaning |
|--------|---------|
| `EID` | Element ID |
| `COMP_X` | Compressive force in gap direction |
| `SHEAR_Y / SHEAR_Z` | Shear forces |
| `AXIAL_U` | Axial displacement |
| `TOTAL_V / TOTAL_W` | Total transverse displacements |
| `SLIP_V / SLIP_W` | Slip displacements |

---

### Nonlinear element results (OESNLXR)

Produced by MSC Nastran for elements with nonlinear material behaviour.

#### CBEAM nonlinear

```python
op2.nl_bar_stresses()   # one row per fiber per station per element
```

Columns: `EID, GRID, FIBER, STRESS, EQ_STRESS, TOTAL_STRAIN, EFF_STRAIN_PLAS, EFF_CREEP`

Two stations (A and B) × four fibers (C/D/E/F) = up to 8 rows per element.

#### CBUSH nonlinear

```python
op2.nl_bush_stresses()   # one row per element
```

Columns: `EID, FORCE_X, FORCE_Y, FORCE_Z, STRESS_TX, STRESS_TY, STRESS_TZ, STRAIN_TX, STRAIN_TY, STRAIN_TZ, MOMENT_X, MOMENT_Y, MOMENT_Z, STRESS_RX, STRESS_RY, STRESS_RZ, STRAIN_RX, STRAIN_RY, STRAIN_RZ`

#### CTETRA nonlinear

```python
op2.nl_solid_stresses()   # one row per node per element (centroid + 4 corners)
```

Columns: `EID, GRID, SX, SY, SZ, SXY, SYZ, SZX, VON_MISES, EFF_STRAIN_PLAS, EFF_CREEP, EX, EY, EZ, EXY, EYZ, EZX`

`GRID == 0` is the element centroid; `GRID > 0` is a corner node.

#### CQUAD4 / CTRIA3 nonlinear

```python
op2.nl_shell_stresses()   # two rows per element (bottom/top fiber)
op2.nl_shell_strains()    # two rows per element (bottom/top fiber)
```

| Method | Columns |
|--------|---------|
| `nl_shell_stresses()` | `EID, FIBER, FD, SX, SY, TXY, VON_MISES` |
| `nl_shell_strains()` | `EID, FIBER, FD, EX, EY, EXY, EFF_STRAIN_PLAS, EFF_CREEP` |

`FIBER=1` is the bottom layer; `FIBER=2` is the top layer.

---

### Modal and global results

| Method | Columns | OP2 table | Description |
|--------|---------|-----------|-------------|
| `eigenvalues()` | `MODE, ORDER, EIGENVALUE, RADIANS, CYCLES, GENM, GENSTIF` | LAMA | Real eigenvalues (SOL 103 / 105) |
| `grid_weight()` | *(dict)* | OGPWG | Total mass, CG, inertia — returns a dict, not a dict-of-DataFrames |
| `metadata()` | `SubcaseMeta` per subcase | — | ACODE, TCODE, TITLE, SUBTITLE, LABEL |

---

## Supported element types

| Category | Element | Nastran type codes |
|----------|---------|-------------------|
| Shell | CQUAD4 | 33, 73, 144 |
| Shell | CTRIA3 | 74 |
| Shell | CQUAD8 | 64 |
| Shell | CTRIA6 | 75 |
| Shell | CQUADR | 82 |
| Shell | CTRIAR | 70 |
| Solid | CTETRA | 39 (MSC), 85 (NX) |
| Solid | CPENTA | 67 (MSC), 91 (NX) |
| Solid | CHEXA | 68 (MSC), 93 (NX) |
| Bar/beam | CBAR | 34, 100 |
| Bar/beam | CBEAM | 2 |
| Spring | CBUSH | 102 |
| Gap | CGAP | 38 |
| Nonlinear | CTETRA NL | 85 (OESNLXR) |
| Nonlinear | CBEAM NL | 94 (OESNLXR) |
| Nonlinear | CBUSH NL | 226 (OESNLXR) |
| Nonlinear | CQUAD4 NL | 90 (OESNLXR) |
| Nonlinear | CTRIA3 NL | 88 (OESNLXR) |

---

## Utility methods

```python
op2.subcases()          # DataFrame: subcase × result-type row-count summary
op2.describe()          # Multi-index stats across all result types
op2.find_by_eid(eid)    # All element results for one element ID
op2.find_by_grid(gid)   # All nodal results for one grid ID
op2.summary()           # Low-level record inventory (offset, length, table name)
op2.to_csv("out/")      # Export all decoded tables to CSV files
op2.clear_cache()       # Discard all cached DataFrames

# Worst-case envelope across all subcases
env = op2.envelope(result="solid_stresses", column="VON_MISES", mode="absmax")
```

---

## Results object

`op2.results(subcase)` returns a `Results` instance with plain DataFrame
attributes — no dict subscripting needed:

```python
r = op2.results(1)
r.displacements        # DataFrame
r.stresses             # DataFrame
r.strains              # DataFrame
r.stresses_corners     # DataFrame (corner output)
r.solid_stresses       # DataFrame
r.solid_strains        # DataFrame
r.solid_strains_el     # DataFrame
r.bar_stresses         # DataFrame
r.bar_strains          # DataFrame
r.bar_strains_el       # DataFrame
r.bush_stresses        # DataFrame
r.bush_strains         # DataFrame
r.bush_forces          # DataFrame
r.gap_forces           # DataFrame
r.element_forces       # DataFrame
r.nl_bar_stresses      # DataFrame
r.nl_bush_stresses     # DataFrame
r.nl_solid_stresses    # DataFrame
r.nl_shell_stresses    # DataFrame
r.nl_shell_strains     # DataFrame
r.spc_forces           # DataFrame
r.applied_loads        # DataFrame
r.contact_forces       # DataFrame
r.initial_separation   # DataFrame
r.deformed_separation  # DataFrame
r.eigenvalues          # DataFrame
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
    ├── oes1x1_shell.py  # Shell element stresses (linear)
    ├── oes_bar.py       # Bar / beam stresses and strains (CBAR, CBEAM)
    ├── oes_cbush.py     # CBUSH spring stresses and strains
    ├── oes_solid.py     # Solid element stresses and strains (CTETRA, CHEXA, CPENTA)
    ├── oes_peek.py      # EKEY / data-record location utilities
    ├── oesnlxr.py       # Nonlinear element results (OESNLXR)
    ├── oef1.py          # Element forces (OEF1): shells, bars, CBUSH, CGAP
    ├── oqg1.py          # SPC / contact forces
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
| OES1X1 (shell) | Shell stresses | `stresses()`, `stresses_with_corners()` |
| OES1X1 (solid) | Solid stresses | `solid_stresses()` |
| OES1X1 (bar/beam) | Bar/beam stresses | `bar_stresses()` |
| OES1X1 (CBUSH) | Bush stresses | `bush_stresses()` |
| OSTR1 (shell) | Shell strains | `strains()` |
| OSTR1 (solid) | Solid strains | `solid_strains()` |
| OSTR1 (bar/beam) | Bar/beam strains | `bar_strains()` |
| OSTR1 (CBUSH) | Bush strains | `bush_strains()` |
| OSTR1EL (solid) | Solid strains, element CS | `solid_strains_el()` |
| OSTR1EL (bar/beam) | Bar/beam strains, element CS | `bar_strains_el()` |
| OSTR1EL (CBUSH) | Bush strains, element CS | `bush_strains_el()` |
| OEF1 (shell) | Shell element forces | `element_forces()` |
| OEF1 (bar/beam) | Bar/beam element forces | `element_forces()` |
| OEF1 (CBUSH) | Bush forces | `bush_forces()` |
| OEF1 (CGAP) | Gap forces | `gap_forces()` |
| OESNLXR (CBEAM) | Nonlinear beam stresses | `nl_bar_stresses()` |
| OESNLXR (CBUSH) | Nonlinear bush forces/stresses | `nl_bush_stresses()` |
| OESNLXR (CTETRA) | Nonlinear solid stresses + strains | `nl_solid_stresses()` |
| OESNLXR (CQUAD4/CTRIA3) | Nonlinear shell stresses | `nl_shell_stresses()` |
| OESNLXR (CQUAD4/CTRIA3) | Nonlinear shell strains | `nl_shell_strains()` |
| OQG1 | SPC / MPC forces | `spc_forces()` |
| OPG1 | Applied loads | `applied_loads()` |
| OQGCF1 | Contact forces | `contact_forces()` |
| OSPDSI1 | Initial separation distance | `initial_separation()` |
| OSPDS1 | Deformed separation distance | `deformed_separation()` |
| OGPWG | Grid point weight | `grid_weight()` |
| LAMA | Real eigenvalues | `eigenvalues()` |
