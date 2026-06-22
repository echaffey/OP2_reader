# op2_native/decoders/__init__.py
"""
Decoder subpackage — one module per Nastran result table type.

Module summary
--------------
ougv1        OUGV1  — displacements, velocities, accelerations
oes_search   Table-type classifier (OES / OEF / OQG / OPG / OSTR)
oes1x1_shell OES1X1 — shell element centroid and corner stresses
oes_bar      OES    — bar / beam stresses (CBAR, CBEAM)
oes_cbush    OES    — CBUSH spring deformations / stresses
oes_solid    OES    — solid element stresses (CTETRA, CHEXA, CPENTA)
oes_peek     EKEY / data-record location utilities
oef1         OEF1   — element forces (shell, bar/beam, CBUSH)
oqg1         OQG1   — SPC forces
opg1         OPG1   — applied load vector
ostr1        OSTR1  — shell element strains
ogpwg        OGPWG  — grid point weight generator
lama         LAMA   — real eigenvalues (SOL 103)
"""
