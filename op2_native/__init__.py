"""
op2_native — lightweight Nastran OP2 reader.

Decodes Nastran binary OP2 result files into pandas DataFrames,
one per subcase, with no external Nastran dependencies.

Supported result types
----------------------
displacements, stresses (shell/bar/solid/bush), strains,
element forces, SPC forces, applied loads, eigenvalues,
grid point weight.

Quick-start
-----------
>>> from op2_native import OP2
>>> op2 = OP2("model.op2")
>>> disp   = op2.displacements()   # {subcase_id: DataFrame}
>>> bstr   = op2.bar_stresses()    # {subcase_id: DataFrame}
>>> sol    = op2.solid_stresses()  # {subcase_id: DataFrame}
>>> r      = op2.results(1)        # Results object for subcase 1
>>> r.displacements                # plain DataFrame
"""

from .reader import OP2, Results
from . import plots

__all__ = ["OP2", "Results", "plots"]
