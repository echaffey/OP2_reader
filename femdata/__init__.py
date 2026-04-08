"""
femdata — index-aligned FEM result arrays
==========================================

Core exports:
    ResultArray   2-D array of FEM results keyed by entity ID
    concat        stack multiple ResultArrays along the entity axis
    stack         combine ResultArrays with same index but different columns
"""

from .result_array import ResultArray, concat, stack

__all__ = ["ResultArray", "concat", "stack"]
