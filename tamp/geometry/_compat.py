"""Compatibility shims for running pykin on Python 3.12 / NumPy.

pykin is unmaintained and assumes Python <3.10 (``from collections import
Iterable``). Importing this module *before* any pykin import restores the
removed ``collections`` ABC aliases. We deliberately use pykin only for FK and
the Jacobian (both verified working); IK is implemented in :mod:`robot_model`
because pykin's quaternion helpers return NaN on this stack.
"""
from __future__ import annotations

import collections
import collections.abc as _abc

for _name in ("Iterable", "Mapping", "MutableMapping", "Sequence", "Callable",
              "Set", "MutableSet", "Hashable", "Container", "Sized"):
    if not hasattr(collections, _name):
        setattr(collections, _name, getattr(_abc, _name))
