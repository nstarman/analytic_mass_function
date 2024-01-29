"""
Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

analytic_mass_function: Sample from analytic mass functions
"""

from __future__ import annotations

__all__ = ["__version__", "MassFunctionDistribution"]

from ._mf import MassFunctionDistribution
from ._version import version as __version__
