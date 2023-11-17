from __future__ import annotations

import importlib.metadata

import analytic_mass_function as m


def test_version():
    assert importlib.metadata.version("analytic_mass_function") == m.__version__
