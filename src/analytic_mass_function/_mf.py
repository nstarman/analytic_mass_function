"""
Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

analytic_mass_function: Sample from analytic mass functions
"""


from __future__ import annotations

__all__ = ["MassFunctionDistribution"]

from typing import Any

import sympy as sp
from scipy import stats


class MassFunctionDistribution(stats.rv_continuous):  # type: ignore[misc]
    """Mass function distribution.

    Parameters
    ----------
    pdf : sympy.Expr
        The unnormalized PDF of the mass function.
        This must be a function of the symbol `m`, and optionally `mmin` and
        `mmax`. All symbols must be positive.

    mmin, mmax : float
        The minimum / maximum mass.
    """

    def __init__(
        self, pdf: sp.Expr, mmin: sp.Number, mmax: sp.Number, **kwargs: Any
    ) -> None:
        self.mmin = sp.Number(mmin)
        self.mmax = sp.Number(mmax)

        m = sp.Symbol("m", positive=True)
        m_min = sp.Symbol("mmin", positive=True)
        m_max = sp.Symbol("mmax", positive=True)
        subs = {m_min: mmin, m_max: mmax}

        # PDF
        self._unnorm_pdf = pdf
        normalization = sp.integrate(pdf, (m, m_min, m_max))
        self._pdf_normalization = normalization.evalf(subs={m_min: mmin, m_max: mmax})
        pdf_ = pdf / normalization
        pdf_expr = sp.Piecewise((0, m < m_min), (0, m >= m_max), (pdf_, True))
        self._pdf = sp.lambdify(m, pdf_expr.subs(subs), modules="numpy")

        # CDF
        cdf_ = sp.simplify(sp.integrate(pdf_, (m, m_min, m)))
        cdf_expr = sp.Piecewise((0, m < m_min), (1, m >= m_max), (cdf_, True))
        self._cdf = sp.lambdify(m, cdf_expr.subs(subs), modules="numpy")

        # Inverse CDF
        # unfortunately, sympy's solve function does not support the piecewise
        # functions, so we need to resolve the CDF.
        q = sp.Symbol("q", nonzer=True)
        ppf_ = sp.solve(sp.Eq(q, cdf_), m)[0]
        ppf = sp.Piecewise((sp.nan, q < 0), (sp.nan, q > 1), (ppf_, True))
        self._ppf = sp.lambdify(q, ppf.subs(subs), modules="numpy")

        super().__init__(a=float(mmin), b=float(mmax), **kwargs)
