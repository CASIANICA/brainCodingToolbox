# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import statsmodels.api as sm

def ols_fit(y, x):
    """Return the R-squared value of the OLS fitted model."""
    x = sm.add_constant(x)
    res = sm.OLS(y, x).fit()
    return res.rsquared


