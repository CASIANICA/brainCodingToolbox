# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
from scipy.stats import gamma

def biGammaHRF(times):
    """Return values for HRF at given times.
    This HRF is derived from the sum of two Gamma function.
    Unit of time: second.
    """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6

def biGammaHRFSignal(end_time, unit):
    """Return a time course from time 0 to `end_time` generated from
    the biGammaHRF.
    Time unit is second. The time resolution is specified with
    parameter `unit`, for example, 1ms -> 0.001.
    """
    hrf_times = np.arange(0, end_time, unit)
    return biGammaHRF(hrf_times)

