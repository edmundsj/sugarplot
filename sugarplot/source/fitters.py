"""
Contains plotters for various types of datasets which require special plotting requirements.
"""
from sugarplot import ureg
from scipy.optimize import curve_fit
from sciparse import to_standard_quantity, title_to_quantity
from liapy import LIA
import pandas as pd
import numpy as np
import pint
from warnings import warn

def weibull(x, beta=1, x0=1):
    """
    Weibull distribution
    """
    return 1 - np.exp(-np.power(x/x0, beta))

def fit_weibull(data):
    """
    Fits a dataset to a weibull distribution by computing the CDF of the dataset, manipulating it appropriately, and fitting to it.

    :param data: 1-dimensional array-like data to fit to. i.e. breakdown field or breakdown charge
    """
    # Correct the CDF for finite size
    new_data = np.append(data, data[-1])

    data_cdf = []
    for i in range(len(new_data)):
        data_cdf.append([new_data[i], (i+1)/len(new_data)])

    data_cdf = np.transpose(np.array(data_cdf))
    data_cdf_unbiased = data_cdf[:,:-1]
    fit_params, pcov = curve_fit(
            weibull, data_cdf_unbiased[0], data_cdf_unbiased[1])

    return (fit_params, pcov, data_cdf_unbiased)

def fit_lia(data, n_points=101):
    """
    Fits the lock-in amplifier phase of a dataset

    :param data: pandas DataFrame which contains a 'Sync' column with synchronization points
    """
    def cos_func(x, a=1, phase=0.1):
        return a*np.cos(x - phase)

    ylabel = data.columns[1]

    lia = LIA(data)
    phase_delays = np.linspace(-np.pi, np.pi, n_points)
    test_value = lia.extract_signal_amplitude(sync_phase_delay=0)
    extracted_v_np = np.vectorize(lia.extract_signal_amplitude)
    all_values = np.array([])
    for phase in phase_delays:
        retval = lia.extract_signal_amplitude(sync_phase_delay=phase)
        if isinstance(retval, pint.Quantity):
            retval = retval.m
        all_values = np.append(all_values, retval)

    (amp, phase), pcov = curve_fit(cos_func, phase_delays, all_values, bounds=((0, -np.pi), (np.inf, np.pi)))
    full_data = pd.DataFrame({
            'Phase (rad)': phase_delays,
            ylabel: all_values
            })
    return full_data, (amp, phase)
