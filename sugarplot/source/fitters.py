"""
Contains plotters for various types of datasets which require special plotting requirements.
"""
from sugarplot import ureg
from scipy.optimize import curve_fit, least_squares
from sciparse import to_standard_quantity, title_to_quantity, column_from_unit
from liapy import LIA
import pandas as pd
import numpy as np
import pint
from warnings import warn

def curve_fit_multi(funcs, xdata, ydata, p0, **kwargs):
    """
    Fits a set of curves with common parameters and input data

    :param funcs: Array-like list af functions where ydata[i] = f[i](xdata, p0)
    :param xdata: input data
    :param ydata: Array-like list of ydata. Should have same dimensionality as funcs.
    :param p0: Initial guess for parameter set
    """
    def cost_function(p0):
        residuals = np.array([func(xdata, *p0) - ydata for func, ydata in zip(funcs, ydata)]).flatten()
        return residuals
    result = least_squares(cost_function, p0, **kwargs)
    params = result.x
    return params

def weibull(x, beta=1, x0=1):
    """
    Weibull distribution
    """
    return 1 - np.exp(-np.power(x/x0, beta))

def transformed_weibull(x, beta, x0):
    """
    - ln (1 - F) of weibull distribution for curve-fitting
    """
    return np.power(x/x0, beta)

def fit_weibull(data):
    """
    Fits a dataset to a weibull distribution by computing the CDF of the dataset, manipulating it appropriately, and fitting to it.

    :param data: 1-dimensional array-like data to fit to. i.e. breakdown field or breakdown charge
    """
    if isinstance(data, pd.DataFrame):
        data = np.array(data.iloc[:,-1])

    data = np.sort(data)
    # Correct the CDF for finite size
    new_data = np.append(data, data[-1])

    data_cdf = []
    for i in range(len(new_data)):
        data_cdf.append([new_data[i], (i+1)/len(new_data)])

    data_cdf = np.transpose(np.array(data_cdf))
    data_cdf_unbiased = data_cdf[:,:-1]
    failure_quantity = data_cdf_unbiased[0]
    cdf = data_cdf_unbiased[1]
    cdf_transformed = - np.log(1 - cdf)
    fit_params, pcov = curve_fit(transformed_weibull, failure_quantity, cdf_transformed)
    print(f'Fit data to beta: {fit_params[0]}: x0: {fit_params[1]}')

    return (fit_params, pcov, data_cdf_unbiased)

def fit_lia(data, n_points=101, fit=True):
    """
    Generates amplitude vs. phase for lock-in-amplifier type data. Optionally fits that phase to a cosine function and returns the fit parameters.

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

    if fit:
        try:
            (amp, phase), pcov = curve_fit(cos_func, phase_delays, all_values)
            if amp < 0:
                amp *= -1
                phase -= np.pi
            phase = np.mod(phase, 2*np.pi)
        except RuntimeError as e:
            breakpoint()
    else:
        (amp, phase) = (None, None)
    full_data = pd.DataFrame({
            'Phase (rad)': phase_delays,
            ylabel: all_values
            })
    return full_data, (amp, phase)

def fit_impedance(data, model='rc', model_config='series', p0=(1, 1)):
    """
    Fits a two-element (resistive and reactive) impedance to an impedance spectrum

    :param data: Pandas dataframe with magnitude/phase impedance data
    :param model: Model to use. Options are "rc", "r", "c".
    :param model_config: "series" or "parallel"
    :param p0: Initial guesses in the form of (r0, c0) or in general (real, reactive)
    """
    # Convert R to kOhm internally and C to nF
    frequency_data = column_from_unit(data, ureg.Hz).to(ureg.Hz).m
    if data.shape[1] == 2: # Assume our data is complex - it has to be.
        z_data = column_from_unit(data, ureg.ohm).to(ureg.ohm).m
        magnitude_data = abs(z_data)
        phase_data = np.angle(z_data)
    else:
        magnitude_data = (column_from_unit(data, ureg.ohm).to(ureg.ohm)).m
        phase_data = column_from_unit(data, ureg.rad).to(ureg.rad).m


    rscale = 1 # expected scale of resistances
    cscale = 1e-9 # expected scale of capacitances

    if model_config == 'series':
        def magnitude_func(f, *rc):
            return np.sqrt(np.square(rscale*rc[0]) + 1 / np.square((2*np.pi * f * (cscale*rc[1]))))
        def phase_func(f, *rc):
            return - np.arctan(1 / (2*np.pi * f * (rscale*rc[0]) * (cscale*rc[1])))

    elif model_config == 'parallel':
        def magnitude_func(f, *rc):
            return rc[0] / np.sqrt(1 + np.square(2*np.pi * f * (rscale*rc[0]) * (cscale*rc[1]) ))
        def phase_func(f, *rc):
            return - np.arctan(2*np.pi*f*(rscale*rc[0])*(cscale*rc[1]))

    else:
        raise NotImplementedError

    def magnitude_func_log(f, *rc):
        return np.log(magnitude_func(f, *rc))

    params = curve_fit_multi(
       [magnitude_func_log, phase_func],
       frequency_data, [np.log(magnitude_data), phase_data], p0,
       bounds=[(0, 0), (np.inf, np.inf)])

    impedance_function = lambda f: magnitude_func(f, *params) * \
                         np.power(np.e, 1j*phase_func(f, *params))

    r_true = params[0] * rscale
    c_true = params[1] * cscale
    return r_true, c_true, impedance_function
