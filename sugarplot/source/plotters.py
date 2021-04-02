"""
Contains plotters for various types of datasets which require special plotting requirements.
"""
from matplotlib.figure import Figure
import sys, pathlib
from sugarplot import normalize_pandas, prettifyPlot, ureg
from sciparse import to_standard_quantity, title_to_quantity
import pandas as pd
import numpy as np

def default_plotter(data, fig=None, ax=None, ydata=None, theory_func=None, theory_kw={}, line_kw={}, subplot_kw={}):
    """
    Default plotter which handles plotting pandas DataFrames, numpy arrays, and regular ol data.

    :param data: pandas DataFrame or array-like xdata
    :param ydata: array-like ydata
    :param theory_func: Function to plot along with xdata
    :param theory_kw: Keyword arguments to pass into theory_func
    :param line_kw: Keyword arguments to pass into ax.plot() function
    :param subplot_kw: Keyword arguments to pass into fig.subplots() function
    :param kwargs: Additional keyword arguments, which will be passed into the ax.plot() function
    """
    if isinstance(data, pd.DataFrame):
        return default_plot_pandas(data, fig=fig, ax=ax,
                theory_func=theory_func, theory_kw=theory_kw,
                subplot_kw=subplot_kw, line_kw=line_kw)

def default_plot_pandas(data, fig=None, ax=None,
        theory_func=None, theory_kw={}, subplot_kw={},line_kw={}):
    """
    Plots a pandas DataFrame, assuming the xdata is located in the first column and the ydata is located in the second column.

    :param data: DataFrame to be plotted.
    :param fig: Figure to plot the data to
    :param ax: axes to plot the data to
    :param theory_func: Function to plot along with xdata, of the form theory_func(xdata, theory_kw)
    :param theory_kw: Keyword arguments to be passed into theory_func
    :param subplot_kw: Keyword arguments to be passed into fig.subplots()
    """
    if 'xlabel' not in subplot_kw.keys():
        subplot_kw = dict(subplot_kw, xlabel=data.columns[0])
    if 'ylabel' not in subplot_kw.keys():
        subplot_kw = dict(subplot_kw, ylabel=data.columns[1])
    if not fig:
        fig = Figure()
    if not ax:
        ax = fig.subplots(subplot_kw=subplot_kw)

    x_data = data.iloc[:, 0].values
    y_data = data.iloc[:, 1].values
    ax.plot(x_data, y_data, **line_kw)

    if theory_func:
        ax.plot(x_data, theory_func(x_data, **theory_kw),
           linestyle='dashed', **line_kw)
        ax.legend(['Measured', 'Theory'])

    return fig, ax

def reflectance_plotter(
        photocurrent, reference_photocurrent, R_ref,
        fig=None, ax=None, theory_func=None,
        theory_kw={}, subplot_kw={},line_kw={}):
    """
    Plotter which takes a photocurrent, normalizes it to a reference photocurrent, and multiplies that be the reference's known or theoretical reflectance.

    :param photocurrent: Pandas DataFrame of measured photocurrent vs. wavelength (or frequency)
    :param reference_photocurrent: Pandas DataFrame of measured photocurrent reflecting from a reference surface with a known reflectance
    :param R_ref: Pandas DataFrame of known reflectance of surface (theoretical or measured)
    :param fig: Optional figure to plot to. If empty, creates a figure.
    :param ax: Optional axes to plot to. If empty, creates a new axes
    :param theory_func: Theoretical reflectance function to plot alongside the measured reflectance
    :param theory_kw: Keyword arguments for theoretical plotting function
    :param subplot_kw: Keyword argumets to pass into the .subplots() function during Axes creation.
    :param line_kw: Keyword arguments to pass into the .plot() function during Line2D creation.
    """
    subplot_kw = dict({'ylabel': 'R', 'xlabel': photocurrent.columns[0]},
            **subplot_kw)
    if not fig:
        fig = Figure()
    if not ax:
        ax = fig.subplots(subplot_kw=subplot_kw)

    R_norm = normalize_pandas(photocurrent, reference_photocurrent, np.divide, new_name='R')
    R_actual = normalize_pandas(R_norm, R_ref, np.multiply, new_name='R')
    x_data = R_actual.iloc[:, 0].values
    y_data = R_actual.iloc[:, 1].values
    ax.plot(x_data, y_data, **line_kw)

    if theory_func:
        ax.plot(x_data, theory_func(x_data, **theory_kw),
           linestyle='dashed', **line_kw)
        ax.legend(['Measured', 'Theory'])

    return fig, ax

def power_spectrum_plot(
        power_spectrum, fig=None, ax=None,
        ydata=None, theory_func=None, theory_kw={},
        line_kw={}, subplot_kw={}):
    """
    Plots a given power spectrum.

    :param power_spectrum: Power spectrum pandas DataFrame with Frequency in the first column and power in the second column
    :param sampling_frequency: Sampling frequency the data was taken at
    :returns fig, ax: Figure, axes pair for power spectrum plot

    """
    if isinstance(power_spectrum, pd.DataFrame):
        return power_spectrum_plot_pandas(
            power_spectrum,
            fig=fig, ax=ax,
            theory_func=theory_func, theory_kw=theory_kw,
            line_kw=line_kw, subplot_kw=subplot_kw)
    else:
        raise NotImplementedError("Power spectrum plot not implemented" +
                                  f" for type {type(power_spectrum)}")

def power_spectrum_plot_pandas(
        power_spectrum, fig=None, ax=None,
        theory_func=None, theory_kw={},
        line_kw={}, subplot_kw={}):
    """
    Implementation of powerSpectrumPlot for a pandas DataFrame. Plots a given power spectrum with units in the form Unit Name (unit type), i.e. Photocurrent (mA).

    :param power_spectrum: The power spectrum to be plotted, with frequency bins on one column and power in the second column
    :param line_kw: Keyword arguments to parameterize line in call to ax.plot()
    :param fig: (optional) Figure to plot the data to
    :param ax: (optional) axes to plot the data to
    :param line_kw: Keyword arguments to pass into ax.plot()
    :param subplot_kw: Keyword arguments to pass into fig.subplots()
    :param theory_func: Theoretical PSD function
    :param theory_kw: Keyword arguments to pass into theory_func
    """

    subplot_kw = dict(
        subplot_kw, xlabel=power_spectrum.columns[0],
        ylabel=power_spectrum.columns[1])
    if not fig:
        fig = Figure()
    if not ax:
        ax = fig.subplots(subplot_kw=subplot_kw)

    frequency_label = power_spectrum.columns.values[0]
    power_label = power_spectrum.columns.values[1]
    power_quantity = title_to_quantity(power_label)
    standard_quantity = to_standard_quantity(power_quantity)
    if '/ hertz' in str(power_quantity):
        is_psd = True
        standard_quantity = to_standard_quantity(power_quantity*ureg.Hz)
    else:
        is_psd = False
        standard_quantity = to_standard_quantity(power_quantity)

    base_units = np.sqrt(standard_quantity).units
    x_data = power_spectrum[frequency_label].values

    if theory_func:
        ax.plot(x_data, theory_func(x_data, **theory_kw),
           linestyle='dashed', **line_kw)
        ax.legend(['Measured', 'Theory'])

    ax.plot(
        x_data,
        10*np.log10(standard_quantity.magnitude * \
        power_spectrum[power_label].values), **line_kw)

    y_label = 'Power (dB{:~}'.format(base_units)
    if is_psd:
        y_label += '/Hz'
    y_label += ')'
    ax.set_ylabel(y_label)
    ax.set_xlabel(frequency_label)
    prettifyPlot(ax=ax, fig=fig)
    return fig, ax
