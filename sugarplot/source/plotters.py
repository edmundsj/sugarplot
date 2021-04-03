"""
Contains plotters for various types of datasets which require special plotting requirements.
"""
from matplotlib.figure import Figure
import sys, pathlib
from sugarplot import normalize_pandas, prettifyPlot, ureg
from sciparse import to_standard_quantity, title_to_quantity
import pandas as pd
import numpy as np

def default_plotter(data, fig=None, ax=None, ydata=None, theory_func=None, theory_kw={}, theory_data=None, line_kw={}, subplot_kw={}):
    """
    Default plotter which handles plotting pandas DataFrames, numpy arrays, and regular ol data.

    :param data: pandas DataFrame or array-like xdata
    :param ydata: array-like ydata
    :param theory_func: Function to plot along with xdata
    :param theory_kw: Keyword arguments to pass into theory_func
    :param theory_data: Theoretical data with same x/y axes as data
    :param line_kw: Keyword arguments to pass into ax.plot() function
    :param subplot_kw: Keyword arguments to pass into fig.subplots() function
    :param kwargs: Additional keyword arguments, which will be passed into the ax.plot() function
    """
    if isinstance(data, pd.DataFrame):
        return default_plot_pandas(data, fig=fig, ax=ax,
                theory_func=theory_func, theory_kw=theory_kw,
                theory_data=theory_data,
                subplot_kw=subplot_kw, line_kw=line_kw)
    else:
        raise ValueError(f'Plot not implemented for type {type(data)}. Only pandas.DataFrame is supported')

def default_plot_pandas(data, fig=None, ax=None,
        theory_func=None, theory_kw={}, theory_data=None,
        subplot_kw={},line_kw={}):
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

    if isinstance(theory_data, pd.DataFrame):
        theory_x_data = theory_data.iloc[:,0].values
        theory_y_data = theory_data.iloc[:,1].values
    else:
        theory_x_data = None
        theory_y_data = None

    x_data = data.iloc[:, 0].values
    y_data = data.iloc[:, 1].values

    fig, ax = default_plot_numpy(x_data, y_data,
            theory_func=theory_func, theory_kw=theory_kw,
            theory_x_data=theory_x_data, theory_y_data=theory_y_data,
            subplot_kw=subplot_kw,
            line_kw=line_kw)

    return fig, ax

def default_plot_numpy(x_data, y_data, fig=None, ax=None,
        theory_func=None, theory_kw={},
        theory_x_data=None, theory_y_data=None,
        subplot_kw={}, line_kw={}):

    if not fig:
        fig = Figure()
    if not ax:
        ax = fig.subplots(subplot_kw=subplot_kw)

    ax.plot(x_data, y_data, **line_kw)

    if theory_func:
        ax.plot(x_data, theory_func(x_data, **theory_kw),
           linestyle='dashed', **line_kw)
        ax.legend(['Measured', 'Theory'])

    if theory_x_data is not None and theory_y_data is not None:
        ax.plot(theory_x_data, theory_y_data,
           linestyle='dashed', **line_kw)
        ax.legend(['Measured', 'Theory'])
        xlim_lower = min(x_data) - abs(min(x_data))*0.1
        xlim_higher = max(x_data) + abs(max(x_data))*0.1
        ax.set_xlim(xlim_lower, xlim_higher)

    return fig, ax

def reflectance_plotter(
        photocurrent, reference_photocurrent, R_ref,
        fig=None, ax=None, theory_func=None, theory_data=None,
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
    fig, ax = default_plotter(R_actual, fig=fig, ax=ax,
            theory_func=theory_func, theory_kw=theory_kw,
            theory_data=theory_data,
            subplot_kw=subplot_kw, line_kw=line_kw)
    return fig, ax

def power_spectrum_plot(
        power_spectrum, fig=None, ax=None,
        ydata=None, theory_func=None, theory_kw={},theory_data=None,
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
            theory_data=theory_data,
            line_kw=line_kw, subplot_kw=subplot_kw)
    else:
        raise NotImplementedError("Power spectrum plot not implemented" +
                                  f" for type {type(power_spectrum)}")

def power_spectrum_plot_pandas(
        power_spectrum, fig=None, ax=None,
        theory_func=None, theory_kw={}, theory_data=None,
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

    ylabel = 'Power (dB{:~}'.format(base_units)
    if is_psd:
        ylabel += '/Hz'
    ylabel += ')'

    subplot_kw = dict(
        subplot_kw,
        xlabel=power_spectrum.columns[0],
        ylabel=ylabel)

    if not fig:
        fig = Figure()
    if not ax:
        ax = fig.subplots(subplot_kw=subplot_kw)

    x_data = power_spectrum[frequency_label].values
    y_data =  10*np.log10(standard_quantity.magnitude * \
        power_spectrum[power_label].values)

    if isinstance(theory_data, pd.DataFrame):
        theory_x_data = theory_data.iloc[:,0].values
        theory_y_data = theory_data.iloc[:,1].values
    else:
        theory_x_data = None
        theory_y_data = None

    fig, ax = default_plot_numpy(x_data, y_data,
            theory_func=theory_func, theory_kw=theory_kw,
            theory_x_data=theory_x_data, theory_y_data=theory_y_data,
            subplot_kw=subplot_kw,
            line_kw=line_kw)
    return fig, ax
