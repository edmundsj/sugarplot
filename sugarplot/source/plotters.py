"""
Contains plotters for various types of datasets which require special plotting requirements.
"""
from matplotlib.figure import Figure
import sys, pathlib
from sugarplot import normalize_pandas, prettifyPlot, ureg, plt
from sciparse import to_standard_quantity, title_to_quantity
from scipy.optimize import curve_fit
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
            fig=fig, ax=ax,
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
            fig=fig, ax=ax,
            theory_func=theory_func, theory_kw=theory_kw,
            theory_x_data=theory_x_data, theory_y_data=theory_y_data,
            subplot_kw=subplot_kw,
            line_kw=line_kw)
    return fig, ax

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


def plot_weibull(data, fig=None, ax=None, line_kw={}, subplot_kw={}):
    """
    Plots a dataset to the best-fit Weibull distribution

    :param data: 1-D array-like data to be plotted
    :param fig: (optional) Figure to plot the data to
    :param ax: (optional) axes to plot the data to
    :param line_kw: Keyword arguments to pass into ax.plot()
    :param subplot_kw: Keyword arguments to pass into fig.subplots()

    """
    subplot_kw = dict(subplot_kw, xscale='log', yscale='log',
            ylabel='-ln(1-F)')
    fit_params, pcov, cdf = fit_weibull(data)
    weibull_kw = {'beta': fit_params[0], 'x0': fit_params[1]}

    def theory_func(x, **kwargs):
        return -np.log(1 - weibull(x, **kwargs))

    x_data = data
    y_data = -np.log(1 - cdf[1])

    fig, ax = default_plot_numpy(x_data, y_data,
            fig=fig, ax=ax, theory_func=theory_func,
            theory_kw=weibull_kw, subplot_kw=subplot_kw,
            line_kw=line_kw)
    return fig, ax

def show_figure(fig):
    """
    create a dummy figure and use its manager to display "fig"
    """

    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    plt.show()
