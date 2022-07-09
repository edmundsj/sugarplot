"""
Contains plotters for various types of datasets which require special plotting requirements.
"""
from matplotlib.figure import Figure
import sys, pathlib
from sugarplot import normalize_pandas, prettifyPlot, ureg, plt, weibull, fit_weibull, \
         fit_lia, fit_impedance, cmap
from sciparse import to_standard_quantity, title_to_quantity, column_from_unit, cname_from_unit
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
from warnings import warn

def default_plotter(data, y_data=None, fig=None, ax=None,
        theory_func=None, theory_args=(), theory_kw={},
        theory_data=None, theory_y_data=None, fit_func=None,
        fit_p0=None,
        line_kw={}, subplot_kw={}, plot_type='line',
        theory_name='Theory', **kwargs):
    """
    Default plotter which handles plotting pandas DataFrames, numpy arrays, and regular ol data.

    :param data: pandas DataFrame or array-like xdata
    :param y_data: array-like ydata
    :param theory_func: Function to plot along with xdata
    :param theory_kw: Keyword arguments to pass into theory_func
    :param theory_data: Theoretical data with same x/y axes as data
    :param line_kw: Keyword arguments to pass into ax.plot() function
    :param subplot_kw: Keyword arguments to pass into fig.subplots() function
    :param plot_type: Type of plot to generate. Current options are "line" and "scatter". Default is "line".
    """
    if isinstance(data, pd.DataFrame):
        return _default_plot_pandas(data, fig=fig, ax=ax,
                theory_func=theory_func, theory_kw=theory_kw,
                theory_args=theory_args,
                theory_name=theory_name,
                theory_data=theory_data,
                subplot_kw=subplot_kw, line_kw=line_kw,
                plot_type=plot_type,
                fit_func=fit_func,
                fit_p0=fit_p0)
    elif isinstance(data, np.ndarray):
        return _default_plot_numpy(data, y_data, fig=fig, ax=ax,
                theory_func=theory_func, theory_kw=theory_kw,
                theory_args=theory_args,
                theory_name=theory_name,
                theory_x_data=theory_data,
                theory_y_data=theory_y_data,
                subplot_kw=subplot_kw, line_kw=line_kw,
                plot_type=plot_type,
                fit_func=fit_func,
                fit_p0=fit_p0)
    else:
        raise ValueError(f'Plot not implemented for type {type(data)}. Only pandas.DataFrame is supported')

def _default_plot_pandas(data, theory_data=None,
        subplot_kw={}, **kwargs):
    """
    Plots a pandas DataFrame, assuming the xdata is located in the first column and the ydata is located in the second column. Not to be called directly.

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


    fig, ax = _default_plot_numpy(
            x_data, y_data,
            theory_x_data=theory_x_data,
            theory_y_data=theory_y_data,
            subplot_kw=subplot_kw,
            **kwargs)

    return fig, ax

def _default_plot_numpy(x_data, y_data, fig=None, ax=None,
        theory_func=None, theory_args=(), theory_kw={},
        theory_x_data=None, theory_y_data=None,
        subplot_kw={}, line_kw={}, theory_name='Theory',
        fit_func=None, fit_p0=None,
        plot_type='line'):

    if fig is None:
        fig = Figure()
    if ax is None:
        if len(fig.axes) == 0:
            ax = fig.subplots(subplot_kw=subplot_kw)
        elif len(fig.axes) >= 1:
            ax = fig.axes[0]
    if 'xlabel' in subplot_kw:
        if subplot_kw['xlabel'] == ax.get_xlabel() and \
            subplot_kw['ylabel'] != ax.get_ylabel():
            ax = ax.twinx()
            twinned=True
            if 'xlabel' in subplot_kw:
                ax.set_xlabel(subplot_kw['xlabel'])
            if 'ylabel' in subplot_kw:
                ax.set_ylabel(subplot_kw['ylabel'])
            line_kw = dict(color=cmap(1), **line_kw)
        else:
            ax.set_xlabel(subplot_kw['xlabel'])
            ax.set_ylabel(subplot_kw['ylabel'])

    if 'xscale' in subplot_kw:
        ax.set_xscale(subplot_kw['xscale'])
    if 'yscale' in subplot_kw:
        ax.set_yscale(subplot_kw['yscale'])

    if plot_type == 'line':
        ax.plot(x_data, y_data, **line_kw)
    elif plot_type == 'scatter':
        ax.scatter(x_data, y_data, **line_kw)
    else:
        raise ValueError(f'Plot type {plot_type} is unavailable. Only "line" and "scatter" are implemented')

    if fit_func is not None:
        theory_args, pcov = curve_fit(fit_func, x_data, y_data,
                p0=fit_p0)
        theory_func = fit_func
        print(f'Fit params: {theory_args}')
        print(f'Fit pcov: {pcov}')

    if theory_func:
        ax.plot(x_data, theory_func(x_data, *theory_args, **theory_kw),
           linestyle='dashed')
        if plot_type == 'line':
            ax.legend(['Measured', theory_name])
        else:
            ax.legend([theory_name, 'Measured'])

    if theory_x_data is not None and theory_y_data is not None:
        ax.plot(theory_x_data, theory_y_data,
           linestyle='dashed', **line_kw)
        if plot_type == 'line':
            ax.legend(['Measured', theory_name])
        else:
            ax.legend([theory_name, 'Measured'])
        if 'xlim' not in subplot_kw:
            xlim_lower = min(x_data) - abs(min(x_data))*0.1
            xlim_higher = max(x_data) + abs(max(x_data))*0.1
            ax.set_xlim(xlim_lower, xlim_higher)

    prettifyPlot(fig=fig)
    return fig, ax

def reflectance_plotter(
        photocurrent, reference_photocurrent, R_ref,
        subplot_kw={}, **kwargs):
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
    fig, ax = default_plotter(R_actual,
            subplot_kw=subplot_kw, **kwargs)
    return fig, ax

def power_spectrum_plot(power_spectrum, **kwargs):
    """
    Plots a given power spectrum.

    :param power_spectrum: Power spectrum pandas DataFrame with Frequency in the first column and power in the second column
    :returns fig, ax: Figure, axes pair for power spectrum plot

    """
    if isinstance(power_spectrum, pd.DataFrame):
        return _power_spectrum_plot_pandas(power_spectrum, **kwargs)
    else:
        raise NotImplementedError("Power spectrum plot not implemented" +
                                  f" for type {type(power_spectrum)}")

def _power_spectrum_plot_pandas(power_spectrum, subplot_kw={},
        theory_data=None, **kwargs):
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

    fig, ax = _default_plot_numpy(x_data, y_data,
            theory_x_data=theory_x_data, theory_y_data=theory_y_data,
            subplot_kw=subplot_kw, **kwargs)
    return fig, ax

def plot_weibull(data, subplot_kw={}, theory_name='Fit',
        plot_type='scatter', **kwargs):
    """
    Plots a dataset to the best-fit Weibull distribution

    :param data: 1-D array-like data to be plotted

    """
    if isinstance(data, pd.DataFrame):
        subplot_kw = dict(subplot_kw, xlabel=data.columns[-1])
        x_data = np.array(data.iloc[:,-1])
    else:
        x_data = data
    x_data = np.sort(x_data)

    subplot_kw = dict(subplot_kw, xscale='log', yscale='log',
            ylabel='-ln(1-F)')
    fit_params, pcov, cdf = fit_weibull(data)
    weibull_kw = {'beta': fit_params[0], 'x0': fit_params[1]}

    def theory_func(x, **kwargs):
        return -np.log(1 - weibull(x, **kwargs))

    y_data = -np.log(1 - cdf[1])

    fig, ax = _default_plot_numpy(x_data, y_data,
            theory_func=theory_func,
            theory_kw=weibull_kw, subplot_kw=subplot_kw,
            theory_name=theory_name, plot_type=plot_type,
            **kwargs)
    return fig, ax

def plot_lia(data, n_points=101, fit=True, **kwargs):
    """
    Plots a cosine-fitted curve to the phase delay used in a lock-in amplifier - used to find the exact offset phase given a known sequence of synchronization pulses.

    :param data: Input data as a pandas DataFrame which is compatible with liapy.
    :returns fig, ax: Figure and axes generated by this function
    """
    fitted_data, params = fit_lia(data, n_points=n_points, fit=fit)
    if fit:
        def theory_func(x, a=1, phase_delay=np.pi):
            return a * np.cos(x - phase_delay)
        theory_kw = {'a': params[0], 'phase_delay': params[1]}
    else:
        theory_func = None
        theory_kw={}
    return default_plotter(
            fitted_data,
            theory_func=theory_func, theory_kw=theory_kw,
            plot_type='scatter', **kwargs)

def plot_fit(data, fit_func, **kwargs):
    xdata = data.iloc[:,0]
    ydata = data.iloc[:,1]
    params, pcov = curve_fit(fit_func, xdata, ydata)
    print(f"Fit params: {params}")

    def theory_func(x):
        return fit_func(x, *params)

    return default_plotter(
            data,
            theory_func=theory_func,
            plot_type='scatter', **kwargs)

def plot_impedance(data, fit=True,
        model='rc', model_config='series', **kwargs):
    """
    Plots the magnitude / phase of a set of impedance data.

    :param fit: Whether to attempt to fit an impedance model to the data
    :param model: Real/reactive model - options are "rc"
    :param model_config: Model configuration, either "series" or "parallel"
    """
# The challenge here is that we need to plot two things with the same x axis and different y axes: the magnitude and phase data. 
    if data.shape[1] == 2:
        data_complex = True
        z_data = column_from_unit(data, ureg.ohm).to(ureg.ohm).m
        phase_data = np.angle(z_data)
        magnitude_data = np.abs(z_data)
        data_to_plot = pd.DataFrame({
                'Frequency (Hz)': data.iloc[:,0],
                '|Z| (ohm)': magnitude_data,
                'Phase (deg)': phase_data * 180/np.pi
                })
    else:
        data_complex = False
        phase_data = column_from_unit(data, ureg.rad).to(ureg.deg).m
        phase_name = cname_from_unit(data, ureg.rad)
        data_to_plot = data.rename(columns={phase_name: 'Phase (deg)'})
        data_to_plot['Phase (deg)'] = phase_data

    if fit:
        _, _, impedance_func = fit_impedance(data)
        # This needs to return a functional form for the impedance vs. frequency in addition to the relevant parameters, as it's not clear
        freq_data = column_from_unit(data, ureg.Hz)
        min_log = np.log10(freq_data.min().to(ureg.Hz).m)
        max_log = np.log10(freq_data.max().to(ureg.Hz).m)
        freq_theory_data = np.logspace(min_log, max_log, 100)
        impedance_theory_data = impedance_func(freq_theory_data)
        mag_theory_data = abs(impedance_theory_data)
        phase_theory_data = np.angle(impedance_theory_data)*180/np.pi
        theory_data = pd.DataFrame({
                'Frequency (Hz)': freq_theory_data,
                'Z (ohm)': mag_theory_data,
                'Phase (deg)': phase_theory_data
                })

    else:
        theory_data = None

    theory_to_plot = None
    subplot_kw = {'xscale': 'log', 'yscale': 'log'}

    if theory_data is not None:
        theory_to_plot = theory_data.iloc[:,[0,1]]

    # Plot the magnitude data
    kwargs.update(subplot_kw=subplot_kw, theory_data=theory_to_plot,
            theory_name='|Z| (Fit)', plot_type='scatter')
    fig, ax = default_plotter(data_to_plot.iloc[:,[0,1]], **kwargs)

    subplot_kw = {'xscale': 'log', 'yscale': 'linear'}
    # Plot the phase data
    if theory_data is not None:
        theory_to_plot = theory_data.iloc[:,[0,-1]]
    kwargs.update(subplot_kw=subplot_kw, fig=fig, ax=ax, theory_data=theory_to_plot,
            theory_name='Phase (Fit)', plot_type='scatter')
    fig, ax = default_plotter(data_to_plot.iloc[:,[0,-1]], **kwargs)

    prettifyPlot(fig=fig, ax=ax)
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
