import pytest
from matplotlib.figure import Figure
import pandas as pd
import numpy as np

from sugarplot import normalize_pandas, default_plotter, reflectance_plotter, power_spectrum_plot
from sugarplot import assert_figures_equal, assert_axes_equal, assert_line_equal

@pytest.fixture
def data():
    xdata = np.array([1, 2, 3])
    ydata = np.array([1, 1/2, 1/3])
    xlabel = 'Time (ms)'
    ylabel = 'Frequency (Hz)'
    data = pd.DataFrame({
        xlabel: xdata, ylabel: ydata})
    return {'xdata': xdata, 'ydata': ydata, 'xlabel': xlabel,
        'ylabel': ylabel, 'data': data}

def test_plot_pandas_default(data):

    default_kw = {'xlabel': data['xlabel'], 'ylabel': data['ylabel']}
    desired_fig = Figure()
    desired_ax = desired_fig.subplots(subplot_kw=default_kw)
    desired_ax.plot(data['xdata'], data['ydata'])

    actual_fig, actual_ax = default_plotter(data['data'])
    assert_figures_equal(actual_fig, desired_fig)

def test_plot_pandas_log(data):
    desired_fig = Figure()
    log_kw= {
        'xlabel': data['xlabel'], 'ylabel': data['ylabel'],
        'xscale': 'log', 'yscale': 'log'}

    desired_ax = desired_fig.subplots(subplot_kw=log_kw)
    desired_ax.plot(data['xdata'], data['ydata'])
    actual_fig, actual_ax = default_plotter(data['data'],
            subplot_kw=log_kw)
    assert_figures_equal(actual_fig, desired_fig)

def test_plot_pandas_theory(data):
    def gaussian(x, a=1, mu=0, sigma=1):
        return a*np.exp(-np.square((x - mu)/(np.sqrt(2)*sigma)))

    subplot_kw= {'xlabel': data['xlabel'], 'ylabel': data['ylabel']}
    line_kw = {'linestyle': 'dashed'}
    theory_kw = {'a': 2, 'mu': 1, 'sigma': 3}
    theory_data = gaussian(data['xdata'], a=2, mu=1, sigma=3)

    desired_fig = Figure()
    desired_ax = desired_fig.subplots(subplot_kw=subplot_kw)
    desired_ax.plot(data['xdata'], data['ydata'])
    desired_ax.plot(data['xdata'], theory_data, **line_kw)
    actual_fig, actual_ax = default_plotter(data['data'],
        theory_func=gaussian, theory_kw=theory_kw)
    assert_figures_equal(actual_fig, desired_fig)

def test_plot_pandas_theory_data(data):
    default_kw = {'xlabel': data['xlabel'], 'ylabel': data['ylabel']}
    desired_fig = Figure()
    desired_ax = desired_fig.subplots(subplot_kw=default_kw)
    theory_line_kw = {'linestyle': 'dashed'}
    theory_data=pd.DataFrame({
            data['xlabel']: [0, 1, 2, 3],
            data['ylabel']: [2, 3, 4, 5],
            })

    desired_ax.plot(data['xdata'], data['ydata'])
    desired_ax.plot(theory_data[data['xlabel']],
            theory_data[data['ylabel']], **theory_line_kw)
    desired_ax.set_xlim(0.9, 3.3)

    actual_fig, actual_ax = default_plotter(data['data'],
            theory_data=theory_data)
    assert_figures_equal(actual_fig, desired_fig)

def test_reflectance_plotter():
    R_ref = pd.DataFrame({
            'Wavelength (nm)': np.arange(100, 150, 1),
            'Reflectance ()': np.linspace(0,1, 50)})
    I_ref = pd.DataFrame({
            'Wavelength (nm)': np.arange(100, 150, 5),
            'Photocurrent (nA)': np.linspace(1, 1, 10)})
    I_meas = pd.DataFrame({
            'Wavelength (nm)': np.linspace(110, 140,30),
            'Photocurrent (nA)': np.linspace(2, 2, 30)})
    R_1 = normalize_pandas(I_meas, I_ref, np.divide, new_name='R')
    R_2 = normalize_pandas(R_1, R_ref, np.multiply, new_name='R')
    fig_actual, ax_actual = reflectance_plotter(I_meas, I_ref, R_ref)
    fig_desired = Figure()
    ax_desired = fig_desired.subplots(
            subplot_kw={'ylabel': 'R', 'xlabel': 'Wavelength (nm)'})
    ax_desired.plot(R_2['Wavelength (nm)'], R_2['R'])
    assert_figures_equal(fig_actual, fig_desired)

def test_reflectance_plotter_theory_data():
    R_ref = pd.DataFrame({
            'Wavelength (nm)': np.arange(100, 150, 1),
            'Reflectance': np.linspace(0,1, 50)})
    I_ref = pd.DataFrame({
            'Wavelength (nm)': np.arange(100, 150, 5),
            'Photocurrent (nA)': np.linspace(1, 1, 10)})
    I_meas = pd.DataFrame({
            'Wavelength (nm)': np.linspace(110, 140,30),
            'Photocurrent (nA)': np.linspace(2, 2, 30)})
    R_1 = normalize_pandas(I_meas, I_ref, np.divide, new_name='R')
    R_2 = normalize_pandas(R_1, R_ref, np.multiply, new_name='R')
    fig_actual, ax_actual = reflectance_plotter(I_meas, I_ref, R_ref,
            theory_data=R_ref)

    fig_desired = Figure()

    ax_desired = fig_desired.subplots(
            subplot_kw={'ylabel': 'R', 'xlabel': 'Wavelength (nm)'})
    ax_desired.plot(R_2['Wavelength (nm)'], R_2['R'])
    ax_desired.plot(R_ref['Wavelength (nm)'], R_ref['Reflectance'],
            linestyle='dashed')

    assert_figures_equal(fig_actual, fig_desired)

def test_power_spectrum_plot():
    power_spectrum = pd.DataFrame({
            'Frequency (Hz)': [1, 2, 3],
            'Power (V ** 2)': [0.1, 0.1, 0.3]})
    fig_actual, ax_actual = power_spectrum_plot(power_spectrum)
    desired_fig = Figure()
    desired_ax = desired_fig.subplots()
    desired_ax.plot([1, 2, 3], 10*np.log10(np.array([0.1, 0.1, 0.3])))
    desired_ax.set_xlabel('Frequency (Hz)')
    desired_ax.set_ylabel('Power (dBV)')
    assert_figures_equal(fig_actual, desired_fig)

def test_power_spectrum_plot_psd():
    power_spectrum = pd.DataFrame({
            'Frequency (Hz)': [1, 2, 3],
            'Power (V ** 2 / Hz)': [0.1, 0.1, 0.3]})
    fig_actual, ax_actual = power_spectrum_plot(power_spectrum)
    desired_fig = Figure()
    desired_ax = desired_fig.subplots()
    desired_ax.plot([1, 2, 3], 10*np.log10(np.array([0.1, 0.1, 0.3])))
    desired_ax.set_xlabel('Frequency (Hz)')
    desired_ax.set_ylabel('Power (dBV/Hz)')
    assert_figures_equal(fig_actual, desired_fig)

