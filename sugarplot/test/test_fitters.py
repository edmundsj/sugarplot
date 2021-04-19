import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from pandas.testing import assert_frame_equal
from liapy import LIA
from sugarplot import weibull, fit_weibull, fit_lia

def test_weibull():
    value_actual = weibull(1, x0=1, beta=2)
    value_desired = 1 - np.e ** -1
    assert_equal(value_actual, value_desired)

    value_actual= weibull(1, x0=1/2, beta=2)
    value_desired = 1 - np.e ** -4
    assert_equal(value_actual, value_desired)

def test_fit_weibull_simple():
    beta_desired = beta = 2
    x0_desired = x0 = 2
    weibull_cdf = np.array([0.2, 0.4, 0.6, 0.8])
    weibull_xval = np.array([0.9447614541548776, 1.4294413227075684, 1.9144615241619822, 2.5372724823590396])

    fit_params, pcov, cdf = fit_weibull(weibull_xval)
    beta_actual, x0_actual = fit_params[0], fit_params[1]
    assert_equal(beta_actual, beta_desired)
    assert_equal(x0_actual, x0_desired)

def test_fit_weibull_real_data():
    xdata = np.array([ 35.223483, 66.50118585, 112.539044, 123.57383,
       125.52207671])
    ydata = np.array([0.16666667, 0.33333333, 0.5, 0.66666667, 0.83333333])
    fit_params, pcov, cdf = fit_weibull(xdata)
    beta_actual, x0_actual = fit_params[0], fit_params[1]
    beta_desired, x0_desired = 8.28563460099443, 118.86758906093989
    assert_equal(beta_actual, beta_desired)
    assert_equal(x0_actual, x0_desired)

def test_fit_weibull_pandas():
    data = pd.DataFrame({
            'random': [1, 2, 3, 4, 5],
            'Qbd': np.array([ 35.223483, 66.50118585, 112.539044, 123.57383, 125.52207671])})
    ydata = np.array([0.16666667, 0.33333333, 0.5, 0.66666667, 0.83333333])
    fit_params, pcov, cdf = fit_weibull(data)
    beta_actual, x0_actual = fit_params[0], fit_params[1]
    beta_desired, x0_desired = 8.28563460099443, 118.86758906093989
    assert_equal(beta_actual, beta_desired)
    assert_equal(x0_actual, x0_desired)

def test_fit_lia_data(lia, lia_data):
    n_points = 5
    phases_desired = np.pi*np.array([-1, -1/2, 0, 1/2, 1])
    fits_desired = 1 / np.sqrt(2) * np.array([0, -1, 0, 1, 0])
    data_desired = pd.DataFrame({
            'Phase (rad)': phases_desired,
            'val': fits_desired})
    data_actual, params_actual  = fit_lia(data=lia_data, n_points=n_points)
    assert_frame_equal(data_actual, data_desired, atol=1e-15)

def test_fit_lia_params(lia, lia_data):
    n_points = 5
    phases_desired = np.pi*np.array([-1, -1/2, 0, 1/2, 1])
    fits_desired = 1 / np.sqrt(2) * np.array([0, -1, 0, 1, 0])
    amp_desired = 1 / np.sqrt(2)
    phase_desired = np.pi/2
    data_actual, params_actual  = fit_lia(data=lia_data, n_points=n_points)
    assert_allclose(params_actual, (amp_desired, phase_desired))

def test_fit_lia_data_units(lia_units, lia_data_units):
    n_points = 5
    phases_desired = np.pi*np.array([-1, -1/2, 0, 1/2, 1])
    fits_desired = 1 / np.sqrt(2) * np.array([0, -1, 0, 1, 0])
    data_desired = pd.DataFrame({
            'Phase (rad)': phases_desired,
            'val (V)': fits_desired})
    data_actual, params_actual  = fit_lia(
            data=lia_data_units, n_points=n_points)
    assert_frame_equal(data_actual, data_desired, atol=1e-15)

def test_fit_lia_params_units(lia_units, lia_data_units):
    n_points = 5
    phases_desired = np.pi*np.array([-1, -1/2, 0, 1/2, 1])
    fits_desired = 1 / np.sqrt(2) * np.array([0, -1, 0, 1, 0])
    amp_desired = 1 / np.sqrt(2)
    phase_desired = np.pi/2
    data_actual, params_actual  = fit_lia(
            data=lia_data_units, n_points=n_points)
    assert_allclose(params_actual, (amp_desired, phase_desired))

def test_fit_lia_no_fit(lia_units, lia_data_units):
    n_points = 5
    phases_desired = np.pi*np.array([-1, -1/2, 0, 1/2, 1])
    fits_desired = 1 / np.sqrt(2) * np.array([0, -1, 0, 1, 0])
    amp_desired = None
    phase_desired = None
    data_actual, params_actual  = fit_lia(
            data=lia_data_units, n_points=n_points, fit=False)
    assert_equal(params_actual, (amp_desired, phase_desired))
