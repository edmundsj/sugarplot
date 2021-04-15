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

def test_fit_weibull():
    beta_desired = 1.0765117953238197
    x0_desired = 0.1960321525466744
    weibull_xval = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    test_data = np.array([0.03112801, 0.08555936, 0.15152679, 0.22351831, 0.2978115 ])

    fit_params, pcov, cdf = fit_weibull(test_data)
    beta_actual, x0_actual = fit_params[0], fit_params[1]
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
