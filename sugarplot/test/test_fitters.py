import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from pandas.testing import assert_frame_equal
from liapy import LIA
from sugarplot import weibull, fit_weibull, fit_lia, fit_impedance
from itertools import product

@pytest.fixture
def r_data():
    data = pd.DataFrame({
            'Frequency (Hz)': [1, 10, 100],
            'Z (ohm)': [1, 1, 1],
            'Theta (rad)': [0, 0, 0]})
    yield data

@pytest.fixture
def c_data_1nF():
    data = pd.DataFrame({
            'Frequency (Hz)': [1, 10, 100],
            'Z (ohm)': [1e9, 1e8, 1e7],
            'Theta (rad)': [-np.pi/2, -np.pi/2, -np.pi/2]})
    yield data

@pytest.fixture
def c_data_1F():
    data = pd.DataFrame({
            'Frequency (Hz)': [1, 10, 100],
            'Z (ohm)': [1, 0.1, 0.01],
            'Theta (rad)': [-np.pi/2, -np.pi/2, -np.pi/2]})
    yield data

@pytest.fixture
def rc_data_complex():
    data = pd.DataFrame({
            'Frequency (Hz)': [100, 1000, 10000],
            'Z (ohm)': [
                1000 - 1j*1.5915494309189536*1e6,
                1000 - 1j*1.5915494309189536*1e5,
                1000 - 1j*1.5915494309189536*1e4,
            ]})
    yield data


rvals = [1, 10, 100, 1000, 10000, 100000]
cvals = [1e-10, 1e-9, 1e-8, 1e-7]
rcvals = [x for x in product(rvals, cvals)]
@pytest.fixture(params=rcvals)
def rc_data_series(request):
    param = request.param
    r = param[0]
    c = param[1]
    frequencies = np.array([100, 1000, 10*1e3])
    magnitudes = np.abs(r + 1 / (1j*2*np.pi*frequencies*c))
    phases = - np.arctan(1 / (2*np.pi * frequencies* r *c))
    data = pd.DataFrame({
            'Frequency (Hz)': frequencies,
            'Z (ohm)': magnitudes,
            'Theta (rad)': phases})
    yield {'params': (r, c), 'data': data}

rvals = [1e4, 1e5, 1e6, 1e7, 5e7]
cvals = [1e-12, 1e-11, 1e-10, 1e-9, 1e-8]
rcvals = [x for x in product(rvals, cvals)]
@pytest.fixture(params=rcvals)
def rc_data_parallel(request):
    param = request.param
    res = param[0]
    cap = param[1]
    frequencies = np.array([100, 1000, 10*1e3])
    magnitudes = res / np.sqrt(1 + np.square(2*np.pi*frequencies*cap*res))
    phases = - np.arctan(2*np.pi * frequencies*cap*res)
    data = pd.DataFrame({
            'Frequency (Hz)': frequencies,
            'Z (ohm)': magnitudes,
            'Theta (rad)': phases})
    yield {'params': (res, cap), 'data': data}

@pytest.fixture
def rc_data_100_1nF():
    data = pd.DataFrame({
            'Frequency (Hz)': [100, 1000, 10000],
            'Z (ohm)': [1.59155*1e6, 159155., 15915.8],
            'Theta (rad)': [-1.5707334949419076, -1.570168008346862, -1.564513224169163]
            })
    yield data

def test_weibull():
    value_actual = weibull(1, x0=1, beta=2)
    value_desired = 1 - np.e ** -1
    assert_allclose(value_actual, value_desired)

    value_actual= weibull(1, x0=1/2, beta=2)
    value_desired = 1 - np.e ** -4
    assert_allclose(value_actual, value_desired)

def test_fit_weibull_simple():
    beta_desired = beta = 2
    x0_desired = x0 = 2
    weibull_cdf = np.array([0.2, 0.4, 0.6, 0.8])
    weibull_xval = np.array([0.9447614541548776, 1.4294413227075684, 1.9144615241619822, 2.5372724823590396])

    fit_params, pcov, cdf = fit_weibull(weibull_xval)
    beta_actual, x0_actual = fit_params[0], fit_params[1]
    assert_allclose(beta_actual, beta_desired)
    assert_allclose(x0_actual, x0_desired)

def test_fit_weibull_real_data():
    xdata = np.array([ 35.223483, 66.50118585, 112.539044, 123.57383,
       125.52207671])
    ydata = np.array([0.16666667, 0.33333333, 0.5, 0.66666667, 0.83333333])
    fit_params, pcov, cdf = fit_weibull(xdata)
    beta_actual, x0_actual = fit_params[0], fit_params[1]
    beta_desired, x0_desired = 8.28563460099443, 118.86758906093989
    assert_allclose(beta_actual, beta_desired, atol=1e-4)
    assert_allclose(x0_actual, x0_desired)

def test_fit_weibull_pandas():
    data = pd.DataFrame({
            'random': [1, 2, 3, 4, 5],
            'Qbd': np.array([ 35.223483, 66.50118585, 112.539044, 123.57383, 125.52207671])})
    ydata = np.array([0.16666667, 0.33333333, 0.5, 0.66666667, 0.83333333])
    fit_params, pcov, cdf = fit_weibull(data)
    beta_actual, x0_actual = fit_params[0], fit_params[1]
    beta_desired, x0_desired = 8.28563460099443, 118.86758906093989
    assert_allclose(beta_actual, beta_desired,  atol=1e-4)
    assert_allclose(x0_actual, x0_desired)

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

def test_fit_rc_r_only(r_data):
    actual_res, actual_cap, _ = \
         fit_impedance(r_data, model='rc',
                 model_config='parallel', p0=(10, 1))
    desired_res, desired_cap = [1, 0]
    actual_params = np.array([actual_res, actual_cap])
    desired_params = np.array([desired_res, desired_cap])

    assert_allclose(actual_params, desired_params, atol=1e-10)


def test_fit_rc_c_1nF(c_data_1nF):
    actual_res, actual_cap, _ = \
         fit_impedance(c_data_1nF, model='rc',
                 model_config='parallel', p0=(1, 1))
    desired_res, desired_cap = [1e13, 1e-9 / (2 * np.pi)]

    assert_allclose(actual_cap, desired_cap, atol=1e-22)

def test_fit_rc_series(rc_data_series):
    data = rc_data_series['data']
    actual_res, actual_cap, _ = \
         fit_impedance(data, model='rc',
                 model_config='series', p0=(1, 1))
    desired_res, desired_cap = rc_data_series['params']

    assert_allclose(actual_cap, desired_cap, rtol=1e-5)
    assert_allclose(actual_res, desired_res, rtol=1e-5)

def test_fit_rc_parallel(rc_data_parallel):
    data = rc_data_parallel['data']
    actual_res, actual_cap, impedance_func = \
         fit_impedance(data, model='rc',
                 model_config='parallel', p0=(1, 1))
    desired_res, desired_cap = rc_data_parallel['params']
    if abs(np.log10(2*np.pi*desired_res*desired_cap * 100)) > 5:
        rtol = 1e-1
    else:
        rtol = 1e-4

    assert_allclose(actual_cap, desired_cap, rtol=rtol)
    assert_allclose(actual_res, desired_res, rtol=rtol)

def test_fit_rc_series_complex(rc_data_complex):
    actual_res, actual_cap, impedance_func = \
         fit_impedance(rc_data_complex, model='rc',
                 model_config='series', p0=(1, 1))
    desired_res, desired_cap = 1000, 1e-9

    assert_allclose(actual_cap, desired_cap, rtol=1e-6)
    assert_allclose(actual_res, desired_res, rtol=1e-6)
