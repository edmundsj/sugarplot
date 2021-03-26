import pytest
import pandas as pd
import numpy as np
from sugarplot import interpolate, normalize_pandas, normalize_reflectance, ureg
from pandas.testing import assert_frame_equal

def test_normalize_pandas_simple_multiply():
    data1 = pd.DataFrame({
            'Time (ms)': [0, 1, 2, 3, 4, 5],
            'Current (nA)': [0, 0.1, 0.2, 0.3, 0.4, 0.5]})
    data2= pd.DataFrame({
            'Time (ms)': [0, 1, 2, 3, 4, 5],
            'Current (nA)': [
            0,
            0.16666666666666669,
            0.33333333333333337,
            0.5,
            0.6666666666666666,
            0.8333333333333335]})
    multiplied_data_desired =  pd.DataFrame({
            'Time (ms)': [0, 1, 2, 3, 4, 5],
            'power (nA ** 2)': [
            0,
            0.016666666666666669,
            0.06666666666666668,
            0.15,
            0.26666666666666666,
            0.41666666666666674]})
    multiplied_data_actual = normalize_pandas(data1, data2)
    assert_frame_equal(multiplied_data_actual, multiplied_data_desired)

def test_normalize_mul_integration():
    data1 = pd.DataFrame({
            'Time (ms)': [0, 1, 2, 3, 4, 5],
            'Current (nA)': [0, 0.1, 0.2, 0.3, 0.4, 0.5]})
    data2 = pd.DataFrame({
            'Time (ms)': [0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4],
            'Current (nA)': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]})
    multiplied_data_desired =  pd.DataFrame({
            'Time (ms)': [0, 1, 2, 3, 4, 5],
            'power (nA ** 2)': [
            0,
            0.016666666666666669,
            0.06666666666666668,
            0.15,
            0.26666666666666666,
            0.41666666666666674]})
    multiplied_data_actual = normalize_pandas(data1, data2)
    assert_frame_equal(multiplied_data_actual, multiplied_data_desired)

def test_normalize_div_integration():
    data1 = pd.DataFrame({
            'Time (ms)': [0, 1, 2, 3, 4, 5],
            'Current (nA)': [0, 0.1, 0.2, 0.3, 0.4, 0.5]})
    data2 = pd.DataFrame({
            'Time (ms)': [0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4],
            'Current (nA)': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]})
    multiplied_data_desired =  pd.DataFrame({
            'Time (ms)': [0, 1, 2, 3, 4, 5],
            'rel': [
            np.NaN,
            0.6,
            0.6,
            0.6,
            0.6,
            0.6]})
    multiplied_data_actual = normalize_pandas(data1, data2, operation=np.divide, new_name='rel')
    assert_frame_equal(multiplied_data_actual, multiplied_data_desired)

def test_normalize_reflectance():
    photocurrent = pd.DataFrame({
            'Wavelength (nm)': [1, 1.5, 2, 2.5, 3, 3.5],
            'Photocurrent (nA)': [1, 1, 1, 1, 1, 1]})
    reference_photocurrent = pd.DataFrame({
            'Wavelength (nm)': [1, 1.5, 2, 2.5, 3, 3.5],
            'Photocurrent (nA)': [2, 2, 2.0, 2, 2, 2]})
    reference_reflectance = pd.DataFrame({
            'Wavelength (nm)': [1, 1.5, 2, 2.5, 3, 3.5],
            'R ()': [0.3, 0.4, 0.5, 0.4, 0.3, 0.2]})
    reflectance_desired = pd.DataFrame({
            'Wavelength (nm)': [1, 1.5, 2, 2.5, 3, 3.5],
            'R': 0.5*np.array([0.3, 0.4, 0.5, 0.4, 0.3, 0.2])})
    reflectance_actual = normalize_reflectance(
            photocurrent, reference_photocurrent, reference_reflectance)
    assert_frame_equal(reflectance_actual, reflectance_desired)

def test_normalize_reflectance_units():
    photocurrent = pd.DataFrame({
            'Boolergs (pA)': [5, 6.5, 0, 2.5, 4, 3.5],
            'Wavelength (nm)': [1, 1.5, 2, 2.5, 3, 3.5],
            'Photocurrent (nA)': [1, 1, 1, 1, 1, 1]})
    reference_photocurrent = pd.DataFrame({
            'Wavelength (nm)': [1, 1.5, 2, 2.5, 3, 3.5],
            'Photocurrent (nA)': [2, 2, 2.0, 2, 2, 2]})
    reference_reflectance = pd.DataFrame({
            'Wavelength (nm)': [1, 1.5, 2, 2.5, 3, 3.5],
            'Reflectance ()': [0.3, 0.4, 0.5, 0.4, 0.3, 0.2]})
    reflectance_desired = pd.DataFrame({
            'Boolergs (pA)': [5, 6.5, 0, 2.5, 4, 3.5],
            'Wavelength (nm)': [1, 1.5, 2, 2.5, 3, 3.5],
            'R': 0.5*np.array([0.3, 0.4, 0.5, 0.4, 0.3, 0.2])})
    reflectance_actual = normalize_reflectance(
            photocurrent, reference_photocurrent, reference_reflectance,
            column_units=ureg.nm)
    assert_frame_equal(reflectance_actual, reflectance_desired)

def test_normalize_reflectance_extra_data():
    photocurrent = pd.DataFrame({
            'Wavelength (nm)': [1, 1, 2, 2, 3, 3],
            'Amplitude': [0.1, 1, 0.1, 1, 0.1, 1],
            'Photocurrent (nA)': [1, 1, 1, 1, 1, 1]})
    reference_photocurrent = pd.DataFrame({
            'Wavelength (nm)': [1, 1.5, 2, 2.5, 3, 3.5],
            'Photocurrent (nA)': [2, 2, 2.0, 2, 2, 2]})
    reference_reflectance = pd.DataFrame({
            'Wavelength (nm)': [1, 1.5, 2, 2.5, 3, 3.5],
            'R': [0.3, 0.4, 0.5, 0.4, 0.3, 0.2]})
    reflectance_desired = pd.DataFrame({
            'Wavelength (nm)': [1, 1, 2, 2, 3, 3],
            'Amplitude': [0.1, 1, 0.1, 1, 0.1, 1],
            'R': 0.5*np.array([0.3, 0.3, 0.5, 0.5, 0.3, 0.3])})
    reflectance_actual = normalize_reflectance(
            photocurrent, reference_photocurrent, reference_reflectance)
    assert_frame_equal(reflectance_actual, reflectance_desired)

def test_normalize_reflectance_extra_data_both():
    photocurrent = pd.DataFrame({
            'Wavelength (nm)': [1, 1, 2, 2, 3, 3],
            'Amplitude': [0.1, 1, 0.1, 1, 0.1, 1],
            'Photocurrent (nA)': [1, 1, 1, 1, 1, 1]})
    reference_photocurrent = pd.DataFrame({
            'Wavelength (nm)': [1, 1, 2, 2, 3, 3],
            'Amplitude': [0.1, 1, 0.1, 1, 0.1, 1],
            'Photocurrent (nA)': [2, 2, 2.0, 2, 2, 2]})
    reference_reflectance = pd.DataFrame({
            'Wavelength (nm)': [1, 1.5, 2, 2.5, 3, 3.5],
            'R': [0.3, 0.4, 0.5, 0.4, 0.3, 0.2]})
    reflectance_desired = pd.DataFrame({
            'Wavelength (nm)': [1, 1, 2, 2, 3, 3],
            'Amplitude': [0.1, 1, 0.1, 1, 0.1, 1],
            'R': 0.5*np.array([0.3, 0.3, 0.5, 0.5, 0.3, 0.3])})
    reflectance_actual = normalize_reflectance(
            photocurrent, reference_photocurrent, reference_reflectance)
    assert_frame_equal(reflectance_actual, reflectance_desired)

def test_interpolate():
    data1 = pd.DataFrame({
            'Time (ms)': [0, 1, 2, 3, 4, 5],
            'Current (nA)': [0, 0.1, 0.2, 0.3, 0.4, 0.5]})
    data2 = pd.DataFrame({
            'Time (ms)': [0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4],
            'Current (nA)': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]})
    interpolated_desired = pd.DataFrame({
            'Time (ms)': [0, 1, 2, 3, 4, 5],
            'Current (nA)': [
            0,
            0.16666666666666669,
            0.33333333333333337,
            0.5,
            0.6666666666666666,
            0.8333333333333335]})
    interpolated_actual = interpolate(data1, data2)
    assert_frame_equal(interpolated_actual, interpolated_desired)

def test_interpolate_units():
    data1 = pd.DataFrame({
            'Phlargen (mV)': [5, 4, 2, 4, 5, 6],
            'Time (ms)': [0, 1, 2, 3, 4, 5],
            'Current (nA)': [0, 0.1, 0.2, 0.3, 0.4, 0.5]})
    data2 = pd.DataFrame({
            'Time (ms)': [0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4],
            'Current (nA)': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]})
    interpolated_desired = pd.DataFrame({
            'Phlargen (mV)': [5, 4, 2, 4, 5, 6],
            'Time (ms)': [0, 1, 2, 3, 4, 5],
            'Current (nA)': [
            0,
            0.16666666666666669,
            0.33333333333333337,
            0.5,
            0.6666666666666666,
            0.8333333333333335]})
    interpolated_actual = interpolate(data1, data2, column_units=ureg.ms)
    assert_frame_equal(interpolated_actual, interpolated_desired)


def test_interpolate_big_data1():
    """
    Checks that we can handle data1 which is larger in length than data2
    """
    data1 = pd.DataFrame({
            'Time (ms)': [0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4],
            'Current (nA)': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]})
    data2 = pd.DataFrame({
            'Time (ms)': [0, 1, 2, 3, 4, 5],
            'Current (nA)': [0, 0.1, 0.2, 0.3, 0.4, 0.5]})
    interpolated_desired = pd.DataFrame({
            'Time (ms)': [0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4],
            'Current (nA)': [0, 0.06, 0.12, 0.18, 0.24, 0.3, 0.36, 0.42, 0.48, 0.5]})
    interpolated_actual = interpolate(data1, data2)
    assert_frame_equal(interpolated_actual, interpolated_desired)

def test_interpolate_with_offset():
    """
    Checks that we can handle data1 which is larger in length than data2
    """
    data1 = pd.DataFrame({
            'Time (ms)': [5, 5.6, 6.2, 6.8, 7.4, 8.0, 8.6, 9.2, 9.8, 10.4],
            'Current (nA)': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]})
    data2 = pd.DataFrame({
            'Time (ms)': np.array([5, 6, 7, 8, 9, 10]),
            'Current (nA)': [0, 0.1, 0.2, 0.3, 0.4, 0.5]})
    interpolated_desired = pd.DataFrame({
            'Time (ms)': [5, 5.6, 6.2, 6.8, 7.4, 8.0, 8.6, 9.2, 9.8, 10.4],
            'Current (nA)': np.array([0, 0.06, 0.12, 0.18, 0.24, 0.3, 0.36, 0.42, 0.48, 0.5])})
    interpolated_actual = interpolate(data1, data2)
    assert_frame_equal(interpolated_actual, interpolated_desired)

def test_interpolate_with_yoffset():
    """
    Checks that we can handle data1 which is larger in length than data2
    """
    data1 = pd.DataFrame({
            'Time (ms)': [5, 5.6, 6.2, 6.8, 7.4, 8.0, 8.6, 9.2, 9.8, 10.4],
            'Current (nA)': [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.65]})
    data2 = pd.DataFrame({
            'Time (ms)': np.array([5, 6, 7, 8, 9, 10]),
            'Current (nA)': [0.5, 0.45, 0.4, 0.35, 0.3, 0.25]})
    interpolated_desired = pd.DataFrame({
            'Time (ms)': [5, 5.6, 6.2, 6.8, 7.4, 8.0, 8.6, 9.2, 9.8, 10.4],
            'Current (nA)': np.array([0.5, 0.47, 0.44, 0.41, 0.38, 0.35, 0.32, 0.29, 0.26, 0.25])})
    interpolated_actual = interpolate(data1, data2)
    assert_frame_equal(interpolated_actual, interpolated_desired)

def test_interpolate_numpy():
    R_ref = pd.DataFrame({
            'Wavelength (nm)': np.arange(100, 151, 1),
            'Reflectance ()': np.linspace(0,1, 51)})
    I_ref = pd.DataFrame({
            'Wavelength (nm)': np.arange(100, 150, 5),
            'Photocurrent (nA)': np.linspace(2, 2, 10)})
    I_meas = pd.DataFrame({
            'Wavelength (nm)': np.linspace(110, 140,30),
            'Photocurrent (nA)': np.linspace(1, 1, 30)})
    R_1 = normalize_pandas(I_meas, I_ref, np.divide,
            new_name='Reflectance')
    R_actual = normalize_pandas(R_1, R_ref, np.multiply,
            new_name='Reflectance')
    R_desired = pd.DataFrame({
            'Wavelength (nm)': np.linspace(110, 140, 30),
            'Reflectance': 0.5*np.linspace(0.2, 0.8, 30)})
    assert_frame_equal(R_actual, R_desired)
