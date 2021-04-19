import pytest
import numpy as np
import pandas as pd
from liapy import LIA


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

@pytest.fixture
def lia_data():
    time_data = [0, 1, 2, 3, 4, 5, 6]
    sin_data = [0, 1, 0, -1, 0, 1, 0]
    sync_data = [0, 1, 0, 0, 0, 1, 0]
    full_data = pd.DataFrame({
            'time (s)': time_data,
            'val': sin_data,
            'Sync': sync_data})
    return full_data

@pytest.fixture
def lia_data_units():
    time_data = [0, 1, 2, 3, 4, 5, 6]
    sin_data = [0, 1, 0, -1, 0, 1, 0]
    sync_data = [0, 1, 0, 0, 0, 1, 0]
    full_data = pd.DataFrame({
            'time (s)': time_data,
            'val (V)': sin_data,
            'Sync': sync_data})
    return full_data


@pytest.fixture
def lia(lia_data):
    lia1 = LIA(lia_data)
    return lia1

@pytest.fixture
def lia_units(lia_data_units):
    lia1 = LIA(lia_data_units)
    return lia1
