import pandas as pd
import numpy as np

def normalize_reflectance(photocurrent,
        reference_photocurrent, reference_reflectance):
    photocurrent_normalized = normalize_pandas(
            photocurrent, reference_photocurrent, operation=np.divide)
    reflectance = normalize_pandas(
            photocurrent_normalized, reference_reflectance,
            operation=np.multiply, new_units='R')
    return reflectance

def normalize_pandas(
        data1, data2, operation=np.multiply,
        operation_args=(), operation_kwargs={},
        new_units=''):
    """
    Performs an operation on data1 and data2, interpolating data from
    data2 as needed, and applying the operation specified (can be any function that takes two numpy arrays). New data will have indices and units of data1

    :param data1: Main data to be manipulated/plotted/used
    :param data2: Data to normalize / multiply / operate on relative to data1
    :param operation: operation function i.e. np.multiply. Operates such that new_data = data1 [OPERATION] data2, i.e. new_data = data1 / data2
    :param operation_args: Optional positional arguments to be passed into operation
    :param operation_kwargs: Optional keyword arguments to be passed into operation
    :param new_units: New column name for unit-laden dataFrame
    :returns new_data: pd.DataFrame = data1 [OPERATION] data2

    """
    if new_units: new_data_name = new_units
    else: new_data_name = data1.columns[-1]
    data2 = interpolate(data1, data2)
    data1_values = data1.iloc[:,-1].values
    data2_values = data2.iloc[:,-1].values
    normalized_values = operation(data1_values, data2_values,
            *operation_args, **operation_kwargs)
    normalized_data = pd.DataFrame({
            data1.columns[0]: data1.iloc[:,0].values})

    extra_columns = data1.shape[1] - 2
    for i in range(extra_columns):
        column_index = i+1
        normalized_data[data1.columns[column_index]] = \
               data1.iloc[:,column_index].values

    normalized_data[new_data_name] = normalized_values
    return normalized_data

def interpolate(data1, data2):
    """
    Interpolates data2 into the locations of data1. Requires data1 and data2 have no less than two values. If data1 and data2 have more than two columns, this will use the first and last columns.

    :param data1: Data you want to manipulate in the future
    :param data2: Data you want to interpolate to be happy with data1.
    :returns interpolated_data2: data2 which has been interpolated to match the x-indices of data1.
    """
    x1_name, y1_name = data1.columns[0], data1.columns[-1]
    x2_name, y2_name = data2.columns[0], data2.columns[-1]
    x1,y1 = data1[x1_name].values, data1[y1_name].values
    x2,y2 = data2[x2_name].values, data2[y2_name].values

    interpolated_data = np.interp(x1, x2, y2)

    interpolated_frame = pd.DataFrame({x1_name: x1})

    extra_columns = data1.shape[1] - 2
    for i in range(extra_columns):
        column_index = i+1
        interpolated_frame[data1.columns[column_index]] = \
               data1.iloc[:,column_index].values

    interpolated_frame[y2_name] = interpolated_data

    return interpolated_frame
