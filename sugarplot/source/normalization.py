# THIS CODE WILL FAIL IF WE BLINDLY PASS IT TWO DATAFRAMES WE WANT TO
# NORMALIZE AND THEY HAVE REPEATED X-VALUES, AS IS THE CASE WHEN
# WE HAVE MULTIPLE CONDITIONS WE WANT TO TEST ALONG WITH A MASTER
# DATASET. WILL HAVE TO ADD TESTS CASES TO THIS.
# ALSO SHOULD NOT RELY ON THE ZEROTH COLUMN TO NORMALIZE REFLECTANCE,
# SHOULD CHECK FOR A COLUMN WITH THE UNITS OF WAVELENGTH.
import pandas as pd
import numpy as np
from sugarplot import ureg
from sciparse import cname_from_unit, title_to_quantity, quantity_to_title

def normalize_reflectance(photocurrent,
        reference_photocurrent, reference_reflectance,
        column_units=ureg.nm):
    """
    Converts a photocurrent spectra into a reflectance spectra using a reference photocurrent and known reference reflectance spectra.

    :param photocurrent: The target photocurrent to normaliize
    :param reference_photocurrent: A pandas DataFrame of a reference photocurrent as a 1-to-1 wavelength-to-reflectance table
    :param reference_reflectance: A pandas DataFrame of a referenence reflectance, with a 1-to-1 wavelength-to-reflectance mapping
    :param column_units: Which units to use for normalization. Defaults to length (nm)
    """
    photocurrent_normalized = normalize_pandas(
            photocurrent, reference_photocurrent,
            operation=np.divide, column_units=column_units)
    reflectance = normalize_pandas(
            photocurrent_normalized, reference_reflectance,
            operation=np.multiply, new_name='R',
            column_units=column_units)
    return reflectance

def normalize_pandas(
        data1, data2, operation=np.multiply,
        operation_args=(), operation_kwargs={},
        new_name='', column_units=None):
    """
    Performs an operation on data1 and data2, interpolating data from
    data2 as needed, and applying the operation specified (can be any function that takes two numpy arrays). New data will have indices and units of data1

    :param data1: Main data to be manipulated/plotted/used
    :param data2: Data to normalize / multiply / operate on relative to data1
    :param operation: operation function i.e. np.multiply. Operates such that new_data = data1 [OPERATION] data2, i.e. new_data = data1 / data2
    :param operation_args: Optional positional arguments to be passed into operation
    :param operation_kwargs: Optional keyword arguments to be passed into operation
    :param column_units: Which units to use for normalization. Defaults to None
    :returns new_data: pd.DataFrame = data1 [OPERATION] data2

    """
    data2 = interpolate(data1, data2, column_units=column_units)
    data1_name = data1.columns[-1]
    data2_name = data2.columns[-1]

    data1_values = data1[data1_name].values
    data2_values = data2[data2_name].values
    data1_units = title_to_quantity(data1_name)
    data2_units = title_to_quantity(data2_name)

    new_quantity = operation(data1_units, data2_units,
            *operation_args, **operation_kwargs)
    new_data_name = quantity_to_title(new_quantity, name=new_name)

    normalized_values = operation(data1_values, data2_values,
            *operation_args, **operation_kwargs)
    normalized_data = data1.iloc[:,:-1]
    normalized_data[new_data_name] = normalized_values
    return normalized_data

def interpolate(data1, data2, column_units=None):
    """
    Interpolates data2 into the locations of data1. Requires data1 and data2 have no less than two values. If data1 and data2 have more than two columns, this will use the first and last columns.

    :param data1: Data you want to manipulate in the future
    :param data2: Data you want to interpolate to be happy with data1.
    :returns interpolated_data2: data2 which has been interpolated to match the x-indices of data1.
    :param column_units: Which units to use for normalization. Defaults to None
    """
    y1_name, y2_name = data1.columns[-1], data2.columns[-1]

    if column_units is not None:
        x1_name = cname_from_unit(data1, column_units)
        x2_name = cname_from_unit(data2, column_units)
    else:
        x1_name = data1.columns[0]
        x2_name = data2.columns[0]

    x1_quantity = title_to_quantity(x1_name)
    x2_quantity = title_to_quantity(x2_name)
    y1_quantity = title_to_quantity(y1_name)
    y2_quantity = title_to_quantity(y2_name)

    x1 = data1[x1_name].values
    x2 = data2[x2_name].values * x2_quantity.to(x1_quantity).m
    y1 = data1[y1_name].values
    y2 = data2[y2_name].values * y2_quantity.to(y1_quantity).m

    interpolated_data = np.interp(x1, x2, y2)
    interpolated_frame = data1.iloc[:,:-1]
    interpolated_frame[y2_name] = interpolated_data
    return interpolated_frame
