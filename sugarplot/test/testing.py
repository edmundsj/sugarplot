from matplotlib.figure import Figure
from itertools import zip_longest
from numpy.testing import assert_equal, assert_array_equal, assert_allclose

def assert_line_equal(actual_line, desired_line, atol=1e-15, rtol=1e-10):
    """
    Check that two matplotlib lines are equal (have the same raw data)
    """
    actual_xdata = actual_line.get_xdata()
    actual_ydata = actual_line.get_ydata()
    desired_xdata = desired_line.get_xdata()
    desired_ydata = desired_line.get_ydata()
    assert_allclose(actual_xdata, desired_xdata,
            err_msg='xdata', atol=atol, rtol=rtol)
    assert_allclose(actual_ydata, desired_ydata,
            err_msg='ydata', atol=atol,rtol=rtol)

    actual_linewidth = actual_line.get_linewidth()
    desired_linewidth = desired_line.get_linewidth()
    assert_equal(actual_linewidth, desired_linewidth, err_msg='linewidth')

    actual_linestyle = actual_line.get_linestyle()
    desired_linestyle = desired_line.get_linestyle()
    assert_equal(actual_linestyle, desired_linestyle,
            err_msg='linestyle')

    actual_alpha = actual_line.get_alpha()
    desired_alpha = desired_line.get_alpha()
    assert_equal(actual_alpha, desired_alpha, err_msg='alpha')

def assert_collection_equal(actual_collection, desired_collection,
        atol=1e-15, rtol=1e-10):
    actual_points = actual_collection.get_offsets()
    desired_points = desired_collection.get_offsets()
    assert_allclose(actual_points, desired_points,
            atol=atol, rtol=rtol, err_msg='Scatter plot points')

def assert_axes_equal(actual_ax, desired_ax, atol=1e-15, rtol=1e-10, skip=[]):
    """
    Asserts that two axes are equal
    """
    if isinstance(skip, str):
        skip = [skip]
    actual_lines = actual_ax.get_lines()
    desired_lines = desired_ax.get_lines()
    assert_equal(len(actual_lines), len(desired_lines),
            err_msg='Number of lines in Axes not equal')
    for actual_line, desired_line in zip_longest(actual_lines, desired_lines):
        assert_line_equal(actual_line, desired_line, atol=atol, rtol=rtol)

    actual_collections = actual_ax.collections
    desired_collections = desired_ax.collections
    for actual_collection, desired_collection in \
        zip_longest(actual_collections, desired_collections):
        assert_collection_equal(actual_collection, desired_collection,
                atol=atol, rtol=rtol)

    if 'xticks' not in skip and 'ticks' not in skip:
        actual_xticks = actual_ax.get_xticks()
        desired_xticks = desired_ax.get_xticks()
        assert_array_equal(
                actual_xticks, desired_xticks, err_msg='xticks')
    if 'yticks' not in skip and 'ticks' not in skip:
        actual_yticks = actual_ax.get_yticks()
        desired_yticks = desired_ax.get_yticks()
        assert_array_equal(
                actual_yticks, desired_yticks, err_msg='yticks')

    if 'xscale' not in skip and 'scale' not in skip:
        actual_xscale = actual_ax.get_xscale()
        desired_xscale = desired_ax.get_xscale()
        assert_equal(actual_xscale, desired_xscale, err_msg='xscale')
    if 'yscale' not in skip and 'scale' not in skip:
        actual_yscale = actual_ax.get_yscale()
        desired_yscale = desired_ax.get_yscale()
        assert_equal(actual_yscale, desired_yscale, err_msg='yscale')

    if 'xlabel' not in skip and 'label' not in skip:
        actual_xlabel = actual_ax.get_xlabel()
        desired_xlabel = desired_ax.get_xlabel()
        assert_equal(actual_xlabel, desired_xlabel, err_msg='xlabel')
    if 'ylabel' not in skip and 'label' not in skip:
        actual_ylabel = actual_ax.get_ylabel()
        desired_ylabel = desired_ax.get_ylabel()
        assert_equal(actual_ylabel, desired_ylabel, err_msg='ylabel')

def assert_figures_equal(actual_fig, desired_fig, atol=1e-15, rtol=1e-10, skip=[]):
    if isinstance(skip, str):
        skip = [skip]
    actual_axes = actual_fig.axes
    desired_axes = desired_fig.axes
    assert_equal(len(actual_axes), len(desired_axes), err_msg='Figures have differing number of axes')
    for actual_ax, desired_ax in zip_longest(actual_axes, desired_axes):
        assert_axes_equal(
                actual_ax, desired_ax, atol=atol, rtol=rtol, skip=skip)
