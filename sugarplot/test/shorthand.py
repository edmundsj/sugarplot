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


def assert_axes_equal(actual_ax, desired_ax, atol=1e-15, rtol=1e-10):
    """
    Asserts that two axes are equal
    """
    actual_lines = actual_ax.get_lines()
    desired_lines = desired_ax.get_lines()
    assert_equal(len(actual_lines), len(desired_lines),
            err_msg='Number of lines in Axes not equal')
    for actual_line, desired_line in zip_longest(actual_lines, desired_lines):
        assert_line_equal(actual_line, desired_line, atol=atol, rtol=rtol)

    actual_xticks = actual_ax.get_xticks()
    actual_yticks = actual_ax.get_yticks()
    desired_xticks = desired_ax.get_xticks()
    desired_yticks = desired_ax.get_yticks()
    assert_array_equal(actual_xticks, desired_xticks, err_msg='xticks')
    assert_array_equal(actual_yticks, desired_yticks, err_msg='yticks')

    actual_xscale = actual_ax.get_xscale()
    actual_yscale = actual_ax.get_yscale()
    desired_xscale = desired_ax.get_xscale()
    desired_yscale = desired_ax.get_yscale()
    assert_equal(actual_xscale, desired_xscale, err_msg='xscale')
    assert_equal(actual_yscale, desired_yscale, err_msg='yscale')

    actual_xlabel = actual_ax.get_xlabel()
    actual_ylabel = actual_ax.get_ylabel()
    desired_xlabel = desired_ax.get_xlabel()
    desired_ylabel = desired_ax.get_ylabel()
    assert_equal(actual_xlabel, desired_xlabel, err_msg='xlabel')
    assert_equal(actual_ylabel, desired_ylabel, err_msg='ylabel')

def assert_figures_equal(actual_fig, desired_fig, atol=1e-15, rtol=1e-10):
    actual_axes = actual_fig.axes
    desired_axes = desired_fig.axes
    for actual_ax, desired_ax in zip_longest(actual_axes, desired_axes):
        assert_axes_equal(actual_ax, desired_ax, atol=atol, rtol=rtol)

