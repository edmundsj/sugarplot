from matplotlib.ticker import ScalarFormatter, NullFormatter, FuncFormatter
from sugarplot import cmap, plt

# Change global settings of all plots generated in the future
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.formatter.useoffset'] = False

def prettifyPlot(ax=None, fig=None):
    #for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             #ax.get_xticklabels() + ax.get_yticklabels()):
        #item.set_fontsize(18)
    if ax is None and fig is not None:
        for inner_ax in fig.axes:
            inner_ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
            inner_ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in')
            inner_ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
            inner_ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in')

            # Get rid of the right and top lines around the figure
            inner_ax.spines['right'].set_visible(False)
            inner_ax.spines['top'].set_visible(False)

            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            inner_ax.xaxis.set_major_formatter(formatter)
            #ax.xaxis.set_minor_formatter(formatter)
    else:

        # Edit the major and minor ticks of the x and y axes to make them thicker
        ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
        ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in')
        ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
        ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in')

        # Get rid of the right and top lines around the figure
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        formatter = ScalarFormatter()
        formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(formatter)
        #ax.xaxis.set_minor_formatter(formatter)

    if fig is not None:
        fig.set_tight_layout(True)

