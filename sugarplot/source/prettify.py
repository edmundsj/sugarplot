import matplotlib.pyplot as plt
# TODO: figure out how to adjust axes so tht we can plot things even when we have y-labels like 0.0001 to avoid getting stuff cut off

# Change global settings of all plots generated in the future
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
cmap = plt.get_cmap("tab10")

def prettifyPlot(ax, fig=None):
    #for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             #ax.get_xticklabels() + ax.get_yticklabels()):
        #item.set_fontsize(18)

    # Edit the major and minor ticks of the x and y axes to make them thicker
    ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.xaxis.set_tick_params(which='minor', size=7, width=2, direction='in')
    ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
    ax.yaxis.set_tick_params(which='minor', size=7, width=2, direction='in')

    # Get rid of the right and top lines around the figure
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if fig != None:
        fig.subplots_adjust(bottom=0.15)
        fig.subplots_adjust(left=0.17)

#plt.tight_layout()
