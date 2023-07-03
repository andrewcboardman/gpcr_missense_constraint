from matplotlib import pyplot as plt
import seaborn as sns

def raincloud_plot(
    data, x, y, order = None, 
    palette = 'Dark2', violin_width = .8,
    ax = None):
    if ax is None:
        ax = plt.gca()

    if order is None:
        order = data[x].unique()

    ax = pt.half_violinplot(
        data = data, x = x, y = y, palette = palette,
        bw=.2,  linewidth=1, cut=0.,width=violin_width, 
        scale="width", inner=None, orient="v",order=order,
        ax = ax
        )
    
    ax = sns.stripplot(
        data = data, x = x, y = y, palette = palette,
        jitter=1,zorder=0, size = 2, 
        edgecolor="white",  orient="v",order=order,
        ax = ax
        )

    ax = sns.boxplot(
        data = data, x = x, y = y, palette = palette,
        color="black",orient="v",width=.15,zorder=10,
        showcaps=True,boxprops={'facecolor':'none', "zorder":10},
        showfliers=True,whiskerprops={'linewidth':2, "zorder":10},
        saturation=1,order=order,
        ax = ax
        )
    return ax