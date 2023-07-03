import pylab
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from itertools import chain


def get_labels(data, fill="number"):
    """
    to get a dict of labels for groups in data
    input
      data: data to get label for
      fill = ["number"|"logic"|"both"], fill with number, logic label, or both
    return
      labels: a dict of labels for different sets
    example:
    In [12]: get_labels([range(10), range(5,15), range(3,8)], fill="both")
    Out[12]:
    {'001': '001: 0',
     '010': '010: 5',
     '011': '011: 0',
     '100': '100: 3',
     '101': '101: 2',
     '110': '110: 2',
     '111': '111: 3'}
    """

    N = len(data)

    sets_data = [set(data[i]) for i in range(N)]  # sets for separate groups
    s_all = set(chain(*data))                             # union of all sets

    # bin(3) --> '0b11', so bin(3).split('0b')[-1] will remove "0b"
    set_collections = {}
    for n in range(1, 2**N):
        key = bin(n).split('0b')[-1].zfill(N)
        value = s_all
        sets_for_intersection = [sets_data[i] for i in range(N) if  key[i] == '1']
        sets_for_difference = [sets_data[i] for i in range(N) if  key[i] == '0']
        for s in sets_for_intersection:
            value = value & s
        for s in sets_for_difference:
            value = value - s
        set_collections[key] = value

    if fill == "number":
        labels = {k: len(set_collections[k]) for k in set_collections}
    elif fill == "logic":
        labels = {k: k for k in set_collections}
    elif fill == "both":
        labels = {k: ("%s: %d" % (k, len(set_collections[k]))) for k in set_collections}
    else:  # invalid value
        raise Exception("invalid value for fill")

    return labels


def venn4(data=None, names=None, fill="number", show_names=True, show_plot=True, alignment = {},ax=None,**kwds):

    if (data is None) or len(data) != 4:
        raise Exception("length of data should be 4!")
    if (names is None) or (len(names) != 4):
        names = ("set 1", "set 2", "set 3", "set 4")

    labels = get_labels(data, fill=fill)

    # set figure size
    if 'figsize' in kwds and len(kwds['figsize']) == 2:
        # if 'figsize' is in kwds, and it is a list or tuple with length of 2
        figsize = kwds['figsize']
    else: # default figure size
        figsize = (10, 10)

    # set colors for different Circles or ellipses
    if 'colors' in kwds and isinstance(kwds['colors'], list) and len(kwds['colors']) >= 4:
        colors = kwds['colors']
    else:
        colors = ['r', 'g', 'b', 'c']
    
    if 'alpha' in kwds and isinstance(kwds['alpha'], float) and kwds['alpha'] > 0 and kwds['alpha'] < 1:
        alpha = kwds['alpha']
    else:
        alpha = 0.5


    # draw ellipse, the coordinates are hard coded in the rest of the function
    if isinstance(ax, plt.Axes):
        pass
    else:
        ax = plt.gca()

    patches = []
    width, height = 170, 110  # width and height of the ellipses
    patches.append(Ellipse((170, 170), width, height, -45, facecolor=colors[0], alpha=alpha))
    patches.append(Ellipse((200, 200), width, height, -45, facecolor=colors[1], alpha=alpha))
    patches.append(Ellipse((200, 200), width, height, -135, facecolor=colors[2], alpha=alpha))
    patches.append(Ellipse((230, 170), width, height, -135, facecolor=colors[3], alpha=alpha))

    patches.append(Ellipse((170, 170), width, height, -45, facecolor='none', alpha=alpha,edgecolor='k',ls=':',lw=2))
    patches.append(Ellipse((200, 200), width, height, -45, facecolor='none', alpha=alpha,edgecolor='k',ls=':',lw=2))
    patches.append(Ellipse((200, 200), width, height, -135, facecolor='none', alpha=alpha,edgecolor='k',ls=':',lw=2))
    patches.append(Ellipse((230, 170), width, height, -135, facecolor='none', alpha=alpha,edgecolor='k',ls=':',lw=2))
    for e in patches:
        ax.add_patch(e)
    ax.set_xlim(80, 320); ax.set_ylim(80, 320)
    ax.set_xticks([])
    ax.set_yticks([])
    #ax.set_aspect("equal")

    ### draw text
    # 1
    pylab.text(120, 200, labels['1000'], **alignment)
    pylab.text(280, 200, labels['0100'], **alignment)
    pylab.text(155, 250, labels['0010'], **alignment, fontweight='bold')
    pylab.text(245, 250, labels['0001'], **alignment)
    # 2
    pylab.text(200, 115, labels['1100'], **alignment)
    pylab.text(140, 225, labels['1010'], **alignment)
    pylab.text(145, 155, labels['1001'], **alignment)
    pylab.text(255, 155, labels['0110'], **alignment)
    pylab.text(260, 225, labels['0101'], **alignment)
    pylab.text(200, 240, labels['0011'], **alignment)
    # 3
    pylab.text(235, 205, labels['0111'], **alignment)
    pylab.text(165, 205, labels['1011'], **alignment)
    pylab.text(225, 135, labels['1101'], **alignment)
    pylab.text(175, 135, labels['1110'], **alignment)
    # 4
    pylab.text(200, 175, labels['1111'], **alignment)
    # names of different groups
    if show_names:
        pylab.text(100, 110, names[0], fontsize=16, **alignment)
        pylab.text(275, 110, names[1], fontsize=16, **alignment)
        pylab.text(75, 250, names[2], fontsize=16, **alignment)
        pylab.text(270, 250, names[3], fontsize=16, **alignment)

    #leg = ax.legend(names, loc='best', fancybox=True)
    #leg.get_frame().set_alpha(0.5)

    return ax
