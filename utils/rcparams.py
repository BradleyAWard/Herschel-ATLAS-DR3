# ====================================================================================
# Dependencies
# =====================================================================================

import matplotlib
from cycler import cycler

# ====================================================================================
# Setup rcparams
# =====================================================================================

def rcparams():

    # Set color cycle:
    matplotlib.rcParams['axes.prop_cycle'] =  cycler('color', ['k', 'r', 'b', 'g', 'y', 'magenta', 'orange'])
    matplotlib.rcParams['figure.facecolor'] = 'white'

    # Set x axis
    matplotlib.rcParams['xtick.direction'] = "in"
    matplotlib.rcParams['xtick.major.size'] = 5
    matplotlib.rcParams['xtick.major.width'] = 0.5
    matplotlib.rcParams['xtick.minor.size'] = 2
    matplotlib.rcParams['xtick.minor.width'] = 0.5
    matplotlib.rcParams['xtick.minor.visible'] = True
    matplotlib.rcParams['xtick.top'] = True

    # Set y axis
    matplotlib.rcParams['ytick.direction'] = "in"
    matplotlib.rcParams['ytick.major.size'] = 5
    matplotlib.rcParams['ytick.major.width'] = 0.5
    matplotlib.rcParams['ytick.minor.size'] = 2
    matplotlib.rcParams['ytick.minor.width'] = 0.5
    matplotlib.rcParams['ytick.minor.visible'] = True
    matplotlib.rcParams['ytick.right'] = True

    # Set line widths
    matplotlib.rcParams['axes.linewidth'] = 1.25
    matplotlib.rcParams['grid.linewidth'] = 0.5
    matplotlib.rcParams['lines.linestyle'] = "-"
    matplotlib.rcParams['lines.linewidth'] = 1.25
    matplotlib.rcParams['lines.marker'] = None
    matplotlib.rcParams['lines.markerfacecolor'] = 'white'

    # Remove legend frame
    matplotlib.rcParams['legend.frameon'] = False
    matplotlib.rcParams['legend.fontsize'] = "small"

    # Always save as 'tight'
    matplotlib.rcParams['savefig.bbox'] = "tight"
    matplotlib.rcParams['savefig.pad_inches'] = 0.05

    # Use serif fonts
    # font.serif : Times
    matplotlib.rcParams['font.family'] = 'Avenir'
    matplotlib.rcParams['font.size'] = 16
    matplotlib.rcParams['mathtext.fontset'] = "dejavuserif"

    # Use LaTeX for math formatting
    matplotlib.rcParams['text.usetex'] = True
    #matplotlib.rcParams['text.latex.preamble'] = \usepackage{amsmath, amssymb}

    # Errorbar
    matplotlib.rcParams['errorbar.capsize'] = 2

    return None
