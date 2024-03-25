import seaborn as sns
import matplotlib as mpl


def set_context(context="paper", font_scale=1.5, rc=None):
    rc = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_context(context, font_scale, rc)
    for key, val in rc.items():
        mpl.rcParams[key] = val