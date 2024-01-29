# %%
import os

from pkg.io import lazycloud

os.environ["LAZYCLOUD_USE_CLOUD"] = "True"
os.environ["LAZYCLOUD_RECOMPUTE"] = "False"


@lazycloud("allen-minnie-phase3", "test", ".pkl")
def my_function(x):
    """Here's my function"""
    return x + 1

my_function(10011)
# %%
