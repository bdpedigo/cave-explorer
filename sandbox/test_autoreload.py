# %%
from IPython import get_ipython

ip = get_ipython()
ip.extension_manager.loaded

from pkg.test import my_func

my_func()