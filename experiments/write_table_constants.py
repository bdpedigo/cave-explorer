# %%
import pkg.constants as constants
from pkg.io import write_variable

for name in dir(constants):
    if name.isupper() and name.endswith("_TABLE"):
        value = getattr(constants, name)
        write_variable(value, name)
