import yaml

from pkg.constants import VAR_PATH


def get_variables():
    with open(VAR_PATH, "r") as f:
        variables = yaml.safe_load(f)

    if variables is None:
        variables = {}

    return variables


def write_variable(value, name):
    # if it already exists, update it
    # if it doesn't exist, create it

    variables = get_variables()

    variables[name] = value

    with open(VAR_PATH, "w") as f:
        yaml.dump(variables, f)
