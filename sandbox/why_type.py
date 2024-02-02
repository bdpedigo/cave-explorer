# %%

from beartype import beartype


def my_function_untyped(a, b):
    """This function is not type annotated.

    Parameters
    ----------
    a : int
        The first number.

    b : int
        The second number.

    Returns
    -------
    int
        The sum of the two numbers.
    """
    if not isinstance(a, int):
        raise TypeError("a must be an int")
    if not isinstance(b, int):
        raise TypeError("b must be an int")
    return a + b


from typing import Union


@beartype
def my_function_typed(a: Union[int, float], b: int) -> int:
    """This function is type annotated.

    Parameters
    ----------
    a :
        The first number.
    b :
        The second number.

    Returns
    -------
    :
        The sum of the two numbers.
    """
    return a + b


# %%
my_function_typed(1, "2")

# %%
