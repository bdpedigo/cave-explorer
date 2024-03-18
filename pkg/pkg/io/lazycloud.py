import json
import os
import pickle
from functools import wraps
from pathlib import Path
from typing import Callable, Literal, Optional, Union

import numpy as np
from cloudfiles import CloudFiles

from pkg.paths import OUT_PATH


def get_cloudfiles(
    use_cloud: bool, cloud_bucket: str, foldername: str, local_path: str = ""
) -> CloudFiles:
    if use_cloud:
        out_path = str(cloud_bucket) + "/" + foldername
        cf = CloudFiles("gs://" + out_path)
    else:
        out_path = str(local_path) + "/" + foldername
        cf = CloudFiles("file://" + str(out_path))
    return cf


def parametrized(dec):
    """This decorator allows you to easily create decorators that take arguments"""
    # REF: https://stackoverflow.com/questions/5929107/decorators-with-parameters

    @wraps(dec)
    def layer(*args, **kwargs):
        @wraps(dec)
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


class CustomJSONizer(json.JSONEncoder):
    def default(self, obj):
        return (
            super().encode(bool(obj))
            if isinstance(obj, np.bool_)
            else super().default(obj)
        )


@parametrized
def lazycloud(
    func: Callable,
    cloud_bucket: str,
    folder: str,
    file_suffix: str,
    arg_keys: Union[int, list[int]] = [],
    kwarg_keys: Union[str, list[str]] = [],
    local_path: Union[str, Path] = OUT_PATH,
    save_format: Literal["pickle", "json"] = "pickle",
    load_func: Optional[Callable] = None,
    save_func: Optional[Callable] = None,
    verify: bool = False,
) -> Callable:
    """
    This decorator is used to cache the results of a function in the cloud (or fallback
    to a local path).

    Parameters
    ----------
    func :
        The function to be decorated.
    cloud_bucket :
        The path to the cloud bucket to use for caching.
    folder :
        The folder within the cloud bucket to use for caching.
    file_suffix :
        The suffix to use when saving the file. This is appended to the argument used to
        index the cache, `arg_key`.
    arg_key :
        The index of the argument to use as the key for the cache. The default is 0. For
        instance, if `arg_key` is 0, then the first argument to the function will be
        used as the key for writing to and retrieving from the cache.
    local_path :
        The local path to use for caching.
    save_format :
        The format to use for saving the cache, can be "pickle" or "json". If None, will
        use the `load_func` and `save_func` instead.
    load_func :
        A function to use prior to pickling/json-ing, depending on `save_format`. This
        can be used to prepare an object for these operations.
    save_func :
        A function to use after unpickling/loading, depending on `save_format`. This can
        be used to prepare an object for use after these operations.
    verify :
        Whether to check if the loaded result from the cache matches the one computed
        when running the function. Requires the result to implement the `__eq__` method.
    """

    if save_format == "pickle":
        loader = pickle.loads
        saver = pickle.dumps
    elif save_format == "json":
        loader = json.loads

        def saver(obj):
            return json.dumps(obj, cls=CustomJSONizer)
    else:
        raise ValueError(f"Unknown save_format: {save_format}")

    @wraps(func)
    def wrapper(*args, **kwargs):
        # use_cloud = (
        #     os.environ.get("LAZYCLOUD_USE_CLOUD", "False").capitalize() == "True"
        # )
        cf = get_cloudfiles(True, cloud_bucket, folder, local_path=local_path)

        file_name = ""
        for arg_key in arg_keys:
            file_name += str(args[arg_key]) + "-"
        for kwarg_key in kwarg_keys:
            file_name += f"{str(kwarg_key)}={str(kwargs[kwarg_key])}-"
        file_name += file_suffix

        if "cache_verbose" in kwargs:
            cache_verbose = kwargs.get("cache_verbose")
        else:
            cache_verbose = False

        if "use_cache" in kwargs:
            use_cache = kwargs.get("use_cache")
            if not use_cache:
                force_recompute = True
            else:
                force_recompute = False
        elif "LAZYCLOUD_RECOMPUTE" in os.environ:
            force_recompute = (
                os.environ.get("LAZYCLOUD_RECOMPUTE").capitalize() == "True"
            )
        else:
            force_recompute = False

        if cache_verbose:
            print(
                f"LAZYCLOUD forcing recompute for function {func.__name__}:",
                force_recompute,
            )

        if "only_load" in kwargs:
            only_load = kwargs.get("only_load")
        else:
            only_load = False

        if (not cf.exists(file_name) or force_recompute) and not only_load:
            result = func(*args, **kwargs)
            if save_func:
                result = save_func(result)

            result = saver(result)
            if cache_verbose:
                print(f"LAZYCLOUD: Writing {file_name} to cloud...")
            cf.put(file_name, result)

        if only_load:
            if not cf.exists(file_name):
                return None

        if cache_verbose:
            print(f"LAZYCLOUD: Loading result {file_name} from cloud...")
        loaded_result = loader(cf.get(file_name))
        if load_func:
            loaded_result = load_func(loaded_result)

        if verify:
            is_same = result == loaded_result
            if not is_same:
                raise ValueError("Loaded result does not match original result.")

        return loaded_result

    return wrapper
