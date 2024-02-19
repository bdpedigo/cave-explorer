import json
import os
import pickle
from functools import wraps
from pathlib import Path
from typing import Callable, Literal, Union

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
        The format to use for saving the cache, can be "pickle" or "json".
    verify :
        Whether to check if the loaded result from the cache matches the one computed
        when running the function. Requires the result to implement the `__eq__` method.
    """
    use_cloud = os.environ.get("LAZYCLOUD_USE_CLOUD") == "True"

    if save_format == "pickle":
        loader = pickle.loads
        saver = pickle.dumps
    elif save_format == "json":
        loader = json.loads
        saver = json.dumps
    else:
        raise ValueError(f"Unknown save_format: {save_format}")

    cf = get_cloudfiles(use_cloud, cloud_bucket, folder, local_path=local_path)

    @wraps(func)
    def wrapper(*args, **kwargs):
        file_name = ""
        for arg_key in arg_keys:
            file_name += str(args[arg_key]) + "-"
        for kwarg_key in kwarg_keys:
            file_name += f"{str(kwarg_key)}={str(kwargs[kwarg_key])}-"
        file_name += file_suffix

        if "use_cache" in kwargs:
            use_cache = kwargs.pop("use_cache")
            if not use_cache:
                recompute = True
            else:
                recompute = False
        else:
            recompute = False

        if "only_load" in kwargs:
            only_load = kwargs.pop("only_load")
        else:
            only_load = False

        if "cache_verbose" in kwargs:
            cache_verbose = kwargs.pop("cache_verbose")
        else:
            cache_verbose = False

        if (not cf.exists(file_name) or recompute) and not only_load:
            result = func(*args, **kwargs)
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

        if verify:
            is_same = result == loaded_result
            if not is_same:
                raise ValueError("Loaded result does not match original result.")

        return loaded_result

    return wrapper
