import os
import pickle
from functools import wraps
from typing import Callable

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


# REF: https://stackoverflow.com/questions/5929107/decorators-with-parameters


def parametrized(dec):
    """This decorator allows you to easily create decorators that take arguments"""

    @wraps(dec)
    def layer(*args, **kwargs):
        @wraps(dec)
        def repl(f):
            return dec(f, *args, **kwargs)

        return repl

    return layer


# TODO these could be expanded or configurable from the lazycloud decorator
# right now they just pickle and unpickle


def loader(data):
    return pickle.loads(data)


def saver(data):
    return pickle.dumps(data)


@parametrized
def lazycloud(
    func: Callable,
    cloud_bucket: str,
    folder: str,
    file_suffix: str,
    arg_key: int = 0,
    local_path=OUT_PATH,
) -> Callable:
    use_cloud = os.environ.get("LAZYCLOUD_USE_CLOUD") == "True"
    recompute = os.environ.get("LAZYCLOUD_RECOMPUTE") == "True"

    cf = get_cloudfiles(use_cloud, cloud_bucket, folder, local_path=local_path)

    @wraps(func)
    def wrapper(*args, **kwargs):
        file_name = str(args[arg_key]) + "-" + file_suffix
        if cf.exists(file_name) and not recompute:
            return loader(cf.get(file_name))
        else:
            result = func(*args, **kwargs)
            result = saver(result)
            cf.put(file_name, result)
            result = loader(cf.get(file_name))
            return result

    return wrapper
