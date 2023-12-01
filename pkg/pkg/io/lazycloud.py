import os
from functools import wraps
from typing import Callable

from cloudfiles import CloudFiles

from pkg.paths import OUT_PATH

CLOUD_BUCKET = "allen-minnie-phase3"


def get_cloudfiles(use_cloud: bool, foldername: str) -> CloudFiles:
    if use_cloud:
        out_path = CLOUD_BUCKET + "/" + foldername
        cf = CloudFiles("gs://" + out_path)
    else:
        out_path = OUT_PATH / foldername
        cf = CloudFiles("file://" + str(out_path))
    return cf


def lazycloud(
    func: Callable, loader: Callable, saver: Callable, foldername: str, filename: str
) -> Callable:
    use_cloud = os.environ.get("LAZYCLOUD_USE_CLOUD") == "True"
    recompute = os.environ.get("LAZYCLOUD_RECOMPUTE") == "True"
    cf = get_cloudfiles(use_cloud, foldername)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if cf.exists(filename) and not recompute:
            return loader(filename)
        else:
            result = func(*args, **kwargs)
            saver(result, filename)
            return result

    return wrapper
