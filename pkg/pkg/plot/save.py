import os
from typing import Optional

import matplotlib.pyplot as plt

from ..paths import DOC_FIG_PATH, FIG_PATH


def savefig(
    name: str,
    fig: plt.figure,
    folder: Optional[str] = None,
    format: str = "png",
    dpi: int = 300,
    bbox_inches="tight",
    doc_save: bool = False,
    **kwargs,
) -> None:
    if folder is not None:
        path = FIG_PATH / folder
    else:
        path = FIG_PATH
    if not os.path.exists(path):
        os.makedirs(path)
    savename = name + "." + format
    fig.savefig(
        path / savename, format=format, dpi=dpi, bbox_inches=bbox_inches, **kwargs
    )
    if doc_save:
        savefig(
            name,
            fig,
            folder=DOC_FIG_PATH,
            format=format,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **kwargs,
        )
