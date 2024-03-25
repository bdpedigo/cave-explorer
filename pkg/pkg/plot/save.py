import os
from typing import Optional

import matplotlib.pyplot as plt

from ..constants import DOC_FIG_PATH, FIG_PATH


def savefig(
    name: str,
    fig: plt.figure,
    folder: Optional[str] = None,
    format: str = "png",
    dpi: int = 300,
    bbox_inches="tight",
    doc_save: bool = False,
    fig_path=None,
    **kwargs,
) -> None:
    if fig_path is None:
        fig_path = FIG_PATH
    if folder is not None:
        path = fig_path / folder
    else:
        path = fig_path
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
            folder=folder,
            format=format,
            dpi=dpi,
            bbox_inches=bbox_inches,
            fig_path=DOC_FIG_PATH,
            doc_save=False,
            **kwargs,
        )
