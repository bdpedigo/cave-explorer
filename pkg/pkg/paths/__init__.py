from pathlib import Path

RESULTS_PATH = Path(__file__).parent.parent.parent.parent
RESULTS_PATH = RESULTS_PATH / "results"

FIG_PATH = RESULTS_PATH / "figs"

OUT_PATH = RESULTS_PATH / "outs"

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data"

DOC_FIG_PATH = Path(__file__).parent.parent.parent.parent / "docs" / "result_images"
