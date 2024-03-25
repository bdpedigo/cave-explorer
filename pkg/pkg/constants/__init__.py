from pathlib import Path

RESULTS_PATH = Path(__file__).parent.parent.parent.parent
RESULTS_PATH = RESULTS_PATH / "results"

FIG_PATH = RESULTS_PATH / "figs"

OUT_PATH = RESULTS_PATH / "outs"

DATA_PATH = Path(__file__).parent.parent.parent.parent / "data"

DOC_FIG_PATH = Path(__file__).parent.parent.parent.parent / "docs" / "result_images"

VAR_PATH = Path(__file__).parent.parent.parent.parent / "docs" / "_variables.yml"

# TODO add more of these table names and write them to the quarto yaml
COLUMN_MTYPES_TABLE = "allen_column_mtypes_v2"
MTYPES_TABLE = "aibs_metamodel_mtypes_v661_v2"
NUCLEUS_TABLE = "nucleus_detection_v0"
