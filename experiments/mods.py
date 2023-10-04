# %%
from itertools import pairwise

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anytree import Node, Walker
from anytree.iterators import PreOrderIter
from requests import HTTPError
from tqdm.autonotebook import tqdm
import pcg_skel
import skeleton_plot

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

# %%
root_id = 864691135510015497
root_id = client.chunkedgraph.get_latest_roots(root_id)[0]

# %%

lineage_graph = client.chunkedgraph.get_lineage_graph(root_id)
links = lineage_graph["links"]

len(lineage_graph["nodes"])

# %%
change_log = client.chunkedgraph.get_change_log(root_id)

# %%

details = client.chunkedgraph.get_operation_details(change_log["operations_ids"])

merges = {}
splits = {}
for operation, detail in details.items():
    operation = int(operation)
    source_coords = detail["source_coords"][0]
    sink_coords = detail["sink_coords"][0]
    x = (source_coords[0] + sink_coords[0]) / 2
    y = (source_coords[1] + sink_coords[1]) / 2
    z = (source_coords[2] + sink_coords[2]) / 2
    pt = [x, y, z]
    row = {"x": x, "y": y, "z": z, "pt": pt}
    if "added_edges" in detail:
        merges[operation] = row
    elif "removed_edges" in detail:
        splits[operation] = row

import pandas as pd

merges = pd.DataFrame.from_dict(merges, orient="index")
merges.index.name = "operation"
splits = pd.DataFrame.from_dict(splits, orient="index")
splits.index.name = "operation"

# %%

import pcg_skel

meshwork = pcg_skel.coord_space_meshwork(root_id, client=client, synapses="all")

# %%

meshwork.add_annotations("splits", splits)
meshwork.add_annotations("merges", merges)

# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
skeleton_plot.plot_tools.plot_mw_skel(
    meshwork,
    ax=ax,
    # pull_radius=True,
    invert_y=True,
    # plot_soma=True,
    # pull_compartment_colors=True,
    plot_presyn=True,
    plot_postsyn=False,
    pre_anno={"splits": "pt"},
    line_width=0.5,
    color='black',
    presyn_size=100
)
