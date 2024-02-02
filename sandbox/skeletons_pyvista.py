# %%
import io
from itertools import chain

import caveclient as cc
import numpy as np
import pandas as pd
import pyvista as pv
import seaborn as sns
from cloudfiles import CloudFiles
from meshparty import meshwork
from requests import HTTPError

client = cc.CAVEclient("minnie65_public_v661")

root_id = 864691135013445270
nuc_id = 262779

filename = f"{root_id}_{nuc_id}.h5"
skel_path = "https://storage.googleapis.com/allen-minnie-phase3/minniephase3-emily-pcg-skeletons/minnie_all/v661/meshworks"


# %%
def load_mw(directory, filename):
    # REF: stolen from https://github.com/AllenInstitute/skeleton_plot/blob/main/skeleton_plot/skel_io.py
    # filename = f"{root_id}_{nuc_id}/{root_id}_{nuc_id}.h5"
    '''
    """loads a meshwork file from .h5 into meshparty.meshwork object

    Args:
        directory (str): directory location of meshwork .h5 file. in cloudpath format as seen in https://github.com/seung-lab/cloud-files
        filename (str): full .h5 filename

    Returns:
        meshwork (meshparty.meshwork): meshwork object containing .h5 data
    """'''

    if "://" not in directory:
        directory = "file://" + directory

    cf = CloudFiles(directory)
    binary = cf.get([filename])

    with io.BytesIO(cf.get(binary[0]["path"])) as f:
        f.seek(0)
        mw = meshwork.load_meshwork(f)

    return mw


neuron = load_mw(skel_path, filename)


points = neuron.skeleton.vertices.astype(float)
edges = neuron.skeleton.edges

lines = np.empty((len(edges), 3), dtype=int)
lines[:, 0] = 2
lines[:, 1:3] = edges


pv.set_jupyter_backend("client")


skeleton_poly = pv.PolyData(points, lines=lines)

points = neuron.anno.pre_syn["pre_pt_position"].apply(pd.Series).values.astype(float)
points *= np.array([4, 4, 40])
pre_syn_poly = pv.PolyData(points)

points = neuron.anno.post_syn["post_pt_position"].apply(pd.Series).values.astype(float)
points *= np.array([4, 4, 40])
post_syn_poly = pv.PolyData(points)


pl = pv.Plotter()
pl.add_mesh(skeleton_poly, color="black", line_width=2)
pl.add_mesh(pre_syn_poly, color="red", point_size=4)
pl.add_mesh(post_syn_poly, color="blue", point_size=4)
pl.show()

# %%

# logic for extracting compartment labels

neuron.reset_mask()

nodes = neuron.anno.lvl2_ids.df.copy()
nodes.set_index("mesh_ind_filt", inplace=True)

nodes["is_basal_dendrite"] = False
nodes.loc[neuron.anno.basal_mesh_labels["mesh_index_filt"], "is_basal_dendrite"] = True

nodes["is_apical_dendrite"] = False
nodes.loc[
    neuron.anno.apical_mesh_labels["mesh_index_filt"], "is_apical_dendrite"
] = True

nodes["is_axon"] = False
nodes.loc[neuron.anno.is_axon_class["mesh_index_filt"], "is_axon"] = True

nodes["is_soma"] = False
nodes.loc[neuron.root_region, "is_soma"] = True

nodes["n_labels"] = nodes[
    ["is_basal_dendrite", "is_apical_dendrite", "is_axon", "is_soma"]
].sum(axis=1)

nodes["is_dendrite"] = nodes["is_basal_dendrite"] | nodes["is_apical_dendrite"]


def label_compartment(row):
    if row["is_soma"]:
        return "soma"
    elif row["is_axon"]:
        return "axon"
    elif row["is_dendrite"]:
        return "dendrite"
    else:
        return "unknown"


nodes["compartment"] = nodes.apply(label_compartment, axis=1)

nodes.drop(
    ["is_basal_dendrite", "is_apical_dendrite", "is_axon", "is_soma"],
    axis=1,
    inplace=True,
)

nodes = nodes.reset_index(drop=False).set_index("lvl2_id")


locs = pd.DataFrame(
    client.l2cache.get_l2data(nodes.index.to_list(), attributes=["rep_coord_nm"])
).T
locs = client.l2cache.get_l2data(nodes.index.to_list(), attributes=["rep_coord_nm"])
locs = pd.DataFrame(locs).T
locs.index = locs.index.astype(int)
locs[["x", "y", "z"]] = locs["rep_coord_nm"].apply(pd.Series)

nodes = nodes.join(locs)

nodes

# %%

edges = pd.DataFrame(
    client.chunkedgraph.level2_chunk_graph(root_id), columns=["source", "target"]
)


# %%


def get_detailed_change_log(root_id, client, filtered=True):
    cg = client.chunkedgraph
    change_log = cg.get_tabular_change_log(root_id, filtered=filtered)[root_id]

    change_log.set_index("operation_id", inplace=True)
    change_log.sort_values("timestamp", inplace=True)
    change_log.drop(columns=["timestamp"], inplace=True)

    try:
        chunk_size = 500  # not sure exactly what the limit is
        details = {}
        for i in range(0, len(change_log), chunk_size):
            sub_details = cg.get_operation_details(
                change_log.index[i : i + chunk_size].to_list()
            )
            details.update(sub_details)
        assert len(details) == len(change_log)
        # details = cg.get_operation_details(change_log.index.to_list())
    except HTTPError:
        raise HTTPError(
            f"Oops, requested details for {chunk_size} operations at once and failed :("
        )
    details = pd.DataFrame(details).T
    details.index.name = "operation_id"
    details.index = details.index.astype(int)

    change_log = change_log.join(details)

    return change_log


changelog = get_detailed_change_log(root_id, client, filtered=True)

# %%


def rough_location(series):
    source_coords = np.array(series["source_coords"])
    sink_coords = np.array(series["sink_coords"])
    centroid = np.concatenate([source_coords, sink_coords]).mean(axis=0)
    centroid_nm = centroid * np.array([8, 8, 40])
    return centroid_nm


changelog["rough_location"] = changelog.apply(rough_location, axis=1)
changelog[["x", "y", "z"]] = changelog["rough_location"].apply(pd.Series)

# %%


def concat_nodes(series):
    if series["is_merge"]:
        edges = series["added_edges"]
    else:
        edges = series["removed_edges"]

    return list(chain.from_iterable(edges))


changelog["modified_nodes"] = changelog.apply(concat_nodes, axis=1)

# %%
changelog.sort_values("operation_id", inplace=True)
modified_nodes = (
    changelog["modified_nodes"].explode().to_frame().reset_index(drop=False)
)

modified_nodes = modified_nodes.join(
    changelog.drop(columns=["modified_nodes", "x", "y", "z"]), on="operation_id"
)


l2_ids = client.chunkedgraph.get_roots(
    modified_nodes["modified_nodes"].to_list(), stop_layer=2
)
modified_nodes["l2_ids"] = l2_ids
modified_nodes.drop_duplicates(subset=["modified_nodes"], inplace=True, keep="last")
modified_nodes = modified_nodes.join(nodes[["x", "y", "z", "compartment"]], on="l2_ids")

modified_nodes

# %%

points = nodes[["x", "y", "z"]].values.astype(float)

iloc_map = dict(zip(nodes.index.values, range(len(nodes))))
iloc_edges = edges.applymap(lambda x: iloc_map[x])

lines = np.empty((len(edges), 3), dtype=int)
lines[:, 0] = 2
lines[:, 1:3] = iloc_edges[["source", "target"]].values

# %%
labels = pd.Categorical(nodes["compartment"])
label_map = dict(zip(labels.categories, range(len(labels.categories))))
code_to_label_map = dict(zip(range(len(labels.categories)), labels.categories))


# %%
colors = sns.color_palette("tab20", 20).as_hex()

palette = {
    "axon": colors[3],  # light orange
    "dendrite": colors[1],  # light blue
    "soma": colors[5],  # light green
    "unknown": "black",
}

cmap = [palette[label] for code, label in code_to_label_map.items()]


# theming has to happen before setting backend
# pv.set_plot_theme(themes.DarkTheme())
pv.set_plot_theme("document")


pv.set_jupyter_backend("client")


pl = pv.Plotter()

l2graph_poly = pv.PolyData(points, lines=lines)
l2graph_poly["compartment"] = labels.codes

merges = modified_nodes.query("is_merge")
merge_poly = pv.PolyData(merges[["x", "y", "z"]].values.astype(float))

splits = modified_nodes.query("~is_merge")
split_poly = pv.PolyData(splits[["x", "y", "z"]].values.astype(float))

pl.add_mesh(l2graph_poly, cmap=cmap, line_width=2, scalars="compartment")

pl.add_mesh(merge_poly, color=colors[8], point_size=10)
pl.add_mesh(split_poly, color=colors[6], point_size=10)

pl.show()

