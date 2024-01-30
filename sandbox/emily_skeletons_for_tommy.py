# %%
import io

import caveclient as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
neuron.reset_mask()

# %%
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

sns.set_context("talk")
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(data=changelog, x="x", y="y", linewidth=0, s=20, hue="is_merge")
sns.scatterplot(data=nodes, x="x", y="y", linewidth=0, s=1, color="black", zorder=-1)
ax.legend(title="Is merge?")

# %%
fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(data=nodes, x="x", y="y", hue="compartment", linewidth=0, s=5)

# %%
