# %%
import caveclient as cc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib_venn import venn2
from tqdm.autonotebook import tqdm

# %%
client = cc.CAVEclient("minnie65_phase3_v1")

# %%

model_preds = client.materialize.query_table("aibs_metamodel_mtypes_v661_v2")

# %%
cg_client = client.chunkedgraph

root_id = model_preds.iloc[0]["pt_root_id"]

out = cg_client.level2_chunk_graph(root_id)

chunk_graph = pd.DataFrame(out, columns=["source", "target"])

# %%
from caveclient import CAVEclient
import pcg_skel

client = CAVEclient("minnie65_phase3_v1")
root_id = 864691135867734294


# %%
skeleton = pcg_skel.coord_space_skeleton(root_id, client, return_mesh=False)


# %%
from meshparty import trimesh_vtk

skeleton_actor = trimesh_vtk.skeleton_actor(skeleton)

trimesh_vtk.render_actors(
    [skeleton_actor],
    filename="my_image.png",
    do_save=True,
    video_width=1600,
    video_height=1200,
)

# trimesh_vtk.render_actors(
#     [actor], filename="my_image.png", do_save=True, video_width=1600, video_height=1200
# )

# %%

require_complete = False

cv = client.info.segmentation_cloudvolume(progress=False)

lvl2_eg = client.chunkedgraph.level2_chunk_graph(root_id)

from pcg_skel import chunk_tools

eg, l2dict_mesh, l2dict_r_mesh, x_ch = chunk_tools.build_spatial_graph(
    lvl2_eg,
    cv,
    client=client,
    method="service",
    require_complete=require_complete,
)

# %%
root_id = 864691135867734294
leaves = client.chunkedgraph.get_leaves(root_id, stop_layer=2)

# %%

changelog = client.chunkedgraph.get_change_log(root_id)

# %%
client.chunkedgraph.get_latest_roots(root_id)

# %%
lineage = client.chunkedgraph.get_lineage_graph(root_id)

# %%

links = lineage["links"]
from anytree import Node

nodes = {}
for link in links:
    source = link["source"]
    target = link["target"]
    if source not in nodes:
        nodes[source] = Node(source)
    if target not in nodes:
        nodes[target] = Node(target)

    if nodes[source].parent is not None:
        print("broke!")
    nodes[source].parent = nodes[target]

root = nodes[target].root

# %%
from anytree import RenderTree

print(RenderTree(nodes[root_id]))

# %%

from anytree.iterators import PreOrderIter

double_count = 0
for node in PreOrderIter(nodes[root_id]):
    if len(node.children) == 2:
        double_count += 1
print(double_count)
# matches the number of mergers!!!

zero_count = 0
for node in PreOrderIter(nodes[root_id]):
    if len(node.children) == 0:
        zero_count += 1
print(zero_count)
# must be the number of original connected components which are part of current root_id

one_count = 0
for node in PreOrderIter(nodes[root_id]):
    if len(node.children) == 1:
        one_count += 1
print(one_count)
# matches the number of splits!!!

# %%
