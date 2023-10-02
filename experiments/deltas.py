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


#%%
sk = pcg_skel.coord_space_skeleton(root_id, client)



# %%
from meshparty import trimesh_vtk

actor = trimesh_vtk.skeleton_actor(sk)

# trimesh_vtk.render_actors(
#     [actor], filename="my_image.png", do_save=True, video_width=1600, video_height=1200
# )

