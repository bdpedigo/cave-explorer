# %%

import time

t0 = time.time()


import caveclient as cc
import pandas as pd
import pcg_skel
import vtk
from meshparty import trimesh_vtk
from meshparty.trimesh_vtk import _setup_renderer

# %%

client = cc.CAVEclient("minnie65_phase3_v1")

cg = client.chunkedgraph

# %%
meta = client.materialize.query_table("allen_v1_column_types_slanted_ref")
meta = meta.sort_values("target_id")
nuc = client.materialize.query_table("nucleus_detection_v0").set_index("id")

# %%
# i = 2#
# i = 14

i = 6  # this one works

target_id = meta.iloc[i]["target_id"]
root_id = nuc.loc[target_id]["pt_root_id"]
root_id = client.chunkedgraph.get_latest_roots(root_id)[0]

# %%

skeleton = pcg_skel.coord_space_skeleton(root_id, client)

# %%
skeleton.vertices

# %%
skeleton.edges

# %%
skeleton.root

# %%
f = "test.swc"
skeleton.export_to_swc(f)
swc = pd.read_csv(f, sep=" ", header=None)
swc.columns = ["node_id", "structure", "x", "y", "z", "radius", "parent_id"]

# %%

import navis

tn = navis.TreeNeuron(swc)

fig = navis.plot3d(tn, soma=False, inline=False)
fig.update_layout(
    template="plotly_white",
    plot_bgcolor="rgba(1,1,0.5,0)",
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)
    ),
)
fig.show()


# %%
sk_actor = trimesh_vtk.skeleton_actor(skeleton, color=(0, 0, 0), line_width=5)

# %%


class vtkTimerCallback:
    def __init__(self, steps, actor, iren):
        self.timer_count = 0
        self.steps = steps
        self.actor = actor
        self.iren = iren
        self.timerId = None

    def execute(self, obj, event):
        step = 0
        while step < self.steps:
            self.actor.GetProperty().SetColor(
                (
                    step / 1000.0,
                    step / 1000.0,
                    step / 1000.0,
                )
            )
            iren.GetRenderWindow().Render()
            self.timer_count += 1
            step += 1
            if step == self.steps:
                iren.ResetTimer(self.timerId)
                step = 0
                self.timer_count = 0


ren, renWin, iren = _setup_renderer(
    video_width=1080, video_height=720, back_color=(1, 1, 1), camera=None
)

# for sk_actor in sk_actors:
ren.AddActor(sk_actor)


ren.ResetCamera()
renWin.Render()

trackCamera = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(trackCamera)
# enable user interface interactor
iren.Initialize()

cb = vtkTimerCallback(1000, sk_actor, iren)
iren.AddObserver("TimerEvent", cb.execute)
cb.timerId = iren.CreateRepeatingTimer(10_000)

iren.Render()
iren.Start()
