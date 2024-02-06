from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pyvista as pv
from beartype import beartype
from tqdm.auto import tqdm

UP_MAP = {
    "x": (1, 0, 0),
    "y": (0, 1, 0),
    "z": (0, 0, 1),
    "-x": (-1, 0, 0),
    "-y": (0, -1, 0),
    "-z": (0, 0, -1),
}


@beartype
def animate_neuron_edit_sequence(
    path: Union[str, Path],
    neurons: dict,
    fps: int = 20,
    window_size: Optional[tuple] = None,
    n_rotation_steps: int = 20,
    azimuth_step_size: float = 1,
    setback: float = -2_000_000,
    highlight_last: int = 2,
    elevation: float = 25,
    up: Literal["x", "y", "z", "-x", "-y", "-z"] = "-y",
    merge_color: str = "purple",
    split_color: str = "red",
    edit_point_size: float = 4,
):
    plotter = pv.Plotter(window_size=window_size)
    plotter.open_gif(path, fps=fps)

    # somehow this seems necessary as a hack for getting the camera in roughly the right
    # position, not sure what I'm missing in the custom camera position
    last_neuron = list(neurons.values())[-1]
    skeleton_poly = last_neuron.to_skeleton_polydata()
    skeleton_actor = plotter.add_mesh(skeleton_poly, color="black", line_width=0.1)

    # set up the camera
    nuc_loc = last_neuron.nodes.loc[last_neuron.nucleus_id, ["x", "y", "z"]].values
    plotter.camera_position = "zx"
    plotter.camera.focal_point = nuc_loc
    plotter.camera.position = nuc_loc + np.array([0, 0, setback])
    plotter.camera.up = UP_MAP[up]
    plotter.camera.elevation = elevation

    # remove dummy skeleton
    plotter.remove_actor(skeleton_actor)

    last_nodes = next(iter(neurons.values())).nodes
    actors_remove_queue = []
    for sample_id, neuron in tqdm(neurons.items(), desc="Writing frames..."):
        # NOTE: there might be a smarter way to do this with masking, but this seems fast
        skeleton_actor = plotter.add_mesh(
            neuron.to_skeleton_polydata(), color="black", line_width=1
        )

        merge_poly, split_poly = neuron.to_edit_polydata()
        if len(merge_poly.points) > 0:
            merge_actor = plotter.add_mesh(
                merge_poly, color=merge_color, point_size=edit_point_size
            )
        if len(split_poly.points) > 0:
            split_actor = plotter.add_mesh(
                split_poly, color=split_color, point_size=edit_point_size
            )

        highlight = neuron.nodes.index.difference(last_nodes.index)
        if len(highlight) > 0:
            highlight_poly = neuron.query_nodes(
                "index.isin(@highlight)", local_dict=locals()
            ).to_skeleton_polydata()
            highlight_actor = plotter.add_mesh(
                highlight_poly,
                color=merge_color,
                point_size=edit_point_size,
                line_width=3,
            )
            actors_remove_queue.append((highlight_actor,))

        for _ in range(n_rotation_steps):
            plotter.camera.azimuth += azimuth_step_size
            plotter.write_frame()

        if len(actors_remove_queue) > highlight_last:
            for actor in actors_remove_queue.pop(0):
                plotter.remove_actor(actor)
        last_nodes = neuron.nodes

        plotter.remove_actor(skeleton_actor)
        plotter.remove_actor(merge_actor)
        plotter.remove_actor(split_actor)

    print("Closing gif...")
    plotter.close()
