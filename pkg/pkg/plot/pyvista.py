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


def set_up_camera(
    plotter: pv.Plotter,
    neuron,
    setback: Union[float, int] = -2_000_000,
    elevation: Union[float, int] = 25,
    up: Literal["x", "y", "z", "-x", "-y", "-z"] = "-y",
):
    nuc_loc = neuron.nodes.loc[neuron.nucleus_id, ["x", "y", "z"]].values
    plotter.camera_position = "zx"
    plotter.camera.focal_point = nuc_loc
    plotter.camera.position = nuc_loc + np.array([0, 0, setback])
    plotter.camera.up = UP_MAP[up]
    plotter.camera.elevation = elevation


@beartype
def animate_neuron_edit_sequence(
    path: Union[str, Path],
    neurons: dict,
    fps: int = 20,
    window_size: Optional[tuple] = None,
    n_rotation_steps: int = 20,
    azimuth_step_size: Union[float, int] = 1,
    setback: Union[float, int] = -2_000_000,
    highlight_last: int = 2,
    elevation: Union[float, int] = 25,
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

    last_neuron = next(iter(neurons.values()))
    merge_remove_queue = []
    split_remove_queue = []
    for i, (sample_id, neuron) in enumerate(
        tqdm(neurons.items(), desc="Writing frames...")
    ):
        # NOTE: there might be a smarter way to do this with masking, but this seems fast
        skeleton_actor = plotter.add_mesh(
            neuron.to_skeleton_polydata(), color="black", line_width=1
        )

        merge_poly = neuron.to_merge_polydata()
        if len(merge_poly.points) > 0:
            merge_actor = plotter.add_mesh(
                merge_poly, color=merge_color, point_size=edit_point_size
            )
        split_poly = neuron.to_split_polydata()
        if len(split_poly.points) > 0:
            split_actor = plotter.add_mesh(
                split_poly, color=split_color, point_size=edit_point_size
            )

        merge_highlight = neuron.nodes.index.difference(last_neuron.nodes.index)
        if len(merge_highlight) > 0:
            highlight_poly = neuron.query_nodes(
                "index.isin(@merge_highlight)", local_dict=locals()
            ).to_skeleton_polydata()
            highlight_actor = plotter.add_mesh(
                highlight_poly,
                color=merge_color,
                point_size=edit_point_size,
                line_width=3,
            )
            merge_remove_queue.append((highlight_actor,))
        else:
            merge_remove_queue.append(())

        split_highlight = last_neuron.nodes.index.difference(neuron.nodes.index)
        if len(split_highlight) > 0:
            highlight_poly = last_neuron.query_nodes(
                "index.isin(@split_highlight)", local_dict=locals()
            ).to_skeleton_polydata()
            highlight_actor = plotter.add_mesh(
                highlight_poly,
                color=split_color,
                point_size=edit_point_size,
                line_width=3,
            )
            split_remove_queue.append((highlight_actor,))
        else:
            split_remove_queue.append(())

        for _ in range(n_rotation_steps):
            plotter.camera.azimuth += azimuth_step_size
            plotter.write_frame()

        if len(merge_remove_queue) > highlight_last:
            for actor in merge_remove_queue.pop(0):
                plotter.remove_actor(actor)
        if len(split_remove_queue) > highlight_last:
            for actor in split_remove_queue.pop(0):
                plotter.remove_actor(actor)

        last_neuron = neuron

        plotter.remove_actor(skeleton_actor)

        if len(merge_poly.points) > 0:
            plotter.remove_actor(merge_actor)
        if len(split_poly.points) > 0:
            plotter.remove_actor(split_actor)

    print("Closing gif...")
    plotter.close()
