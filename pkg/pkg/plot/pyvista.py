from typing import Any, Callable, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from tqdm.auto import tqdm

from ..constants import DOC_FIG_PATH, FIG_PATH

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
    location: Union[np.ndarray, list, pd.Series, "NetworkFrame"],
    setback: Union[float, int] = -2_000_000,
    elevation: Union[float, int] = 25,
    up: Literal["x", "y", "z", "-x", "-y", "-z"] = "-y",
):
    if isinstance(location, (np.ndarray, list, pd.Series)):
        nuc_loc = location
    else:
        nuc_loc = location.nodes.loc[location.nucleus_id, ["x", "y", "z"]].values
    plotter.camera_position = "zx"
    plotter.camera.focal_point = nuc_loc
    plotter.camera.position = nuc_loc + np.array([0, 0, setback])
    plotter.camera.up = UP_MAP[up]
    plotter.camera.elevation = elevation


def animate_neuron_edit_sequence(
    neuron_sequence: Any,
    name: str,
    folder: str,
    fps: int = 20,
    window_size: Optional[tuple] = (1024, 768),
    n_rotation_steps: int = 20,
    azimuth_step_size: Union[float, int] = 1,
    setback: Union[float, int] = -2_000_000,
    highlight_last: int = 2,
    highlight_point_size: float = 4,
    highlight_merge_color: str = "blue",
    highlight_split_color: str = "red",
    highlight_decay: float = 0.95,
    elevation: Union[float, int] = 25,
    up: Literal["x", "y", "z", "-x", "-y", "-z"] = "-y",
    merge_color: str = "lightblue",
    split_color: str = "lightcoral",
    edit_point_size: float = 3,
    line_width: float = 1,
    font_size: int = 40,
    fig: Optional[plt.Figure] = None,
    update: Optional[Callable] = None,
    doc_save: bool = False,
    caption: str = "",
    group: Optional[str] = None,
    verbose: bool = False,
):
    path = FIG_PATH / folder / f"{name}.gif"
    plotter = pv.Plotter(window_size=window_size, off_screen=True)
    plotter.open_gif(str(path), fps=fps)

    neurons = neuron_sequence.resolved_sequence

    # somehow this seems necessary as a hack for getting the camera in roughly the right
    # position, not sure what I'm missing in the custom camera position
    last_neuron = list(neurons.values())[-1]
    skeleton_poly = last_neuron.to_skeleton_polydata()
    skeleton_actor = plotter.add_mesh(
        skeleton_poly, color="black", line_width=line_width
    )

    set_up_camera(plotter, last_neuron, setback, elevation, up)

    plotter.camera.zoom(1)

    # remove dummy skeleton
    plotter.remove_actor(skeleton_actor)

    if fig is not None:
        chart = pv.ChartMPL(fig, size=(0.5, 0.3), loc=(0.0, 0.0))
        chart.background_color = (1.0, 1.0, 1.0, 0.0)
        chart.border_width = 0
        chart.active_border_color = (0.0, 0.0, 0.0, 0.0)
        chart.border_color = (0.0, 0.0, 0.0, 0.0)
        plotter.add_chart(chart)

    font_size = font_size
    gap = 40

    merge_label = pv.Text("Merges:", position=(gap, window_size[1] - gap))
    merge_label.prop.font_size = font_size
    merge_label.prop.justification_vertical = "top"
    merge_label.prop.color = highlight_merge_color
    plotter.add_actor(merge_label)

    split_label = pv.Text(
        "Splits:", position=(gap, window_size[1] - gap - font_size - 10)
    )
    split_label.prop.font_size = font_size
    split_label.prop.justification_vertical = "top"
    split_label.prop.color = highlight_split_color
    plotter.add_actor(split_label)

    last_neuron = next(iter(neurons.values()))
    merge_remove_queue = []
    split_remove_queue = []
    for i, (sample_id, neuron) in tqdm(
        enumerate(neurons.items()), disable=not verbose, desc="Writing frames..."
    ):
        if neuron.nodes.index.equals(last_neuron.nodes.index) and i != 0:
            continue
        if sample_id in neuron_sequence.sequence_info.index:
            edit_ids = neuron_sequence.sequence_info.loc[sample_id, "applied_edits"]
            edits = neuron.edits.query("index in @edit_ids")

            n_merges = edits["is_merge"].sum()
            n_splits = len(edits) - n_merges
        else:
            n_merges = 0
            n_splits = 0

        merge_label.input = f"Merges: {n_merges}"
        split_label.input = f"Splits: {n_splits}"

        if i == 0:
            title = pv.Text(
                "Original state",
                position=(window_size[0] / 2, window_size[1] - gap - font_size),
            )
            title.prop.font_size = font_size
            title.prop.justification_horizontal = "center"
            plotter.add_actor(title)

        if i == (len(neurons) - 1):
            title = pv.Text(
                "Final state",
                position=(window_size[0] / 2, window_size[1] - gap - font_size),
            )
            title.prop.font_size = font_size
            title.prop.justification_horizontal = "center"
            plotter.add_actor(title)

        if update is not None:
            # if sample_id is not None:
            update(sample_id)

        # NOTE: there might be a smarter way to do this with masking, but this seems fast
        skeleton_actor = plotter.add_mesh(
            neuron.to_skeleton_polydata(), color="black", line_width=line_width
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
                color=highlight_merge_color,
                point_size=highlight_point_size,
                line_width=3 * line_width,
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
                color=highlight_split_color,
                point_size=highlight_point_size,
                line_width=3 * line_width,
            )
            split_remove_queue.append((highlight_actor,))
        else:
            split_remove_queue.append(())

        if i == 0:
            _n_rotation_steps = n_rotation_steps * 6
        elif i == (len(neurons) - 1):
            _n_rotation_steps = n_rotation_steps * 10
        else:
            _n_rotation_steps = n_rotation_steps
        for _ in range(_n_rotation_steps):
            plotter.camera.azimuth += azimuth_step_size
            if len(merge_remove_queue) > 0:
                for element in merge_remove_queue:
                    if len(element) > 0:
                        actor = element[0]
                        actor.prop.line_width = actor.prop.line_width * highlight_decay
                        actor.prop.opacity = actor.prop.opacity * highlight_decay
            if len(split_remove_queue) > 0:
                for element in split_remove_queue:
                    if len(element) > 0:
                        actor = element[0]
                        actor.prop.line_width = actor.prop.line_width * highlight_decay
                        actor.prop.opacity = actor.prop.opacity * highlight_decay
            plotter.write_frame()

        if len(merge_remove_queue) > highlight_last:
            for actor in merge_remove_queue.pop(0):
                plotter.remove_actor(actor)

        if len(split_remove_queue) > highlight_last:
            for actor in split_remove_queue.pop(0):
                plotter.remove_actor(actor)

        last_neuron = neuron

        plotter.remove_actor(skeleton_actor)

        if i == 0:
            plotter.remove_actor(title)

        if len(merge_poly.points) > 0:
            plotter.remove_actor(merge_actor)
        if len(split_poly.points) > 0:
            plotter.remove_actor(split_actor)

    plotter.close()

    if doc_save:
        import shutil

        shutil.copyfile(path, DOC_FIG_PATH / folder / f"{name}.gif")

        markdown_out = f"![{caption}](result_images/{folder}/{name}.gif)"

        if group is not None:
            markdown_out += "{" + f'group="{group}"' + "}"
        print(markdown_out)
        print()
