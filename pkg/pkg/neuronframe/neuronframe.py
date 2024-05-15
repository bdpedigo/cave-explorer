import random
from typing import Literal, Optional, Self, Union

import caveclient as cc
import numpy as np
import pandas as pd
import pyvista as pv
from networkframe import NetworkFrame

from ..plot import set_up_camera


class NeuronFrame(NetworkFrame):
    def __init__(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        pre_synapses: Optional[pd.DataFrame] = None,
        post_synapses: Optional[pd.DataFrame] = None,
        edits: Optional[pd.DataFrame] = None,
        nucleus_id: Optional[int] = None,
        neuron_id: Optional[int] = None,
        pre_synapse_mapping_col: str = "pre_pt_level2_id",
        post_synapse_mapping_col: str = "post_pt_level2_id",
        validate: bool = False,
        **kwargs,
    ):
        super().__init__(nodes, edges, validate=validate, **kwargs)

        if pre_synapses is None:
            pre_synapses = pd.DataFrame()
        if post_synapses is None:
            post_synapses = pd.DataFrame()
        if edits is None:
            edits = pd.DataFrame()

        self.pre_synapses = pre_synapses
        self.post_synapses = post_synapses
        self.edits = edits
        self.nucleus_id = nucleus_id
        self.neuron_id = neuron_id
        self.pre_synapse_mapping_col = pre_synapse_mapping_col
        self.post_synapse_mapping_col = post_synapse_mapping_col

    def __repr__(self) -> str:
        out = (
            "NeuronFrame(\n"
            + f"    neuron_id={self.neuron_id},\n"
            + f"    nodes={self.nodes.shape},\n"
            + f"    edges={self.edges.shape},\n"
            + f"    pre_synapses={self.pre_synapses.shape},\n"
            + f"    post_synapses={self.post_synapses.shape},\n"
            + f"    edits={self.edits.shape},\n"
            + f"    nucleus_id={self.nucleus_id}\n"
            + ")"
        )
        return out

    @property
    def nucleus_id(self) -> int:
        return self._nucleus_id

    @nucleus_id.setter
    def nucleus_id(self, nucleus_id):
        if nucleus_id is not None and nucleus_id not in self.nodes.index:
            raise ValueError(f"nucleus_id {nucleus_id} not in nodes table index")
        self._nucleus_id = nucleus_id

    @property
    def has_pre_synapses(self) -> bool:
        return not self.pre_synapses.empty

    @property
    def has_post_synapses(self) -> bool:
        return not self.post_synapses.empty

    @property
    def has_edits(self) -> bool:
        return not self.edits.empty

    @property
    def pre_synapse_mapping_col(self) -> str:
        return self._pre_synapse_mapping_col

    @pre_synapse_mapping_col.setter
    def pre_synapse_mapping_col(self, pre_synapse_mapping_col: str):
        # check if the column exists
        if self.has_pre_synapses and (
            pre_synapse_mapping_col not in self.pre_synapses.columns
        ):
            msg = (
                f"pre_synapse_mapping_col '{pre_synapse_mapping_col}' not in "
                "pre_synapses table columns"
            )
            raise ValueError(msg)

        # check if all elements in the column are in the nodes index
        if (
            self.has_pre_synapses
            and not self.pre_synapses[pre_synapse_mapping_col]
            .isin(self.nodes.index)
            .all()
        ):
            msg = (
                f"pre_synapse_mapping_col '{pre_synapse_mapping_col}' contains "
                "values not in nodes index"
            )
            raise ValueError(msg)

        self._pre_synapse_mapping_col = pre_synapse_mapping_col

    @property
    def post_synapse_mapping_col(self) -> str:
        return self._post_synapse_mapping_col

    @post_synapse_mapping_col.setter
    def post_synapse_mapping_col(self, post_synapse_mapping_col: str):
        # check if the column exists
        if self.has_post_synapses and (
            post_synapse_mapping_col not in self.post_synapses.columns
        ):
            msg = (
                f"post_synapse_mapping_col '{post_synapse_mapping_col}' not in "
                "post_synapses table columns"
            )
            raise ValueError(msg)

        # check if all elements in the column are in the nodes index
        if (
            self.has_post_synapses
            and not self.post_synapses[post_synapse_mapping_col]
            .isin(self.nodes.index)
            .all()
        ):
            msg = (
                f"post_synapse_mapping_col '{post_synapse_mapping_col}' contains "
                "values not in nodes index"
            )
            raise ValueError(msg)

        self._post_synapse_mapping_col = post_synapse_mapping_col

    @property
    def metaedits(self, by="metaoperation_id", agg_rules=None) -> pd.DataFrame:
        if agg_rules is None:
            agg_rules = {
                "centroid_x": "mean",
                "centroid_y": "mean",
                "centroid_z": "mean",
                "centroid_distance_to_nuc_um": "min",
                "datetime": "max",  # using the latest edit in a bunch as the time
            }
        groupby = self.edits.groupby("metaoperation_id")
        groups = groupby.groups
        metaoperation_stats = groupby.agg(agg_rules)
        metaoperation_stats["time"] = metaoperation_stats["datetime"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        metaoperation_stats["operation_ids"] = metaoperation_stats.index.map(
            lambda x: groups[x].tolist()
        )
        metaoperation_stats["is_merges"] = metaoperation_stats.index.map(
            lambda x: self.edits.loc[groups[x], "is_merge"].tolist()
        )
        metaoperation_stats["has_merge"] = metaoperation_stats["is_merges"].apply(any)
        metaoperation_stats["has_split"] = metaoperation_stats["is_merges"].apply(
            lambda x: not all(x)
        )
        metaoperation_stats["n_operations"] = metaoperation_stats[
            "operation_ids"
        ].apply(len)
        metaoperation_stats["has_filtered"] = metaoperation_stats.index.map(
            lambda x: any(self.edits.loc[groups[x], "is_filtered"].tolist())
        )
        return metaoperation_stats

    def set_edits(self, edit_ids: Union[list[int], int], inplace=False, prefix=""):
        if isinstance(edit_ids, int):
            edit_ids = [edit_ids]

        # by convention -1 represents original things
        query = (
            f"({prefix}operation_added.isin(@edit_ids) | {prefix}operation_added == -1)"
            f" & ~{prefix}operation_removed.isin(@edit_ids)"
        )
        if inplace:
            self.query_nodes(query, local_dict=locals(), inplace=inplace)
            self.query_edges(query, local_dict=locals(), inplace=inplace)
        else:
            return self.query_nodes(
                query, local_dict=locals(), inplace=inplace
            ).query_edges(query, local_dict=locals(), inplace=inplace)

    def set_additions(self, edit_ids: Union[list[int], int], inplace=False, prefix=""):
        if isinstance(edit_ids, int):
            edit_ids = [edit_ids]

        # query does not do any removals
        query = (
            f"{prefix}operation_added.isin(@edit_ids) | {prefix}operation_added == -1"
        )
        if inplace:
            self.query_nodes(query, local_dict=locals(), inplace=inplace)
            self.query_edges(query, local_dict=locals(), inplace=inplace)
        else:
            return self.query_nodes(
                query, local_dict=locals(), inplace=inplace
            ).query_edges(query, local_dict=locals(), inplace=inplace)

    def remove_unused_synapses(
        self, which: Literal["both", "pre", "post"] = "both", inplace=False
    ) -> None:
        pre_synapses = self.pre_synapses
        post_synapses = self.post_synapses
        if which == "pre" or which == "both":
            pre_synapses = self.pre_synapses.query(
                f"{self.pre_synapse_mapping_col} in @self.nodes.index"
            )
        if which == "post" or which == "both":
            post_synapses = self.post_synapses.query(
                f"{self.post_synapse_mapping_col} in @self.nodes.index"
            )
        return self._return(
            pre_synapses=pre_synapses, post_synapses=post_synapses, inplace=inplace
        )

    def select_nucleus_component(self, inplace=False):
        if self.nucleus_id in self.nodes.index:
            return self.select_component_from_node(
                self.nucleus_id, directed=False, inplace=inplace
            )
        else:
            print("Warning: nucleus_id not in nodes index, returning unmodified")

            return self._return(inplace=inplace)

    def select_by_ball(self, radius: Union[float, int], inplace: bool = False) -> Self:
        """Select nodes within a ball of radius `radius` around the nucleus.

        Parameters
        ----------
        radius :
            Radius of the ball in nanometers.
        inplace :
            Whether to modify the current object or return a new one, by default False

        Returns
        -------
        Self
            NeuronFrame with selected nodes and edges, only returned if `inplace=False`.
        """
        from sklearn.metrics import pairwise_distances

        positions = self.nodes[["x", "y", "z"]].values
        nucleus_loc = self.nodes.loc[self.nucleus_id, ["x", "y", "z"]].values.reshape(
            1, 3
        )
        dists = np.squeeze(pairwise_distances(positions, nucleus_loc))

        return self.query_nodes(
            "index in @self.nodes.index[@dists < @radius]",
            inplace=inplace,
            local_dict=locals(),
        )

    def _generate_link_bases(self, client: cc.CAVEclient):
        from nglui import statebuilder

        sbs = []
        dfs = []
        viewer_resolution = client.info.viewer_resolution()
        img_layer = statebuilder.ImageLayerConfig(
            client.info.image_source(),
        )
        seg_layer = statebuilder.SegmentationLayerConfig(
            client.info.segmentation_source(), alpha_3d=0.3
        )
        seg_layer.add_selection_map(selected_ids_column="root_id")

        base_sb = statebuilder.StateBuilder(
            [img_layer, seg_layer],
            client=client,
            resolution=viewer_resolution,
        )
        base_df = pd.DataFrame({"root_id": [self.neuron_id]})
        sbs.append(base_sb)
        dfs.append(base_df)

        return sbs, dfs

    def generate_neuroglancer_link(
        self, client: cc.CAVEclient, color_edits=True, return_as="html"
    ):
        from nglui import statebuilder

        sbs, dfs = self._generate_link_bases(client)
        viewer_resolution = client.info.viewer_resolution()

        if ("source_rep_coord_nm" not in self.edges.columns) or (
            "target_rep_coord_nm" not in self.edges.columns
        ):
            edges = self.apply_node_features("rep_coord_nm", inplace=False).edges
        else:
            edges = self.edges

        # show unmodified level 2 graph in gray
        line_mapper = statebuilder.LineMapper(
            point_column_a="source_rep_coord_nm",
            point_column_b="target_rep_coord_nm",
            set_position=True,
        )
        line_layer = statebuilder.AnnotationLayerConfig(
            name="level 2 graph og",
            color="#d3d3d3",
            data_resolution=[1, 1, 1],
            mapping_rules=line_mapper,
        )
        line_sb = statebuilder.StateBuilder(
            [line_layer],
            client=client,
            resolution=viewer_resolution,
        )
        line_df = edges.query("operation_added == -1")
        sbs.append(line_sb)
        dfs.append(line_df)

        if color_edits:
            # show merges in blue
            merge_mapper = statebuilder.LineMapper(
                point_column_a="source_rep_coord_nm",
                point_column_b="target_rep_coord_nm",
                set_position=False,
            )
            merge_layer = statebuilder.AnnotationLayerConfig(
                name="merges",
                color="#0000ff",
                data_resolution=[1, 1, 1],
                mapping_rules=merge_mapper,
            )
            merge_sb = statebuilder.StateBuilder(
                [merge_layer],
                client=client,
                resolution=viewer_resolution,
            )
            merges = self.edits.query("is_merge").index
            merge_df = edges.query("operation_added.isin(@merges)", local_dict=locals())
            sbs.append(merge_sb)
            dfs.append(merge_df)

            # show splits in red
            split_mapper = statebuilder.LineMapper(
                point_column_a="source_rep_coord_nm",
                point_column_b="target_rep_coord_nm",
                set_position=False,
            )
            split_layer = statebuilder.AnnotationLayerConfig(
                name="splits",
                color="#ff0000",
                data_resolution=[1, 1, 1],
                mapping_rules=split_mapper,
            )
            split_sb = statebuilder.StateBuilder(
                [split_layer],
                client=client,
                resolution=viewer_resolution,
            )
            splits = self.edits.query("~is_merge").index
            split_df = edges.query("operation_added.isin(@splits)", local_dict=locals())
            sbs.append(split_sb)
            dfs.append(split_df)

            check = edges.query(
                "operation_added != -1 & ~operation_added.isin(@merges) & ~operation_added.isin(@splits)"
            )
            if len(check) > 0:
                raise ValueError(
                    "There are edges that are not from a merge, split, or original segmentation. "
                    "This seems to be an error."
                )

        sb = statebuilder.ChainedStateBuilder(sbs)
        return statebuilder.helpers.package_state(
            dfs, sb, client=client, return_as=return_as
        )

    def generate_neuroglancer_link_by_component(
        self, client: cc.CAVEclient, return_as="html", key="component_label"
    ):
        from nglui import statebuilder

        sbs, dfs = self._generate_link_bases(client)
        viewer_resolution = client.info.viewer_resolution()

        if ("source_rep_coord_nm" not in self.edges.columns) or (
            "target_rep_coord_nm" not in self.edges.columns
        ):
            edges = (
                self.apply_node_features("rep_coord_nm", inplace=False)
                .apply_node_features(key, inplace=False)
                .edges
            )
        else:
            edges = self.edges

        within_edges = edges.query(f"source_{key} == target_{key}")

        groups = within_edges.groupby(f"source_{key}")
        # colors = sns.color_palette("husl", n_colors=len(groups.groups)).as_hex()
        colors = [_random_hex() for _ in range(len(groups.groups))]
        print(len(groups.groups))
        for i, (group_label, within_group) in enumerate(groups):
            line_mapper = statebuilder.LineMapper(
                point_column_a="source_rep_coord_nm",
                point_column_b="target_rep_coord_nm",
                set_position=True,
            )
            line_layer = statebuilder.AnnotationLayerConfig(
                name=f"level 2 graph og {group_label}",
                color=colors[i],
                data_resolution=[1, 1, 1],
                mapping_rules=line_mapper,
            )
            line_sb = statebuilder.StateBuilder(
                [line_layer],
                client=client,
                resolution=viewer_resolution,
            )
            line_df = within_group.query("operation_added == -1")
            sbs.append(line_sb)
            dfs.append(line_df)

        between_edges = edges.query(f"source_{key} != target_{key}")

        merge_ids = self.edits.query("is_merge").index
        merges = between_edges.query(
            "operation_added.isin(@merge_ids)", local_dict=locals()
        )
        merge_mapper = statebuilder.LineMapper(
            point_column_a="source_rep_coord_nm",
            point_column_b="target_rep_coord_nm",
            set_position=False,
        )
        merge_layer = statebuilder.AnnotationLayerConfig(
            name="merges",
            color="#0000ff",
            data_resolution=[1, 1, 1],
            mapping_rules=merge_mapper,
        )
        merge_sb = statebuilder.StateBuilder(
            [merge_layer],
            client=client,
            resolution=viewer_resolution,
        )
        sbs.append(merge_sb)
        dfs.append(merges)

        split_ids = self.edits.query("~is_merge").index
        splits = between_edges.query(
            "operation_added.isin(@split_ids)", local_dict=locals()
        )
        split_mapper = statebuilder.LineMapper(
            point_column_a="source_rep_coord_nm",
            point_column_b="target_rep_coord_nm",
            set_position=False,
        )
        split_layer = statebuilder.AnnotationLayerConfig(
            name="splits",
            color="#ff0000",
            data_resolution=[1, 1, 1],
            mapping_rules=split_mapper,
        )
        split_sb = statebuilder.StateBuilder(
            [split_layer],
            client=client,
            resolution=viewer_resolution,
        )
        sbs.append(split_sb)
        dfs.append(splits)

        sb = statebuilder.ChainedStateBuilder(sbs)
        return statebuilder.helpers.package_state(
            dfs, sb, client=client, return_as=return_as
        )

    def remove_unused_edits(self):
        pass

    def to_skeleton_polydata(
        self, label: Optional[str] = None, draw_lines: bool = True
    ) -> pv.PolyData:
        nodes = self.nodes
        edges = self.edges

        points = nodes[["x", "y", "z"]].values.astype(float)

        if draw_lines:
            lines = _edges_to_lines(nodes, edges)
        else:
            lines = None

        skeleton = pv.PolyData(points, lines=lines)

        if label is not None:
            skeleton[label] = nodes[label].values

        return skeleton

    def to_merge_polydata(self, draw_edges=False, prefix="") -> pv.PolyData:
        if prefix == "meta":
            merge_ids = self.metaedits.query("has_merge").index
        else:
            merge_ids = self.edits.query("is_merge").index
        merge_nodes = self.nodes.query(
            f"({prefix}operation_added in @merge_ids) and ({prefix}operation_added != -1)"
        )
        points = merge_nodes[["x", "y", "z"]].values.astype(float)

        if draw_edges:
            merge_edges = self.edges.query(f"{prefix}operation_added.isin(@merge_ids)")
            merge_edges = merge_edges.query(
                "source in @merge_nodes.index and target in @merge_nodes.index"
            )
            lines = _edges_to_lines(merge_nodes, merge_edges)
        else:
            lines = None

        merge_poly = pv.PolyData(points, lines=lines)

        return merge_poly

    def to_split_polydata(self, draw_edges=False, prefix="") -> pv.PolyData:
        if prefix == "meta":
            split_ids = self.metaedits.query("~has_merge").index
        else:
            split_ids = self.edits.query("~is_merge").index
        split_nodes = self.nodes.query(
            f"({prefix}operation_removed in @split_ids) and ({prefix}operation_removed != -1)"
        )
        points = split_nodes[["x", "y", "z"]].values.astype(float)

        if draw_edges:
            split_edges = self.edges.query(
                f"{prefix}operation_removed.isin(@split_ids)"
            )
            # split_edges = split_edges.query(
            #     "source in @split_nodes.index and target in @split_nodes.index"
            # )
            lines = _edges_to_lines(self.nodes, split_edges)
        else:
            lines = None

        split_poly = pv.PolyData(points, lines=lines)

        return split_poly

    def plot_pyvista(
        self,
        plotter=None,
        color="black",
        line_width=0.1,
        show_edits=False,
        show=True,
        scalar=None,
    ):
        if plotter is None:
            plotter = pv.Plotter()
        set_up_camera(plotter, self)
        plotter.add_mesh(
            self.to_skeleton_polydata(label=scalar),
            color=color,
            line_width=line_width,
            scalars=scalar,
        )

        if self.has_edits and show_edits:
            merge_poly_points = self.to_merge_polydata(draw_edges=False)
            merge_poly_lines = self.to_merge_polydata(draw_edges=True)
            if len(merge_poly_points.points) > 0:
                plotter.add_mesh(merge_poly_points, color="blue", point_size=5)
                plotter.add_mesh(merge_poly_lines, color="blue", line_width=5)

            split_poly_points = self.to_split_polydata(draw_edges=False)
            split_poly_lines = self.to_split_polydata(draw_edges=True)
            if len(split_poly_points.points) > 0:
                plotter.add_mesh(
                    split_poly_points, color="red", point_size=5, line_width=5
                )
                plotter.add_mesh(
                    split_poly_lines, color="red", point_size=5, line_width=5
                )

        plotter.enable_fly_to_right_click(callback=None)
        if show:
            plotter.show()

    def apply_edge_lengths(self, inplace=False):
        sources = self.edges["source"]
        targets = self.edges["target"]
        source_pos = self.nodes.loc[sources, ["x", "y", "z"]].values
        target_pos = self.nodes.loc[targets, ["x", "y", "z"]].values
        edge_lengths = np.linalg.norm(source_pos - target_pos, axis=1)
        if inplace:
            self.edges["length"] = edge_lengths
        else:
            new_edges = self.edges.copy()
            new_edges["length"] = edge_lengths
            return self._return(edges=new_edges)

    # @property
    # def path_length(self):
    #     cols = ["source_x", "source_y", "source_z", "target_x", "target_y", "target_z"]
    #     if not np.isin(cols, self.edges.columns).all():
    #         edges = self.apply_node_features(["x", "y", "z"], inplace=False).edges
    #     else:
    #         edges = self.edges
    #     source_cols = cols[:3]
    #     target_cols = cols[3:]
    #     edge_lengths = np.linalg.norm(
    #         edges[source_cols].values - edges[target_cols].values, axis=1
    #     )
    #     return edge_lengths.sum()

    @property
    def path_length(self):
        if not np.isin(["length"], self.edges.columns).all():
            edges = self.apply_edge_lengths(inplace=False).edges
        else:
            edges = self.edges
        return edges["length"].sum()


def _random_rgb():
    return tuple(random.randint(0, 255) for _ in range(3))


def _random_hex():
    return "#%02X%02X%02X" % _random_rgb()


def _edges_to_lines(nodes, edges):
    iloc_map = dict(zip(nodes.index.values, range(len(nodes))))
    iloc_edges = edges[["source", "target"]].applymap(lambda x: iloc_map[x])

    lines = np.empty((len(edges), 3), dtype=int)
    lines[:, 0] = 2
    lines[:, 1:3] = iloc_edges[["source", "target"]].values

    return lines
