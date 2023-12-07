import numpy as np
import pandas as pd
import seaborn as sns
from nglui.statebuilder import (
    AnnotationLayerConfig,
    ChainedStateBuilder,
    PointMapper,
    StateBuilder,
)
from nglui.statebuilder.helpers import from_client, package_state

from ..utils import get_nucleus_point_nm


def generate_neuron_base_builders(root_id, client):
    # REF: mostly stolen from nglui.statebuilder.helpers

    # find the soma position for setting the view
    nuc_pt_nm = get_nucleus_point_nm(root_id, client, method="table")

    # first df is just the root_id
    root_ids = [root_id]
    dataframes = [pd.DataFrame({"root_id": root_ids})]

    contrast = None

    # generate some generic segmentation/image layers
    img_layer, seg_layer = from_client(client, contrast=contrast)
    seg_layer.add_selection_map(selected_ids_column="root_id")

    # set position to the soma
    view_kws = {"position": np.array(nuc_pt_nm) / np.array([4, 4, 40])}
    base_sb = StateBuilder(
        layers=[img_layer, seg_layer], client=client, view_kws=view_kws
    )

    state_builders = [base_sb]

    return state_builders, dataframes


def generate_neurons_base_builders(root_ids, client):
    if isinstance(root_ids, (int, np.integer)):
        root_ids = [root_ids]
    # REF: mostly stolen from nglui.statebuilder.helpers

    # find the soma position for setting the view
    # nuc_pt_nm = get_nucleus_point_nm(root_ids[0], client, method="table")

    # first df is just the root_id
    dataframes = [pd.DataFrame({"root_id": root_ids})]

    contrast = None

    # generate some generic segmentation/image layers
    img_layer, seg_layer = from_client(client, contrast=contrast)
    seg_layer.add_selection_map(selected_ids_column="root_id")

    # set position to the soma
    # view_kws = {"position": np.array(nuc_pt_nm) / np.array([4, 4, 40])}
    view_kws = {}
    base_sb = StateBuilder(
        layers=[img_layer, seg_layer], client=client, view_kws=view_kws
    )

    state_builders = [base_sb]

    return state_builders, dataframes


def add_level2_edits(
    state_builders, dataframes, edit_df, client, by="metaoperation_id"
):
    # level2_graph_mapper = PointMapper(
    #     point_column="rep_coord_nm",
    #     description_column="l2_id",
    #     split_positions=False,
    #     gather_linked_segmentations=False,
    #     set_position=False,
    #     collapse_groups=True,
    # )
    # level2_graph_layer = AnnotationLayerConfig(
    #     "level2-graph",
    #     data_resolution=[1, 1, 1],
    #     color=(0.2, 0.2, 0.2),
    #     mapping_rules=level2_graph_mapper,
    # )
    # level2_graph_statebuilder = StateBuilder(
    #     layers=[level2_graph_layer],
    #     client=client,  # view_kws=view_kws
    # )
    # state_builders.append(level2_graph_statebuilder)
    # dataframes.append(final_nf.nodes.reset_index())

    if by is None:
        edit_df["_dummy"] = "all"
        by = "_dummy"

    colors = sns.color_palette("husl", len(edit_df[by].unique()))

    for i, (operation_id, operation_data) in enumerate(edit_df.groupby(by)):
        edit_point_mapper = PointMapper(
            point_column="rep_coord_nm",
            description_column="level2_node_id",
            split_positions=False,
            gather_linked_segmentations=False,
            set_position=False,
        )
        edit_layer = AnnotationLayerConfig(
            f"level2-operation-{operation_id}",
            data_resolution=[1, 1, 1],
            color=colors[i],
            mapping_rules=edit_point_mapper,
        )
        sb_edits = StateBuilder([edit_layer], client=client)  # view_kws=view_kws)
        state_builders.append(sb_edits)
        dataframes.append(operation_data)

    if by == "_dummy":
        edit_df.drop(columns="_dummy", inplace=True)

    return state_builders, dataframes


def finalize_link(state_builders, dataframes, client):
    return_as = "html"
    shorten = "always"
    ngl_url = None
    link_text = "Neuroglancer Link"

    sb = ChainedStateBuilder(state_builders)

    link = package_state(dataframes, sb, client, shorten, return_as, ngl_url, link_text)
    return link
