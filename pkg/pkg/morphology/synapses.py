import time

import pandas as pd
from tqdm import tqdm

from ..edits import get_level2_lineage_components


def get_pre_post_synapses(
    root_id, client, synapse_table=None, remove_self=True, verbose=False
):
    if synapse_table is None:
        synapse_table = client.info.get_datastack_info()["synapse_table"]

    # find all of the original objects that at some point were part of this neuron
    t = time.time()
    original_roots = client.chunkedgraph.get_original_roots(root_id)
    if verbose:
        print(f"Getting original roots took {time.time() - t:.2f} seconds")

    # now get all of the latest versions of those objects
    # this will likely be a larger set of objects than we started with since those
    # objects could have seen further editing, etc.
    t = time.time()
    latest_roots = client.chunkedgraph.get_latest_roots(original_roots)
    if verbose:
        print(f"Getting latest roots took {time.time() - t:.2f} seconds")

    tables = []
    for side in ["pre", "post"]:
        if verbose:
            print("Querying synapse table...")

        # get the pre/post-synapses that correspond to those objects
        t = time.time()
        syn_df = client.materialize.query_table(
            synapse_table,
            filter_in_dict={f"{side}_pt_root_id": latest_roots},
        )
        if verbose:
            print(f"Querying synapse table took {time.time() - t:.2f} seconds")

        if remove_self:
            syn_df = syn_df.query("pre_pt_root_id != post_pt_root_id")
        tables.append(syn_df)
    return (tables[0], tables[1])


# def _map_synapse_level2_ids(synapses, level2_lineage_component_map, side, client):
#     synapses[f"{side}_pt_current_level2_id"] = client.chunkedgraph.get_roots(
#         synapses[f"{side}_pt_supervoxel_id"], stop_layer=2
#     )

#     mapped_level2_id = {}
#     for idx, row in tqdm(list(synapses.iterrows())):
#         current_l2 = row[f"{side}_pt_current_level2_id"]

#         # this means that that level2 node was never part of a merge or split
#         # therefore we can safely keep the current mapping from supervoxel to level2 ID
#         if current_l2 not in level2_lineage_component_map:
#             mapped_level2_id[idx] = current_l2

#         # otherwise, we need to go back in the history and figure out what level2 IDs this
#         # synapse would have been apart of. We have a small set of nodes to check, because
#         # we kept track of what level2 IDs were part of what lineage components.
#         else:
#             component_label = level2_lineage_component_map[current_l2]
#             component_nodes = level2_lineage_component_map[
#                 level2_lineage_component_map == component_label
#             ].index

#             raise NotImplementedError(
#                 f"{side}: Need to finish this part if it ever even comes up."
#             )
#             mapped_level2_id[idx] = -1

#     synapses[f"{side}_pt_level2_id"] = pd.Series(mapped_level2_id)


def map_synapse_level2_ids(
    synapses, level2_lineage_component_map, l2dict_mesh, side, client, verbose=False
):
    synapses[f"{side}_pt_current_level2_id"] = client.chunkedgraph.get_roots(
        synapses[f"{side}_pt_supervoxel_id"], stop_layer=2
    )
    is_current = ~synapses[f"{side}_pt_current_level2_id"].isin(
        level2_lineage_component_map
    )

    if verbose:
        print(f"{len(is_current) - is_current.sum()} synapses are not current")

    synapses[f"{side}_pt_level2_id"] = pd.Series(index=synapses.index, dtype="Int64")

    synapses.loc[is_current, f"{side}_pt_level2_id"] = synapses.loc[
        is_current, f"{side}_pt_current_level2_id"
    ]

    for idx, row in synapses.query(f"{side}_pt_level2_id.isna()").iterrows():
        supervoxel = row[f"{side}_pt_supervoxel_id"]
        current_l2 = client.chunkedgraph.get_roots([supervoxel], stop_layer=2)[0]
        component = level2_lineage_component_map[current_l2]
        candidate_level2_ids = level2_lineage_component_map[
            level2_lineage_component_map == component
        ].index

        candidate_level2_ids = candidate_level2_ids[
            candidate_level2_ids.isin(l2dict_mesh.keys())
        ]

        winning_level2_id = None
        for candidate_level2_id in candidate_level2_ids:
            candidate_supervoxels = client.chunkedgraph.get_children(
                candidate_level2_id
            )
            if supervoxel in candidate_supervoxels:
                winning_level2_id = candidate_level2_id
                synapses.loc[idx, f"{side}_pt_level2_id"] = winning_level2_id
                break


def map_synapses(
    pre_synapses, post_synapses, networkdeltas_by_operation, l2dict_mesh, client
):
    level2_lineage_component_map = get_level2_lineage_components(
        networkdeltas_by_operation
    )

    outs = []
    for side, synapses in zip(["pre", "post"], [pre_synapses, post_synapses]):
        # put the level2 IDs into the synapse table, based on current state of neuron
        # as well as the lineage history
        map_synapse_level2_ids(
            synapses, level2_lineage_component_map, l2dict_mesh, side, client
        )

        # now we can map each of the synapses to the mesh index, via the level 2 id
        synapses = synapses.query(
            f"{side}_pt_level2_id.isin(@l2dict_mesh.keys())"
        ).copy()
        synapses[f"{side}_pt_mesh_ind"] = synapses[f"{side}_pt_level2_id"].map(
            l2dict_mesh
        )
        outs.append(synapses)

    return tuple(outs)


def apply_synapses(meshwork, pre_synapses, post_synapses, overwrite=True):
    # apply these synapse -> mesh index mappings to the meshwork
    for side, synapses in zip(["pre", "post"], [pre_synapses, post_synapses]):
        meshwork.anno.add_annotations(
            f"{side}_syn",
            synapses,
            index_column=f"{side}_pt_mesh_ind",
            point_column="ctr_pt_position",
            voxel_resolution=synapses.attrs.get("dataframe_resolution"),
            overwrite=overwrite,
        )
    return meshwork
