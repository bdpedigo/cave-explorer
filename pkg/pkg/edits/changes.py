import json

import caveclient as cc
import networkx as nx
import numpy as np
import pandas as pd
from networkframe import NetworkFrame
from requests import HTTPError
from tqdm import tqdm

from ..morphology import (
    find_component_by_l2_id,
    get_alltime_synapses,
    map_synapse_level2_ids,
)
from ..utils import (
    get_all_nodes_edges,
    get_level2_nodes_edges,
    get_nucleus_point_nm,
    pt_to_xyz,
)


def get_changed_edges(before_edges, after_edges):
    before_edges.drop_duplicates()
    before_edges["is_before"] = True
    after_edges.drop_duplicates()
    after_edges["is_before"] = False
    delta_edges = pd.concat([before_edges, after_edges]).drop_duplicates(
        ["source", "target"], keep=False
    )
    removed_edges = delta_edges.query("is_before").drop(columns=["is_before"])
    added_edges = delta_edges.query("~is_before").drop(columns=["is_before"])
    return removed_edges, added_edges


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


class NetworkDelta:
    def __init__(
        self, removed_nodes, added_nodes, removed_edges, added_edges, metadata={}
    ):
        self.removed_nodes = removed_nodes
        self.added_nodes = added_nodes
        self.removed_edges = removed_edges
        self.added_edges = added_edges
        self.metadata = metadata

    def __repr__(self):
        rep = f"NetworkDelta(removed_nodes={self.removed_nodes.shape[0]}, "
        rep += f"added_nodes={self.added_nodes.shape[0]}, "
        rep += f"removed_edges={self.removed_edges.shape[0]}, "
        rep += f"added_edges={self.added_edges.shape[0]}, "
        rep += f"metadata={self.metadata}"
        rep += ")"
        return rep

    def to_dict(self):
        out = dict(
            removed_nodes=self.removed_nodes.index.to_list(),
            added_nodes=self.added_nodes.index.to_list(),
            removed_edges=self.removed_edges.values.tolist(),
            added_edges=self.added_edges.values.tolist(),
            metadata=self.metadata,
        )
        return out

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, input):
        removed_nodes = pd.DataFrame(index=input["removed_nodes"])
        added_nodes = pd.DataFrame(index=input["added_nodes"])
        removed_edges = pd.DataFrame(
            input["removed_edges"], columns=["source", "target"]
        )
        added_edges = pd.DataFrame(input["added_edges"], columns=["source", "target"])
        metadata = input["metadata"]
        return cls(
            removed_nodes, added_nodes, removed_edges, added_edges, metadata=metadata
        )

    @classmethod
    def from_json(cls, input):
        return cls.from_dict(json.loads(input))


def combine_deltas(deltas):
    total_added_nodes = pd.concat(
        [delta.added_nodes for delta in deltas], verify_integrity=True
    )
    total_removed_nodes = pd.concat(
        [delta.removed_nodes for delta in deltas], verify_integrity=True
    )

    total_added_edges = pd.concat(
        [
            delta.added_edges.set_index(["source", "target"], drop=True)
            for delta in deltas
        ],
        verify_integrity=True,
    ).reset_index(drop=False)
    total_removed_edges = pd.concat(
        [
            delta.removed_edges.set_index(["source", "target"], drop=True)
            for delta in deltas
        ],
        verify_integrity=True,
    ).reset_index(drop=False)

    return NetworkDelta(
        total_removed_nodes,
        total_added_nodes,
        total_removed_edges,
        total_added_edges,
    )


def get_network_edits(root_id, client, verbose=True):
    change_log = get_detailed_change_log(root_id, client, filtered=False)
    filtered_change_log = get_detailed_change_log(root_id, client, filtered=True)
    change_log["is_filtered"] = False
    change_log.loc[filtered_change_log.index, "is_filtered"] = True

    networkdeltas_by_operation = {}
    for operation_id in tqdm(
        change_log.index,
        desc="Finding network changes for each edit",
        disable=not verbose,
    ):
        row = change_log.loc[operation_id]

        before_root_ids = row["before_root_ids"]
        after_root_ids = row["roots"]

        # grabbing the union of before/after nodes/edges
        # NOTE: this is where all the compute time comes from
        all_before_nodes, all_before_edges = get_all_nodes_edges(
            before_root_ids, client, positions=False
        )
        all_after_nodes, all_after_edges = get_all_nodes_edges(
            after_root_ids, client, positions=False
        )

        # finding the nodes that were added or removed, simple set logic
        added_nodes_index = all_after_nodes.index.difference(all_before_nodes.index)
        added_nodes = all_after_nodes.loc[added_nodes_index]
        removed_nodes_index = all_before_nodes.index.difference(all_after_nodes.index)
        removed_nodes = all_before_nodes.loc[removed_nodes_index]

        # finding the edges that were added or removed, simple set logic again
        removed_edges, added_edges = get_changed_edges(
            all_before_edges, all_after_edges
        )

        # keep track of what changed
        # metadata = dict(
        #     operation_id=operation_id,
        #     is_merge=bool(is_merge),
        #     before_root_ids=before_root_ids,
        #     after_root_ids=after_root_ids,
        #     timestamp=row["timestamp"],
        # )
        metadata = {
            **row.to_dict(),
            "operation_id": operation_id,
            "root_id": root_id,
            "n_added_nodes": len(added_nodes),
            "n_removed_nodes": len(removed_nodes),
            "n_modified_nodes": len(added_nodes) + len(removed_nodes),
            "n_added_edges": len(added_edges),
            "n_removed_edges": len(removed_edges),
            "n_modified_edges": len(added_edges) + len(removed_edges),
        }

        networkdeltas_by_operation[operation_id] = NetworkDelta(
            removed_nodes, added_nodes, removed_edges, added_edges, metadata=metadata
        )

    return networkdeltas_by_operation


def get_network_metaedits(networkdeltas_by_operation, root_id, client):
    # find the nodes that are modified in any way by each operation
    mod_sets = {}
    for edit_id, delta in networkdeltas_by_operation.items():
        mod_set = []
        mod_set += delta.added_nodes.index.tolist()
        mod_set += delta.removed_nodes.index.tolist()
        mod_set += delta.added_edges["source"].tolist()
        mod_set += delta.added_edges["target"].tolist()
        mod_set += delta.removed_edges["source"].tolist()
        mod_set += delta.removed_edges["target"].tolist()
        mod_set = np.unique(mod_set)
        mod_sets[edit_id] = mod_set

    # make an incidence matrix of which nodes are modified by which operations
    index = np.unique(np.concatenate(list(mod_sets.values())))
    node_edit_indicators = pd.DataFrame(
        index=index, columns=networkdeltas_by_operation.keys(), data=False
    )
    for edit_id, mod_set in mod_sets.items():
        node_edit_indicators.loc[mod_set, edit_id] = True

    # this inner product matrix tells us which operations are connected with at least
    # one overlapping node in common
    X = node_edit_indicators.values.astype(int)
    product = X.T @ X
    product = pd.DataFrame(
        index=node_edit_indicators.columns,
        columns=node_edit_indicators.columns,
        data=product,
    )

    # meta-operations are connected components according to the above graph
    from scipy.sparse.csgraph import connected_components

    _, labels = connected_components(product.values, directed=False)

    meta_operation_map = {}
    for label in np.unique(labels):
        meta_operation_map[label] = node_edit_indicators.columns[
            labels == label
        ].tolist()

    # get the final network state for checking "relevance"
    nodes, edges = get_level2_nodes_edges(root_id, client, positions=False)
    final_nf = NetworkFrame(nodes, edges)

    # for each meta-operation, combine the deltas of the operations that make it up
    networkdeltas_by_meta_operation = {}
    for meta_operation_id, operation_ids in meta_operation_map.items():
        deltas = [
            networkdeltas_by_operation[operation_id] for operation_id in operation_ids
        ]
        meta_networkdelta = combine_deltas(deltas)

        is_relevant = meta_networkdelta.added_nodes.index.isin(
            final_nf.nodes.index
        ).any()

        all_metadata = {}
        for delta in deltas:
            all_metadata[delta.metadata["operation_id"]] = delta.metadata

        is_merge = [
            networkdeltas_by_operation[operation_id].metadata["is_merge"]
            for operation_id in operation_ids
        ]
        is_filtered = [
            networkdeltas_by_operation[operation_id].metadata["is_filtered"]
            for operation_id in operation_ids
        ]
        meta_networkdelta.metadata = dict(
            meta_operation_id=meta_operation_id,
            root_id=root_id,
            operation_ids=operation_ids,
            is_merge=is_merge,
            any_merge=np.any(is_merge),
            is_relevant=is_relevant,
            is_filtered=is_filtered,
            n_added_nodes=len(meta_networkdelta.added_nodes),
            n_removed_nodes=len(meta_networkdelta.removed_nodes),
            n_modified_nodes=len(meta_networkdelta.added_nodes)
            + len(meta_networkdelta.removed_nodes),
            n_added_edges=len(meta_networkdelta.added_edges),
            n_removed_edges=len(meta_networkdelta.removed_edges),
            n_modified_edges=len(meta_networkdelta.added_edges)
            + len(meta_networkdelta.removed_edges),
        )
        networkdeltas_by_meta_operation[meta_operation_id] = meta_networkdelta

    return networkdeltas_by_meta_operation, meta_operation_map


def find_supervoxel_component(supervoxel: int, nf: NetworkFrame, client):
    # supervoxel_l2_id = client.chunkedgraph.get_root_id(supervoxel, level2=True)
    # for component in nf.connected_components():
    #     if supervoxel_l2_id in component.nodes.index:
    #         return component
    # return None
    query_nf = nf.label_nodes_by_component()
    supervoxel_l2_id = client.chunkedgraph.get_root_id(supervoxel, level2=True)
    query_component = query_nf.nodes.loc[supervoxel_l2_id, "component"]
    query_nf = query_nf.query_nodes(f"component == {query_component}")
    return query_nf


def get_initial_node_ids(root_id, client):
    lineage_g = client.chunkedgraph.get_lineage_graph(root_id, as_nx_graph=True)
    node_in_degree = pd.Series(dict(lineage_g.in_degree()))
    original_node_ids = node_in_degree[node_in_degree == 0].index
    return original_node_ids


def get_initial_network(root_id, client, positions=False, verbose=True):
    original_node_ids = get_initial_node_ids(root_id, client)

    all_nodes = []
    all_edges = []
    had_error = False
    for leaf_id in tqdm(
        original_node_ids,
        desc="Finding L2 graphs for original segmentation objects",
        disable=not verbose,
    ):
        try:
            nodes, edges = get_level2_nodes_edges(leaf_id, client, positions=positions)
            all_nodes.append(nodes)
            all_edges.append(edges)
        except HTTPError:
            if isinstance(positions, bool) and positions:
                raise ValueError(
                    f"HTTPError: no level 2 graph found for node ID: {leaf_id}"
                )
            else:
                had_error = True
    if had_error:
        print("HTTPError on at least one leaf node, continuing...")
    all_nodes = pd.concat(all_nodes, axis=0)
    all_edges = pd.concat(all_edges, axis=0, ignore_index=True)

    nf = NetworkFrame(all_nodes, all_edges)
    return nf


def get_supervoxel_level2_map(root_id, networkdeltas_by_operation, client):
    ever_referenced_level2_ids = []

    nf = get_initial_network(root_id, client, positions=False)
    ever_referenced_level2_ids.extend(nf.nodes.index)

    for edit in networkdeltas_by_operation.values():
        ever_referenced_level2_ids.extend(edit.added_nodes)

    supervoxel_map = []
    for l2_id in tqdm(
        ever_referenced_level2_ids, desc="Getting supervoxel -> level 2 map"
    ):
        supervoxels = client.chunkedgraph.get_children(l2_id)
        this_map = pd.Series(index=supervoxels, data=l2_id)
        supervoxel_map.append(this_map)
    supervoxel_map = pd.concat(supervoxel_map)

    if supervoxel_map.index.duplicated().any():
        raise UserWarning("WARNING: supervoxel -> level 2 map has duplicates")

    return supervoxel_map


def apply_edit(
    network_frame: NetworkFrame,
    network_delta,
    key=None,
    label=None,
    label_dtype=int,
    copy=False,
):
    if copy:
        network_delta.added_nodes = network_delta.added_nodes.copy()
        network_delta.added_edges = network_delta.added_edges.copy()
    if key is not None:
        network_delta.added_nodes[key] = label
        network_delta.added_edges[key] = label
        network_delta.added_nodes = network_delta.added_nodes.astype(label_dtype)
        network_delta.added_edges = network_delta.added_edges.astype(label_dtype)
    network_frame.add_nodes(network_delta.added_nodes, inplace=True)
    network_frame.add_edges(network_delta.added_edges, inplace=True)
    network_frame.remove_nodes(network_delta.removed_nodes, inplace=True)
    network_frame.remove_edges(network_delta.removed_edges, inplace=True)


def pseudo_apply_edit(
    network_frame: NetworkFrame,
    network_delta,
    operation_label=None,
    metaoperation_label=None,
    label_dtype=int,
    copy=False,
):
    if copy:
        network_delta.added_nodes = network_delta.added_nodes.copy()
        network_delta.added_edges = network_delta.added_edges.copy()

    for col in [
        "operation_added",
        "operation_removed",
        "metaoperation_added",
        "metaoperation_removed",
    ]:
        network_delta.added_nodes[col] = -1
        network_delta.added_nodes[col] = network_delta.added_nodes[col].astype(
            label_dtype
        )
        network_delta.added_edges[col] = -1
        network_delta.added_edges[col] = network_delta.added_edges[col].astype(
            label_dtype
        )

    for df in [network_delta.added_nodes, network_delta.added_edges]:
        df["operation_added"] = operation_label
        df["operation_removed"] = -1
        df["metaoperation_added"] = metaoperation_label
        df["metaoperation_removed"] = -1

    network_frame.add_nodes(network_delta.added_nodes, inplace=True)
    network_frame.add_edges(network_delta.added_edges, inplace=True)

    network_frame.nodes.loc[
        network_delta.removed_nodes.index, "operation_removed"
    ] = operation_label
    network_frame.edges.loc[
        network_delta.removed_edges.index, "operation_removed"
    ] = operation_label
    network_frame.nodes.loc[
        network_delta.removed_nodes.index, "metaoperation_removed"
    ] = metaoperation_label
    network_frame.edges.loc[
        network_delta.removed_edges.index, "metaoperation_removed"
    ] = metaoperation_label


def apply_additions(
    network_frame, network_delta, key=None, label=None, label_dtype=int, copy=False
):
    if copy:
        network_delta.added_nodes = network_delta.added_nodes.copy()
        network_delta.added_edges = network_delta.added_edges.copy()
    if key is not None:
        network_delta.added_nodes[key] = label
        network_delta.added_edges[key] = label
        network_delta.added_nodes = network_delta.added_nodes.astype(label_dtype)
        network_delta.added_edges = network_delta.added_edges.astype(label_dtype)
    network_frame.add_nodes(network_delta.added_nodes, inplace=True)
    network_frame.add_edges(network_delta.added_edges, inplace=True)


def reverse_edit(network_frame: NetworkFrame, network_delta):
    network_frame.add_nodes(network_delta.removed_nodes, inplace=True)
    network_frame.add_edges(network_delta.removed_edges, inplace=True)
    nodes_to_remove = network_delta.added_nodes.index.intersection(
        network_frame.nodes.index
    )
    added_edges = network_delta.added_edges.set_index(["source", "target"])
    added_edges_index = added_edges.index
    current_edges_index = network_frame.edges.set_index(["source", "target"]).index
    edges_to_remove_index = added_edges_index.intersection(current_edges_index)
    edges_to_remove = added_edges.loc[edges_to_remove_index].reset_index()
    if len(nodes_to_remove) > 0 or len(edges_to_remove) > 0:
        network_frame.remove_nodes(nodes_to_remove, inplace=True)
        network_frame.remove_edges(edges_to_remove, inplace=True)
    else:
        print("Skipping edit:", network_delta.metadata)


def get_level2_lineage_components(networkdeltas_by_operation):
    graph = nx.DiGraph()

    for operation_id, delta in networkdeltas_by_operation.items():
        for node1 in delta.removed_nodes.index:
            for node2 in delta.added_nodes.index:
                graph.add_edge(node1, node2, operation_id=operation_id)

    level2_lineage_components = list(nx.weakly_connected_components(graph))

    level2_lineage_component_map = {}
    for i, component in enumerate(level2_lineage_components):
        for node in component:
            level2_lineage_component_map[node] = i

    level2_lineage_component_map = pd.Series(level2_lineage_component_map)

    return level2_lineage_component_map


def get_operation_metaoperation_map(networkdeltas_by_metaoperation):
    operation_to_metaoperation = {}
    for metaoperation_id, networkdelta in networkdeltas_by_metaoperation.items():
        metadata = networkdelta.metadata
        for operation_id in metadata["operation_ids"]:
            operation_to_metaoperation[operation_id] = metaoperation_id
    return operation_to_metaoperation


def collate_edit_info(
    networkdeltas_by_operation, operation_to_metaoperation, root_id, client
):
    nuc_pt_nm = get_nucleus_point_nm(root_id, client, method="table")

    raw_modified_nodes = []
    rows = []
    for operation_id, networkdelta in networkdeltas_by_operation.items():
        info = {
            **networkdelta.metadata,
            "nuc_pt_nm": nuc_pt_nm,
            "nuc_x": nuc_pt_nm[0],
            "nuc_y": nuc_pt_nm[1],
            "nuc_z": nuc_pt_nm[2],
            "metaoperation_id": operation_to_metaoperation[operation_id],
        }
        rows.append(info)

        modified_nodes = pd.concat(
            (networkdelta.added_nodes, networkdelta.removed_nodes)
        )
        modified_nodes.index.name = "level2_node_id"
        modified_nodes["root_id"] = root_id
        modified_nodes["operation_id"] = operation_id
        modified_nodes["is_merge"] = info["is_merge"]
        modified_nodes["is_relevant"] = info["is_relevant"]
        modified_nodes["is_filtered"] = info["is_filtered"]
        modified_nodes["metaoperation_id"] = info["metaoperation_id"]
        modified_nodes["is_added"] = modified_nodes.index.isin(
            networkdelta.added_nodes.index
        )
        raw_modified_nodes.append(modified_nodes)

    edit_stats = pd.DataFrame(rows)
    modified_level2_nodes = pd.concat(raw_modified_nodes)

    raw_node_coords = client.l2cache.get_l2data(
        np.unique(modified_level2_nodes.index.to_list()), attributes=["rep_coord_nm"]
    )

    node_coords = pd.DataFrame(raw_node_coords).T
    node_coords[node_coords["rep_coord_nm"].isna()]
    node_coords[["x", "y", "z"]] = pt_to_xyz(node_coords["rep_coord_nm"])
    node_coords.index = node_coords.index.astype(int)
    node_coords.index.name = "level2_node_id"

    modified_level2_nodes = modified_level2_nodes.join(
        node_coords, validate="many_to_one"
    )

    edit_centroids = modified_level2_nodes.groupby("operation_id")[
        ["x", "y", "z"]
    ].mean()

    edit_centroids.columns = ["centroid_x", "centroid_y", "centroid_z"]

    edit_stats = edit_stats.set_index("operation_id").join(edit_centroids)

    edit_stats["centroid_distance_to_nuc_nm"] = np.sqrt(
        (edit_stats["centroid_x"] - edit_stats["nuc_x"]) ** 2
        + (edit_stats["centroid_y"] - edit_stats["nuc_y"]) ** 2
        + (edit_stats["centroid_z"] - edit_stats["nuc_z"]) ** 2
    )

    edit_stats["centroid_distance_to_nuc_um"] = (
        edit_stats["centroid_distance_to_nuc_nm"] / 1000
    )

    edit_stats["datetime"] = pd.to_datetime(edit_stats["timestamp"], format="ISO8601")
    edit_stats["time"] = edit_stats["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

    metaoperation_stats = edit_stats.groupby("metaoperation_id").agg(
        {
            "centroid_x": "mean",
            "centroid_y": "mean",
            "centroid_z": "mean",
            "centroid_distance_to_nuc_um": "min",
            "datetime": "max",  # using the latest edit in a bunch as the time
        }
    )

    metaoperation_stats["time"] = metaoperation_stats["datetime"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    return edit_stats, metaoperation_stats, modified_level2_nodes


def apply_metaoperation_info(
    nf: NetworkFrame, networkdeltas_by_metaoperation: dict, edit_stats: pd.DataFrame
):
    nf.nodes["metaoperation_id"] = np.nan
    nf.nodes["metaoperation_id"] = nf.nodes["metaoperation_id"].astype("Int64")

    for metaoperation_id, networkdelta in networkdeltas_by_metaoperation.items():
        # TODO more robustly deal with add->removes in the metaedits
        # TODO maybe just take the intersection with the final network as a shortcut
        #      Since some of the metaedits might not have merges which are relevant so added
        #      nodes get removed later or something...
        net_added_node_ids = networkdelta.added_nodes.index.difference(
            networkdelta.removed_nodes.index
        )

        net_added_node_ids = networkdelta.added_nodes.index.intersection(nf.nodes.index)

        if len(net_added_node_ids) == 0:
            print(networkdelta.metadata["is_merges"])
            for operation_id in networkdelta.metadata["operation_ids"]:
                print(edit_stats.loc[operation_id, "is_filtered"])
            print()

        if not nf.nodes.loc[net_added_node_ids, "metaoperation_id"].isna().all():
            raise AssertionError("Some nodes already exist")
        else:
            nf.nodes.loc[net_added_node_ids, "metaoperation_id"] = metaoperation_id

    nf.nodes["has_operation"] = nf.nodes["metaoperation_id"].notna()


def apply_edit_history(nf, networkdeltas_by_operation, operation_to_metaoperation):
    for col in [
        "operation_added",
        "operation_removed",
        "metaoperation_added",
        "metaoperation_removed",
    ]:
        nf.nodes[col] = -1
        nf.nodes[col] = nf.nodes[col].astype(int)
        nf.edges[col] = -1
        nf.edges[col] = nf.edges[col].astype(int)

    nf.edges.set_index(["source", "target"], inplace=True, drop=False)

    for networkdelta in tqdm(networkdeltas_by_operation.values()):
        # this step is necessary to match the indexing set above
        networkdelta.added_edges = networkdelta.added_edges.set_index(
            ["source", "target"], drop=False
        )
        networkdelta.removed_edges = networkdelta.removed_edges.set_index(
            ["source", "target"], drop=False
        )

        # TODO what happens here for multiple operations in the same metaoperation?
        # since they by definition are touching some of the same nodes?
        operation_id = networkdelta.metadata["operation_id"]
        pseudo_apply_edit(
            nf,
            networkdelta,
            operation_label=operation_id,
            metaoperation_label=operation_to_metaoperation[operation_id],
        )

    # give the edges info about when those nodes were added
    nf.apply_node_features("operation_added", inplace=True)
    nf.apply_node_features("metaoperation_added", inplace=True)

    nf.edges["was_removed"] = nf.edges["operation_removed"] != -1
    nf.nodes["was_removed"] = nf.nodes["operation_removed"] != -1


def find_soma_nuc_merge_metaoperation(
    networkdeltas_by_metaoperation: dict,
    edit_stats: pd.DataFrame,
    nuc_dist_threshold=10,
    n_modified_nodes_threshold=10,
):
    soma_nuc_merge_metaoperation = None

    for metaoperation_id, networkdelta in networkdeltas_by_metaoperation.items():
        metadata = networkdelta.metadata
        operation_ids = metadata["operation_ids"]

        for operation_id in operation_ids:
            # check if any of the operations in this metaoperation are a soma/nucleus merge
            is_merge = edit_stats.loc[operation_id, "is_merge"]
            if is_merge:
                dist_um = edit_stats.loc[operation_id, "centroid_distance_to_nuc_um"]
                n_modified_nodes = edit_stats.loc[operation_id, "n_modified_nodes"]
                if (
                    dist_um < nuc_dist_threshold
                    and n_modified_nodes >= n_modified_nodes_threshold
                ):
                    soma_nuc_merge_metaoperation = metaoperation_id
                    print("Found soma/nucleus merge operation: ", operation_id)
                    print("Found soma/nucleus merge metaoperation: ", metaoperation_id)
                    break

    return soma_nuc_merge_metaoperation


def resolve_synapses_from_edit_selections(
    nf: NetworkFrame,
    edit_selections: dict,
    root_id: int,
    client: cc.CAVEclient,
    prefix="meta",
):
    """
    Assumes that several steps have been run prior to this

    - operation_added has been set on nodes and edges
    """

    resolved_pre_synapses = {}
    resolved_post_synapses = {}

    nuc = client.materialize.query_table(
        "nucleus_detection_v0",
        filter_equal_dict={"pt_root_id": root_id},
        select_columns=["pt_supervoxel_id", "pt_root_id", "pt_position"],
    ).set_index("pt_root_id")
    nuc_supervoxel = nuc.loc[root_id, "pt_supervoxel_id"]
    current_nuc_level2 = client.chunkedgraph.get_roots([nuc_supervoxel], stop_layer=2)[
        0
    ]

    first = True
    for selection_name, edit_selection in tqdm(edit_selections.items()):
        if not isinstance(edit_selection, list):
            edit_selection = list(edit_selection)

        # -1 for the initial state of the neuron
        if -1 not in edit_selection:
            edit_selection.append(-1)

        if first:
            print("Querying nodes...")
            print("Edit selection: ", edit_selection)
        sub_nf = nf.query_nodes(
            f"{prefix}operation_added.isin(@edit_selection)",
            local_dict=locals(),
        ).query_edges(
            f"{prefix}operation_added.isin(@edit_selection)",
            local_dict=locals(),
        )

        if first:
            print("Finding component...")
        # this takes up 90% of the time
        # i think it's from the operation of cycling through connected components
        instance_neuron_nf = find_component_by_l2_id(sub_nf, current_nuc_level2)

        if instance_neuron_nf is None:
            print("Missing nucleus component, assuming no synapses...")
            # this can happen if the lack of edits means the nucleus is no longer
            # connected
            found_pre_synapses = []
            found_post_synapses = []
        else:
            found_pre_synapses = []
            for synapses in instance_neuron_nf.nodes["pre_synapses"]:
                found_pre_synapses.extend(synapses)

            found_post_synapses = []
            for synapses in instance_neuron_nf.nodes["post_synapses"]:
                found_post_synapses.extend(synapses)

        resolved_pre_synapses[selection_name] = found_pre_synapses
        resolved_post_synapses[selection_name] = found_post_synapses

        first = False

    return resolved_pre_synapses, resolved_post_synapses


def count_synapses_by_sample(
    synapses: pd.DataFrame, resolved_pre_synapses: dict, by: str
) -> pd.DataFrame:
    """
    Count number of synapses belonging to some group (`by`) for each sample.

    Parameters
    ----------
    synapses : pd.DataFrame
        Synapses table.
    resolved_pre_synapses : dict
        Dictionary of resolved synapses by edit selection. Keys are edit selection
        identifiers, values are lists of synapse ids.
    by : str
        Column name to group by. Synapses will be grouped by this column, within each
        sample.
    """
    counts_by_sample = []
    for i, key in enumerate(resolved_pre_synapses.keys()):
        sample_resolved_synapses = resolved_pre_synapses[key]

        sample_counts = synapses.loc[sample_resolved_synapses].groupby(by).size()
        sample_counts.name = i
        counts_by_sample.append(sample_counts)

    count = pd.concat(counts_by_sample, axis=1).fillna(0).astype(int).T
    count.index.name = "sample"
    return count


def map_synapses_to_spatial_graph(
    pre_synapses,
    post_synapses,
    networkdeltas_by_operation,
    nodelist,
    client,
    l2dict_mesh=None,
    verbose=False,
):
    level2_lineage_component_map = get_level2_lineage_components(
        networkdeltas_by_operation
    )

    outs = []
    for side, synapses in zip(["pre", "post"], [pre_synapses, post_synapses]):
        if verbose:
            print(f"Mapping {side}-synapses to level2 IDs...")
        # put the level2 IDs into the synapse table, based on current state of neuron
        # as well as the lineage history
        map_synapse_level2_ids(
            synapses,
            level2_lineage_component_map,
            nodelist,
            side,
            client,
            verbose=verbose,
        )

        # now we can map each of the synapses to the mesh index, via the level 2 id
        synapses = synapses.query(f"{side}_pt_level2_id.isin(@nodelist)").copy()
        if l2dict_mesh is not None:
            synapses[f"{side}_pt_mesh_ind"] = synapses[f"{side}_pt_level2_id"].map(
                l2dict_mesh
            )
        outs.append(synapses)

    return tuple(outs)


def apply_synapses(
    nf: NetworkFrame,
    networkdeltas_by_operation: dict,
    root_id: int,
    client: cc.CAVEclient,
    verbose: bool = True,
):
    # map synapses onto the network
    # this involves looking at the entire set of synapses at any point in time so to speak

    pre_synapses, post_synapses = get_alltime_synapses(root_id, client, verbose=verbose)

    pre_synapses, post_synapses = map_synapses_to_spatial_graph(
        pre_synapses,
        post_synapses,
        networkdeltas_by_operation,
        nf.nodes.index,
        client,
        verbose=verbose,
    )

    # record this mapping onto the networkframe
    nf.nodes["synapses"] = [[] for _ in range(len(nf.nodes))]
    nf.nodes["pre_synapses"] = [[] for _ in range(len(nf.nodes))]
    nf.nodes["post_synapses"] = [[] for _ in range(len(nf.nodes))]

    for idx, synapse in pre_synapses.iterrows():
        nf.nodes.loc[synapse["pre_pt_current_level2_id"], "synapses"].append(idx)
        nf.nodes.loc[synapse["pre_pt_current_level2_id"], "pre_synapses"].append(idx)

    for idx, synapse in post_synapses.iterrows():
        nf.nodes.loc[synapse["post_pt_current_level2_id"], "synapses"].append(idx)
        nf.nodes.loc[synapse["post_pt_current_level2_id"], "post_synapses"].append(idx)

    nf.nodes["has_synapses"] = nf.nodes["synapses"].apply(len) > 0

    return pre_synapses, post_synapses
