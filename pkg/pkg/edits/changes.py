import json

import networkx as nx
import numpy as np
import pandas as pd
from requests import HTTPError
from tqdm import tqdm

from neuropull.graph import NetworkFrame

from ..utils import get_all_nodes_edges, get_level2_nodes_edges
from .lineage import get_lineage_tree


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
            f"Oopsies, requested details for {chunk_size} operations at once and failed :("
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
        total_removed_nodes, total_added_nodes, total_removed_edges, total_added_edges
    )


def get_network_edits(root_id, client, filtered=True, verbose=True):
    change_log = get_detailed_change_log(root_id, client, filtered=filtered)

    edit_lineage_graph = nx.DiGraph()
    networkdeltas_by_operation = {}
    for operation_id in tqdm(
        change_log.index,
        desc="Finding network changes for each edit",
        disable=not verbose,
    ):
        row = change_log.loc[operation_id]
        is_merge = row["is_merge"]
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
        metadata = dict(
            operation_id=operation_id,
            is_merge=bool(is_merge),
            before_root_ids=before_root_ids,
            after_root_ids=after_root_ids,
            timestamp=row["timestamp"],
        )
        networkdeltas_by_operation[operation_id] = NetworkDelta(
            removed_nodes, added_nodes, removed_edges, added_edges, metadata=metadata
        )

        # summarize in edit lineage for L2 level
        for node1 in removed_nodes.index:
            for node2 in added_nodes.index:
                edit_lineage_graph.add_edge(
                    node1, node2, operation_id=operation_id, is_merge=is_merge
                )

    return networkdeltas_by_operation, edit_lineage_graph


def get_network_metaedits(networkdeltas_by_operation):
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

    # for each meta-operation, combine the deltas of the operations that make it up
    networkdeltas_by_meta_operation = {}
    for meta_operation_id, operation_ids in meta_operation_map.items():
        meta_networkdelta = combine_deltas(
            [networkdeltas_by_operation[operation_id] for operation_id in operation_ids]
        )
        meta_networkdelta.metadata = dict(
            meta_operation_id=meta_operation_id,
            operation_ids=operation_ids,
        )
        networkdeltas_by_meta_operation[meta_operation_id] = meta_networkdelta

    return networkdeltas_by_meta_operation, meta_operation_map


def _get_network_metaedits(networkdeltas_by_operation, edit_lineage_graph):
    meta_operation_map = {}
    for i, component in enumerate(nx.weakly_connected_components(edit_lineage_graph)):
        subgraph = edit_lineage_graph.subgraph(component)
        subgraph_operations = set()
        for _, _, data in subgraph.edges(data=True):
            subgraph_operations.add(data["operation_id"])
        meta_operation_map[i] = list(subgraph_operations)

    networkdeltas_by_meta_operation = {}
    for meta_operation_id, operation_ids in meta_operation_map.items():
        meta_networkdelta = combine_deltas(
            [networkdeltas_by_operation[operation_id] for operation_id in operation_ids]
        )
        meta_networkdelta.metadata = dict(
            meta_operation_id=meta_operation_id,
            operation_ids=operation_ids,
        )
        networkdeltas_by_meta_operation[meta_operation_id] = meta_networkdelta

    return networkdeltas_by_meta_operation, meta_operation_map


def find_supervoxel_component(supervoxel: int, nf: NetworkFrame, client):
    supervoxel_l2_id = client.chunkedgraph.get_root_id(supervoxel, level2=True)
    for component in nf.connected_components():
        if supervoxel_l2_id in component.nodes.index:
            return component
    return None


def get_initial_node_ids(root_id, client):
    lineage_g = client.chunkedgraph.get_lineage_graph(root_id, as_nx_graph=True)
    node_in_degree = pd.Series(dict(lineage_g.in_degree()))
    original_node_ids = node_in_degree[node_in_degree == 0].index
    return original_node_ids


# def get_initial_node_ids(root_id, client):
#     root = get_lineage_tree(root_id, client, flip=True, recurse=True, labels=False)


def get_initial_network(root_id, client, positions=False):
    original_node_ids = get_initial_node_ids(root_id, client)

    all_nodes = []
    all_edges = []
    for leaf_id in tqdm(
        original_node_ids, desc="Finding L2 graphs for original segmentation objects"
    ):
        try:
            nodes, edges = get_level2_nodes_edges(leaf_id, client, positions=positions)
        except HTTPError:
            print("HTTPError on node", leaf_id)
            continue
        all_nodes.append(nodes)
        all_edges.append(edges)
    all_nodes = pd.concat(all_nodes, axis=0)
    all_edges = pd.concat(all_edges, axis=0, ignore_index=True)

    nf = NetworkFrame(all_nodes, all_edges)
    return nf
