import numpy as np
import pandas as pd
from caveclient import CAVEclient
from networkframe import NetworkFrame
from requests.exceptions import HTTPError

FEATURES = [
    "area_nm2",
    "max_dt_nm",
    "mean_dt_nm",
    "pca",
    "pca_val",
    "rep_coord_nm",
    "size_nm3",
]


def _unwrap_pca(pca):
    if np.isnan(pca).all():
        return np.full(9, np.nan)
    return np.abs(np.array(pca).ravel())


def _unwrap_pca_val(pca):
    if np.isnan(pca).all():
        return np.full(3, np.nan)

    return np.array(pca).ravel()


# def rewrap_pca(pca):
#     # take the vector and transform back into 3x3 matrix
#     if np.isnan(pca).all():
#         return np.full((3, 3), np.nan)
#     return np.abs(np.array(pca).reshape(3, 3))


def process_node_data(node_data):
    scalar_features = node_data[
        ["area_nm2", "max_dt_nm", "mean_dt_nm", "size_nm3"]
    ].astype(float)

    pca_unwrapped = np.stack(node_data["pca"].apply(_unwrap_pca).values)
    pca_unwrapped = pd.DataFrame(
        pca_unwrapped,
        columns=[f"pca_unwrapped_{i}" for i in range(9)],
        index=node_data.index,
    )

    pca_val_unwrapped = np.stack(node_data["pca_val"].apply(_unwrap_pca_val).values)
    pca_val_unwrapped = pd.DataFrame(
        pca_val_unwrapped,
        columns=[f"pca_val_unwrapped_{i}" for i in range(3)],
        index=node_data.index,
    )
    pca_val_unwrapped["pca_ratio_01"] = (
        pca_val_unwrapped["pca_val_unwrapped_0"]
        / pca_val_unwrapped["pca_val_unwrapped_1"]
    )

    rep_coord_unwrapped = np.stack(node_data["rep_coord_nm"].values)
    rep_coord_unwrapped = pd.DataFrame(
        rep_coord_unwrapped,
        columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"],
        index=node_data.index,
    )

    clean_node_data = pd.concat(
        [scalar_features, pca_unwrapped, pca_val_unwrapped, rep_coord_unwrapped], axis=1
    )

    return clean_node_data


class L2FeatureExtractor:
    def __init__(
        self,
        client: CAVEclient,
        n_jobs=1,
        continue_on_error=True,
        verbose=False,
        drop_self_in_neighborhood=True,
    ):
        self.client = client
        self.n_jobs = n_jobs
        self.continue_on_error = continue_on_error
        self.verbose = verbose
        self.drop_self_in_neighborhood = drop_self_in_neighborhood

    def print(self, msg, level=0):
        if self.verbose >= level:
            print(msg)

    def get_features(self, object_ids, neighborhood_hops=5):
        if isinstance(object_ids, (int, np.integer)):
            object_ids = [object_ids]

        missing_object_ids = []
        data_by_object = []
        for object_id in object_ids:
            self.print(f"Extracting features for object {object_id}", level=2)
            object_node_data = self._extract_node_features(object_id)
            if object_node_data is None:
                missing_object_ids.append(object_id)
                continue

            self.print(f"Extracting level 2 graph for object {object_id}", level=2)
            object_edges = self._extract_edges(object_id)
            object_nf = NetworkFrame(object_node_data, object_edges)

            self.print(
                f"Extracting neighborhood features for object {object_id}", level=2
            )
            object_neighborhood_features = self._compute_neighborhood_features(
                object_nf, k=neighborhood_hops
            )

            object_node_data = object_node_data.join(object_neighborhood_features)

            object_node_data["object_id"] = object_id
            data_by_object.append(object_node_data)

        node_data = pd.concat(data_by_object)
        node_data.reset_index(inplace=True)
        node_data.set_index(["object_id", "l2_id"], inplace=True)
        return node_data

    def _extract_node_features(self, object_id):
        l2_ids = self.client.chunkedgraph.get_leaves(object_id, stop_layer=2)
        try:
            node_data = pd.DataFrame(
                self.client.l2cache.get_l2data(l2_ids, attributes=FEATURES)
            ).T
            node_data.index = node_data.index.astype(int)
            node_data.index.name = "l2_id"
            node_data = process_node_data(node_data)
            return node_data

        except HTTPError as e:
            if self.continue_on_error:
                if self.verbose:
                    print(f"Error fetching data for object {object_id}: {e}")
                return None
            else:
                raise e

    def _extract_edges(self, object_id):
        edges = self.client.chunkedgraph.level2_chunk_graph(object_id)
        edges = pd.DataFrame(edges, columns=["source", "target"])
        return edges

    def _compute_neighborhood_features(self, nf: NetworkFrame, k=None, distance=None):
        if distance is not None:
            raise NotImplementedError(
                "Distance-based neighborhood features not yet implemented"
            )
        else:
            assert k is not None
            # hard-coded as undirected for now
            neighborhoods = nf.k_hop_decomposition(k=k, directed=False)

        rows = []
        for node, neighborhood in neighborhoods.items():
            neighborhood_nodes = neighborhood.nodes
            if self.drop_self_in_neighborhood:
                neighborhood_nodes = neighborhood_nodes.drop(node)
            # TODO could be generalized to other permutation invariant aggregations
            agg_neighbor_features = neighborhood_nodes.mean(skipna=True).to_frame().T
            rows.append(agg_neighbor_features)
        neighborhood_features = pd.concat(rows)
        neighborhood_features.rename(
            columns=lambda x: f"{x}_neighbor_agg", inplace=True
        )
        neighborhood_features.index = list(neighborhoods.keys())
        neighborhood_features.index.name = "l2_id"
        return neighborhood_features


def _neighborhoods_to_features(neighborhoods):
    rows = []
    for node, neighborhood in neighborhoods.items():
        self_features = (
            neighborhood.nodes.loc[node].drop(["root_id", "compartment"]).to_frame().T
        )
        neighbor_features = neighborhood.nodes.drop(node).drop(
            ["root_id", "compartment"], axis=1
        )
        agg_neighbor_features = neighbor_features.mean(skipna=True).to_frame().T
        agg_neighbor_features.index = [node]
        features = self_features.join(agg_neighbor_features, rsuffix="_neighbor_agg")
        rows.append(features)

    return pd.concat(rows)
