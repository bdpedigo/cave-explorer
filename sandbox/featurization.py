# %%
import os
import time

import caveclient as cc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm.auto import tqdm

os.environ["TORCH"] = torch.__version__
print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

client = cc.CAVEclient("minnie65_phase3_v1")
seg_res = client.chunkedgraph.base_resolution

proofreading_table = client.materialize.query_table(
    "proofreading_status_public_release"
)

# %%


def get_bbox_cg(point_in_cg, bbox_halfwidth=10_000):
    point_in_nm = point_in_cg * seg_res

    x_center, y_center, z_center = point_in_nm

    x_start = x_center - bbox_halfwidth
    x_stop = x_center + bbox_halfwidth
    y_start = y_center - bbox_halfwidth
    y_stop = y_center + bbox_halfwidth
    z_start = z_center - bbox_halfwidth
    z_stop = z_center + bbox_halfwidth

    start_point_cg = np.round(np.array([x_start, y_start, z_start]) / seg_res)
    stop_point_cg = np.round(np.array([x_stop, y_stop, z_stop]) / seg_res)

    bbox_cg = np.array([start_point_cg, stop_point_cg], dtype=int).T

    return bbox_cg


def unwrap_pca(pca):
    if np.isnan(pca).all():
        return np.zeros(9)
    return np.array(pca).ravel()


def unwrap_pca_val(pca):
    if np.isnan(pca).all():
        return np.zeros(3)
    return np.array(pca).ravel()


def edgelist_to_torch(edgelist, directed=False):
    if isinstance(edgelist, list):
        edgelist = np.array(edgelist)

    unique_nodes = np.unique(edgelist)
    node_data = pd.DataFrame(client.l2cache.get_l2data(unique_nodes)).T

    index = node_data.index.copy().astype(int)
    node_data.reset_index(drop=True, inplace=True)

    scalar_features = node_data[
        ["area_nm2", "max_dt_nm", "mean_dt_nm", "size_nm3"]
    ].astype(float)

    pca_unwrapped = np.stack(node_data["pca"].apply(unwrap_pca).values)
    pca_unwrapped = pd.DataFrame(
        pca_unwrapped, columns=[f"pca_unwrapped_{i}" for i in range(9)]
    )
    pca_val_unwrapped = np.stack(node_data["pca_val"].apply(unwrap_pca_val).values)
    pca_val_unwrapped = pd.DataFrame(
        pca_val_unwrapped, columns=[f"pca_val_unwrapped_{i}" for i in range(3)]
    )

    rep_coord_unwrapped = np.stack(node_data["rep_coord_nm"].values)
    rep_coord_unwrapped = pd.DataFrame(
        rep_coord_unwrapped,
        columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"],
    )

    clean_node_data = pd.concat(
        [scalar_features, pca_unwrapped, pca_val_unwrapped, rep_coord_unwrapped], axis=1
    )

    index_to_iloc_mapping = dict(zip(index, clean_node_data.index))

    clean_edgelist = np.vectorize(index_to_iloc_mapping.get)(edgelist)

    if directed:
        edge_index = torch.tensor(clean_edgelist.T, dtype=torch.long)
    else:
        edge_index = torch.tensor(
            np.concatenate([clean_edgelist.T, clean_edgelist[:, ::-1].T], axis=1),
            dtype=torch.long,
        )

    x = torch.tensor(clean_node_data.values, dtype=torch.float)

    return x, edge_index, index_to_iloc_mapping


# %%
currtime = time.time()
client.chunkedgraph.get_tabular_change_log(864691134884807418)
print(f"{time.time() - currtime:.3f} seconds elapsed.")

# %%

graphs_by_operation = {}

bbox_halfwidth = 20_000

root_ids = proofreading_table["pt_root_id"].iloc[:20]

currtime = time.time()
change_logs = client.chunkedgraph.get_tabular_change_log(root_ids)
print(f"{time.time() - currtime:.3f} seconds elapsed to get change log.")

for root_id in root_ids:
    change_log = change_logs[root_id].set_index("operation_id")

    splits = change_log.query("~is_merge")

    # generate a dataset of split operations, centered around the point of the split
    # using the segmentation at the time of the split

    for operation_id in tqdm(splits.index):
        operation_id = change_log.index[0]
        row = change_log.loc[operation_id]
        details = client.chunkedgraph.get_operation_details([operation_id])[
            str(operation_id)
        ]

        point_in_cg = 0.5 * np.mean(details["sink_coords"], axis=0) + 0.5 * np.mean(
            details["source_coords"], axis=0
        )

        bbox_cg = get_bbox_cg(point_in_cg, bbox_halfwidth=bbox_halfwidth)

        root_at_operation_time = row["before_root_ids"][0]
        edgelist = client.chunkedgraph.level2_chunk_graph(
            root_at_operation_time, bounds=bbox_cg
        )

        x, edge_index, index_to_iloc_mapping = edgelist_to_torch(edgelist)

        # label these a 1 for split
        y = torch.tensor([1], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data = data.to(device)
        graphs_by_operation[operation_id] = data

    n_splits = len(splits)

    # generate an equally sized dataset of random points on the neuron, using the
    # segmentation at the final (cleaned) timepoint

    # get the edges at the final state
    full_edgelist = np.array(client.chunkedgraph.level2_chunk_graph(root_id))

    full_nodes_index = np.unique(full_edgelist)

    out_counts = pd.Series(full_edgelist[:, 0]).value_counts()
    in_counts = pd.Series(full_edgelist[:, 1]).value_counts()
    # query_nodes = out_counts[out_counts > 2 & out_counts < 4].index

    # get the elements of out counts that are between 2 and 4
    query_nodes = out_counts[(out_counts > 2) & (out_counts < 4)].index

    query_nodes = np.random.choice(query_nodes, size=n_splits)

    graphs_by_node = {}
    for node in tqdm(query_nodes):
        point_in_nm = np.array(
            client.l2cache.get_l2data(node)[str(node)]["rep_coord_nm"]
        )
        point_in_cg = point_in_nm / seg_res
        bbox_cg = get_bbox_cg(point_in_cg, bbox_halfwidth=bbox_halfwidth)

        edgelist = client.chunkedgraph.level2_chunk_graph(root_id, bounds=bbox_cg)

        x, edge_index, index_to_iloc_mapping = edgelist_to_torch(edgelist)

        # label these a 0 for not split
        y = torch.tensor([0], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        data = data.to(device)
        graphs_by_node[node] = data

# %%

# combine the two datasets into a format that can be used by PyTorch Geometric

graphs = list(graphs_by_operation.values()) + list(graphs_by_node.values())

train_loader = DataLoader(graphs, batch_size=32, shuffle=True)

# %%

# create a graph neural network model using PyTorch Geometric


N_NODE_FEATURES = len(graphs[0].x[0])
N_CLASSES = 2


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        # torch.manual_seed(12345)
        self.conv1 = GCNConv(N_NODE_FEATURES, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, N_CLASSES)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GCN(hidden_channels=16)
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(
            data.x, data.edge_index, data.batch
        )  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


print(test(train_loader))

# %%
for epoch in tqdm(range(200)):
    train()
    train_acc = test(train_loader)
    print(f"Epoch: {epoch}, Train Acc: {train_acc}")

# %%
