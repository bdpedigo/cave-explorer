# %%

import caveclient as cc
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm.auto import tqdm

from pkg.neuronframe import NeuronFrameSequence, load_neuronframe
from pkg.sequence import create_merge_and_clean_sequence
from pkg.utils import load_manifest

manifest = load_manifest()

client = cc.CAVEclient("minnie65_phase3_v1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%


def unwrap_pca(pca):
    if np.isnan(pca).all():
        return np.zeros(9)
    return np.array(pca).ravel()


def unwrap_pca_val(pca):
    if np.isnan(pca).all():
        return np.zeros(3)
    return np.array(pca).ravel()


def process_node_data(node_data):
    scalar_features = node_data[
        ["area_nm2", "max_dt_nm", "mean_dt_nm", "size_nm3"]
    ].astype(float)

    pca_unwrapped = np.stack(node_data["pca"].apply(unwrap_pca).values)
    pca_unwrapped = pd.DataFrame(
        pca_unwrapped,
        columns=[f"pca_unwrapped_{i}" for i in range(9)],
        index=node_data.index,
    )
    pca_val_unwrapped = np.stack(node_data["pca_val"].apply(unwrap_pca_val).values)
    pca_val_unwrapped = pd.DataFrame(
        pca_val_unwrapped,
        columns=[f"pca_val_unwrapped_{i}" for i in range(3)],
        index=node_data.index,
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


from requests.exceptions import HTTPError

graphs = []
for root_id in manifest.query("is_sample").index[:]:
    neuron = load_neuronframe(root_id, client)
    sequence = create_merge_and_clean_sequence(neuron, root_id)

    node_id_to_iloc_map = {node_id: i for i, node_id in enumerate(neuron.nodes.index)}
    unique_nodes = neuron.nodes.index.to_list()
    features = [
        "area_nm2",
        "max_dt_nm",
        "mean_dt_nm",
        "pca",
        "pca_val",
        "rep_coord_nm",
        "size_nm3",
    ]
    try:
        node_data = pd.DataFrame(
            client.l2cache.get_l2data(unique_nodes, attributes=features)
        ).T
    except HTTPError:
        print(f"Error loading node data for {root_id}")
        continue
    node_data.index = node_data.index.astype(int)
    node_data.index.name = "node_id"

    clean_node_data = process_node_data(node_data)
    features = clean_node_data.columns

    neuron.nodes = neuron.nodes.join(clean_node_data, how="left")
    # neuron.nodes["node_iloc"] = neuron.nodes.index.map(node_id_to_iloc_map)

    def convert_to_torch(neuron, directed=False):
        clean_node_data = neuron.nodes[features]
        node_mapping = dict(zip(neuron.nodes.index, range(len(neuron.nodes))))
        clean_edgelist = (
            neuron.edges[["source", "target"]].applymap(node_mapping.get).values
        )
        if directed:
            edge_index = torch.tensor(clean_edgelist.T, dtype=torch.long)
        else:
            edge_index = torch.tensor(
                np.concatenate([clean_edgelist.T, clean_edgelist[:, ::-1].T], axis=1),
                dtype=torch.long,
            )

        x = torch.tensor(clean_node_data.values, dtype=torch.float)
        return x, edge_index

    new_sequence = NeuronFrameSequence(
        neuron.deepcopy(), prefix="meta", edit_label_name="metaoperation_id"
    )
    sequence_info = sequence.sequence_info
    for i in tqdm(range(1, len(sequence))):
        applied_edits = sequence_info.iloc[i].loc["edit_ids_added"]
        new_sequence.apply_edits(applied_edits)
        current_neuron = new_sequence.current_resolved_neuron
        x, edge_index = convert_to_torch(current_neuron)
        ops = sequence_info.iloc[i].loc["cumulative_n_operations"]
        y = torch.tensor([[ops]], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        data = data.to(device)
        graphs.append(data)
        if edge_index.max() > len(x):
            print("error")
            break

# %%
from sklearn.model_selection import train_test_split

# TODO
# this is a dumb splitting since some neurons repeated in the sequence
# should really be split by neuron
# just want to see if its possible even when cheating
train_graphs, test_graphs = train_test_split(graphs, test_size=0.2)


# %%

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

# save the train and test loaders
torch.save(train_loader, "train_loader.pt")
torch.save(test_loader, "test_loader.pt")


# %%

N_NODE_FEATURES = len(train_graphs[0].x[0])
# N_CLASSES = 2

# create a graph neural network for regression on y variable of each graph
# using PyTorch Geometric

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=16):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(N_NODE_FEATURES, hidden_channels)
        self.mlp1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.mlp2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.mlp3 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.mlp1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # 2. Readout layer
        x = global_mean_pool(x, batch)

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.lin(x)

        return x


model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# %%
# model.train()
# for data in train_loader:
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.mse_loss(out, data.y)
#     loss.backward()
#     optimizer.step()
#     print(loss.item())

criterion = torch.nn.MSELoss()


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
    # evaluate MSE
    mse = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        mse += F.mse_loss(out, data.y)
    return mse / len(loader)


rows = []

for epoch in range(1, 200):
    train()
    train_mse = test(train_loader)
    test_mse = test(DataLoader(test_graphs, batch_size=1))
    print(f"Epoch: {epoch:03d}, Train MSE: {train_mse}, Test MSE: {test_mse}")
    rows.append(
        {"epoch": epoch, "train_mse": train_mse.item(), "test_mse": test_mse.item()}
    )
    if epoch % 10 == 0 or epoch == 1:
        for example in np.random.choice(len(test_graphs), 5, replace=False):
            data = test_graphs[example]
            out = model(data.x, data.edge_index, data.batch)
            print(f"Example {example}: {out.item()} vs {data.y.item()}")

progress = pd.DataFrame(rows)


# %%
data = test_graphs[20]
out = model(data.x, data.edge_index, data.batch)
out

# %%
data
test_graphs[0].y

# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 1, figsize=(5, 4))
sns.lineplot(data=progress, x="epoch", y="train_mse", label="train", ax=ax)
sns.lineplot(data=progress, x="epoch", y="test_mse", label="test", ax=ax)
ax.set_yscale("log")
