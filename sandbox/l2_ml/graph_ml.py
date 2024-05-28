# %%
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from caveclient import CAVEclient
from skops.io import load
from tqdm.auto import tqdm
from troglobyte.features import CAVEWrangler

from pkg.neuronframe import load_neuronframe
from pkg.utils import load_manifest

# %%
manifest = load_manifest()

client = CAVEclient("minnie65_phase3_v1")

root_ids = manifest.index[1:2]
neuronframes = []
new_root_ids = []
for root_id in tqdm(root_ids):
    nf = load_neuronframe(root_id, client=client, only_load=True)
    if nf is not None:
        neuronframes.append(nf)
        new_root_ids.append(root_id)

root_ids = new_root_ids

# %%
splits = nf.edits.query("~is_merge").dropna(subset="centroid_x")

removed_edges = nf.edges.query("operation_removed.isin(@splits.index)")
added_edges = nf.edges.query("operation_added.isin(@splits.index)")

# %%
edges_removed_by_operation = removed_edges.groupby("operation_removed").groups
edges_added_by_operation = added_edges.groupby("operation_added").groups

# %%

# going to query the object right before, at that edit
before_roots = splits["before_root_ids"].explode()
points_by_root = {}
for operation_id, before_root in before_roots.items():
    point = splits.loc[operation_id, ["centroid_x", "centroid_y", "centroid_z"]]
    points_by_root[before_root] = point.values
points_by_root = pd.Series(points_by_root)
# %%


model_path = Path("data/models/local_compartment_classifier_bd_boxes.skops")

model = load(model_path)

neighborhood_hops = 5
box_width = 20_000
verbose = False
split_wrangler = CAVEWrangler(client=client, n_jobs=-1, verbose=verbose)
split_wrangler.set_objects(before_roots.to_list())
split_wrangler.set_query_boxes_from_points(points_by_root, box_width=box_width)
split_wrangler.query_level2_shape_features()
split_wrangler.prune_query_to_boxes()
split_wrangler.query_level2_synapse_features(method="update")
split_wrangler.register_model(model, "bd_boxes")
split_wrangler.query_level2_edges(warn_on_missing=False)
split_wrangler.query_level2_networks()
split_wrangler.query_level2_graph_features()
split_wrangler.aggregate_features_by_neighborhood(
    aggregations=["mean", "std"], neighborhood_hops=neighborhood_hops
)


# %%

from torch_geometric.data import Data


def networkframe_to_torch_geometric(networkframe, directed=False, weight_col=None):
    nodes = networkframe.nodes
    edges = networkframe.edges
    if isinstance(edges, list):
        edges = np.array(edges)

    # remapped_edges = edges.copy()
    edgelist = edges[["source", "target"]].values
    remapped_sources = nodes.index.get_indexer_for(edgelist[:, 0])
    remapped_targets = nodes.index.get_indexer_for(edgelist[:, 1])
    remapped_edges = np.stack([remapped_sources, remapped_targets], axis=1)
    # remapped_edges[["source", "target"]] = remapped_edges[["source", "target"]].apply(
    #     nodes.index.get_indexer_for
    # )
    remapped_nodes = nodes.reset_index(drop=True)

    if directed:
        edge_index = torch.tensor(remapped_edges.T, dtype=torch.long)
    else:
        edge_index = torch.tensor(
            np.concatenate([remapped_edges.T, remapped_edges[:, ::-1].T], axis=1),
            dtype=torch.long,
        )

    if weight_col is not None:
        if directed:
            edge_attr = torch.tensor(
                edges[weight_col].values, dtype=torch.float
            ).unsqueeze(1)
        else:
            edge_attr = torch.tensor(
                np.concatenate([edges[weight_col].values, edges[weight_col].values]),
                dtype=torch.float,
            ).unsqueeze(1)
    else:
        edge_attr = None

    x = torch.tensor(nodes.fillna(0.0).values, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


from networkframe import NetworkFrame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

local_nfs = split_wrangler.object_level2_networks_
new_local_nfs = {}
torch_data = {}
for operation in before_roots.index:
    before_root = before_roots[operation]
    local_nf = local_nfs[before_root]
    local_nf = NetworkFrame(local_nf.nodes.copy(), local_nf.edges.copy())
    local_nf.edges.set_index(["source", "target"], inplace=True, drop=False)
    local_nf.edges["removed"] = 0.0
    local_nf.edges.loc[edges_removed_by_operation[operation], "removed"] = 1.0
    new_local_nfs[operation] = local_nf
    data = networkframe_to_torch_geometric(
        local_nf, directed=False, weight_col="removed"
    )
    torch_data[operation] = data

local_nfs = new_local_nfs

# %%
import pyvista as pv

boxes = split_wrangler.query_boxes_

pv.set_jupyter_backend("client")
plotter = pv.Plotter()

for box in boxes:
    box_mesh = pv.Box(
        [box[0, 0], box[1, 0], box[0, 1], box[1, 1], box[0, 2], box[1, 2]]
    )
    plotter.add_mesh(box_mesh, color="red", opacity=0.5, style="wireframe")


def _edges_to_lines(nodes, edges):
    iloc_map = dict(zip(nodes.index.values, range(len(nodes))))
    iloc_edges = edges[["source", "target"]].applymap(lambda x: iloc_map[x])

    lines = np.empty((len(edges), 3), dtype=int)
    lines[:, 0] = 2
    lines[:, 1:3] = iloc_edges[["source", "target"]].values

    return lines


def to_skeleton_polydata(
    nf: NetworkFrame,
    spatial_columns=["rep_coord_x", "rep_coord_y", "rep_coord_z"],
    label=None,
    draw_lines=True,
) -> pv.PolyData:
    nodes = nf.nodes
    edges = nf.edges

    points = nodes[spatial_columns].values.astype(float)

    if draw_lines:
        lines = _edges_to_lines(nodes, edges)
    else:
        lines = None

    skeleton = pv.PolyData(points, lines=lines)

    if label is not None:
        skeleton[label] = nodes[label].values

    return skeleton


for nf in local_nfs.values():
    nf = nf.largest_connected_component()
    skel_poly = to_skeleton_polydata(nf)
    plotter.add_mesh(
        skel_poly,
        scalars=nf.edges["removed"].values,
        render_lines_as_tubes=True,
        line_width=3,
        # color='black'
    )
    plotter.add_mesh(
        skel_poly,
        point_size=7,
        scalars=nf.edges["removed"].values,
        style="points",
        render_points_as_spheres=True,
        # color='black'
    )

plotter.show()

# %%

# create a simple graph neural network for edge classification with pytorch geometric

import torch
from torch.nn import Linear

N_NODE_FEATURES = data.x.shape[1]


class Net(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        # self.conv1 = GCNConv(N_NODE_FEATURES, hidden_channels)
        # self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.mlp1 = Linear(N_NODE_FEATURES, hidden_channels)
        self.mlp2 = Linear(hidden_channels, out_channels)

    # def encode(self, x, edge_index):
    #     x = self.conv1(x, edge_index).relu()
    #     return self.conv2(x, edge_index)
    def encode(self, x, edge_index):
        # x = self.conv1(x, edge_index).relu()
        # x = self.conv2(x, edge_index).relu()
        x = self.mlp1(x).relu()
        return self.mlp2(x)

    def decode(self, z, edge_label_index):
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


from torch_geometric.loader import DataLoader

graphs = list(torch_data.values())
# graphs = [T.NormalizeFeatures()(graph) for graph in graphs]

train_loader = DataLoader(graphs, batch_size=4, shuffle=True)

# set up the model and the optimizer

model = Net(128, 16).to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
criterion = torch.nn.BCEWithLogitsLoss()


from sklearn.metrics import roc_auc_score


def train(train_loader):
    model.train()

    for train_data in train_loader:
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)

        edge_label_index = train_data.edge_index
        edge_label = train_data.edge_attr.squeeze(1)

        out = model.decode(z, edge_label_index).view(-1)
        loss = criterion(out, edge_label)
        loss.backward()
        optimizer.step()
    return loss


@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_label_index).view(-1).sigmoid()
    return roc_auc_score(data.edge_label.cpu().numpy(), out.cpu().numpy())


for epoch in range(1, 1000):
    loss = train(train_loader)


    # val_auc = test(val_data)
    # test_auc = test(test_data)
    # if val_auc > best_val_auc:
    #     best_val_auc = val_auc
    #     final_test_auc = test_auc
    if epoch % 10 == 0:
        print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}")

z = model.encode(data.x, data.edge_index)
out = model.decode(z, data.edge_index).sigmoid()
print(out)
# %%
# final prediction

for data in graphs:
    z = model.encode(data.x, data.edge_index)
    out = model.decode(z, data.edge_index).sigmoid().detach().cpu().numpy().reshape(-1)
    print(out)
    print(data.edge_attr.cpu().numpy())
    print()

# %%
