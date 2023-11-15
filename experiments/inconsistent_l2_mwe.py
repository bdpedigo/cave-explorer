# %%
import caveclient as cc
import pandas as pd

client = cc.CAVEclient("minnie65_phase3_v1")

all_modified_nodes = pd.read_csv(
    "https://raw.githubusercontent.com/bdpedigo/skedits/main/modified_l2_nodes.csv",
    index_col=0,
).index

raw_node_coords = client.l2cache.get_l2data(
    all_modified_nodes.to_list(), attributes=["rep_coord_nm"]
)

node_coords = pd.DataFrame(raw_node_coords).T
node_coords.index = node_coords.index.astype(int)

query_l2 = 156098560020972550

print("Coordinates from querying as list:", node_coords.loc[query_l2, "rep_coord_nm"])
print(
    "Coordinates from querying solo:",
    client.l2cache.get_l2data([query_l2], attributes=["rep_coord_nm"])[str(query_l2)][
        "rep_coord_nm"
    ],
)
