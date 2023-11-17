# %%
import caveclient as cc
import numpy as np
import pandas as pd

client = cc.CAVEclient("minnie65_phase3_v1")

# this is just a collection of L2 nodes that have been modified at some point that I
# was working with
all_modified_nodes = pd.read_csv(
    "https://raw.githubusercontent.com/bdpedigo/skedits/main/modified_l2_nodes.csv",
    index_col=0,
).index

# grab the coordinates of all these L2 nodes (in a list of about 4k nodes)
raw_node_coords = client.l2cache.get_l2data(
    all_modified_nodes.to_list(), attributes=["rep_coord_nm"]
)

# just putting them in a dataframe
node_coords = pd.DataFrame(raw_node_coords).T
node_coords.index = node_coords.index.astype(int)

# now, for this peculiar node, query its coordinates individually from the L2 cache, as
# well as check its position in the dataframe above
#
# note this is just one example node that I found that has this inconsistency
query_l2 = 156098560020972550

solo_query_coords = client.l2cache.get_l2data([query_l2], attributes=["rep_coord_nm"])[
    str(query_l2)
]["rep_coord_nm"]

group_query_coords = node_coords.loc[query_l2, "rep_coord_nm"]

print("Coordinates from querying in a group:", group_query_coords)
print("Coordinates from querying solo:", solo_query_coords)
print("Coordinates are the same? ", np.allclose(group_query_coords, solo_query_coords))
print()

point = np.array(node_coords.loc[query_l2, "rep_coord_nm"])
point /= np.array([8, 8, 40])
print("Coordinates from group query, divided by (8x8x40):", list(point))
