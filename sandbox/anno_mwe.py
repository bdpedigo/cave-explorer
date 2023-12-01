# %%

import caveclient as cc
import pandas as pd

import pcg_skel

client = cc.CAVEclient("minnie65_phase3_v1")

# %%
root_id = 864691135510015497

change_log = client.chunkedgraph.get_change_log(root_id)

details = client.chunkedgraph.get_operation_details(change_log["operations_ids"])

merges = {}
splits = {}
for operation, detail in details.items():
    operation = int(operation)
    source_coords = detail["source_coords"][0]
    sink_coords = detail["sink_coords"][0]
    x = (source_coords[0] + sink_coords[0]) / 2
    y = (source_coords[1] + sink_coords[1]) / 2
    z = (source_coords[2] + sink_coords[2]) / 2
    pt = [x, y, z]
    row = {"x": x, "y": y, "z": z, "pt": pt}
    if "added_edges" in detail:
        merges[operation] = row
    elif "removed_edges" in detail:
        splits[operation] = row

merges = pd.DataFrame.from_dict(merges, orient="index")
merges.index.name = "operation"
splits = pd.DataFrame.from_dict(splits, orient="index")
splits.index.name = "operation"

meshwork = pcg_skel.coord_space_meshwork(root_id, client=client)

meshwork.add_annotations("splits", splits, anchored=False)
meshwork.add_annotations("merges", merges, anchored=False)
