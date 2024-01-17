# %%
import caveclient as cc

client = cc.CAVEclient("minnie65_public")


min_x = 208876 - 10000 / 4
min_y = 86590 - 10000 / 4
min_z = 24931 - 10000 / 40
max_x = 208876 + 10000 / 4
max_y = 86590 + 10000 / 4
max_z = 24931 + 10000 / 40
bounding_box = [[min_x, min_y, min_z], [max_x, max_y, max_z]]

synapse_table = client.info.get_datastack_info()["synapse_table"]

df = client.materialize.query_table(
    synapse_table,
    filter_equal_dict={"pre_pt_root_id": 864691135697251738},
    filter_spatial_dict={"pre_pt_position": bounding_box},
)
