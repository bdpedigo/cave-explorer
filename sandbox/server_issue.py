#%%
import caveclient as cc

client = cc.CAVEclient("minnie65_phase3_v1")

proofreading_table = client.materialize.query_table(
    "proofreading_status_public_release"
)
