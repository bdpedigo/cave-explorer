# %%
import os

import caveclient as cc
import pandas as pd

from pkg.neuronframe import load_neuronframe

os.environ["LAZYCLOUD_USE_CLOUD"] = "True"

client = cc.CAVEclient("minnie65_phase3_v1")

load_neuronframe(864691135013628918, client)
# %%
proofreading_table = client.materialize.query_table(
    "proofreading_status_public_release"
)
proofreading_table = proofreading_table.set_index("pt_root_id").sort_index()

# %%
edit_counts = {}
for root_id in proofreading_table.index[:50]:
    lineage = client.chunkedgraph.get_lineage_graph(root_id, as_nx_graph=True)
    edit_counts[root_id] = len(lineage)
edit_counts = pd.Series(edit_counts).sort_values(ascending=True)


# %%
base_halfwidth = 20_000
for root_id in edit_counts.index:
    print()
    print()
    try:
        nf = load_neuronframe(
            root_id,
            client,
            cache_verbose=True,
            use_cache=True,
            bounds_halfwidth=base_halfwidth,
        )
    except ValueError:
        print("trying 2x bbox halfwidth")
        try:
            nf = load_neuronframe(
                root_id,
                client,
                cache_verbose=True,
                use_cache=True,
                bounds_halfwidth=2 * base_halfwidth,
            )
        except ValueError:
            print("trying no bbox")
            try:
                nf = load_neuronframe(
                    root_id,
                    client,
                    cache_verbose=True,
                    use_cache=True,
                    bounds_halfwidth=None,
                )
            except ValueError as e:
                msg = "Wow, I tried really hard to load this NeuronFrame"
                raise e
    print()
    print()

#%%
load_neuronframe(864691135013628918, client)