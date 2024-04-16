# %%


import caveclient as cc

from pkg.metrics import annotate_mtypes, annotate_pre_synapses
from pkg.neuronframe import NeuronFrameSequence, load_neuronframe
from pkg.plot import set_context
from pkg.utils import load_manifest, load_mtypes

# %%

set_context()

client = cc.CAVEclient("minnie65_phase3_v1")
mtypes = load_mtypes(client)

manifest = load_manifest()

# %%

example_root_ids = manifest.query("is_sample").index

# %%

root_id = example_root_ids[2]
neuron = load_neuronframe(root_id, client)
mtypes = load_mtypes(client)

annotate_pre_synapses(neuron, mtypes)
annotate_mtypes(neuron, mtypes)

# %%
sequence = NeuronFrameSequence(
    neuron, prefix="meta", edit_label_name="metaoperation_id_dropped"
)

from tqdm.autonotebook import tqdm

pre_synapse_sets = {}
post_synapse_sets = {}
metaedits = neuron.metaedits
for metaoperation_id, metaedit in tqdm(metaedits.iterrows(), total=len(metaedits)):
    edits_to_apply = metaedits.query("metaoperation_id != @metaoperation_id").index
    sequence.apply_edits(edits_to_apply, label=metaoperation_id, replace=True)

sequence.apply_edits(metaedits.index, label=None, replace=True)


# %%

from pkg.metrics import (
    compute_counts,
    compute_spatial_target_proportions,
    compute_target_counts,
    compute_target_proportions,
)

sequence_feature_dfs = {}
counts = sequence.apply_to_synapses_by_sample(
    compute_counts, which="pre", output="scalar", name="count"
)
sequence_feature_dfs["counts"] = counts

counts_by_mtype = sequence.apply_to_synapses_by_sample(
    compute_target_counts, which="pre", by="post_mtype"
)
sequence_feature_dfs["counts_by_mtype"] = counts_by_mtype

props_by_mtype = sequence.apply_to_synapses_by_sample(
    compute_target_proportions, which="pre", by="post_mtype"
)
sequence_feature_dfs["props_by_mtype"] = props_by_mtype

spatial_props = sequence.apply_to_synapses_by_sample(
    compute_spatial_target_proportions, which="pre", mtypes=mtypes
)
sequence_feature_dfs["spatial_props"] = spatial_props

spatial_props_by_mtype = sequence.apply_to_synapses_by_sample(
    compute_spatial_target_proportions,
    which="pre",
    mtypes=mtypes,
    by="post_mtype",
)
sequence_feature_dfs["spatial_props_by_mtype"] = spatial_props_by_mtype
