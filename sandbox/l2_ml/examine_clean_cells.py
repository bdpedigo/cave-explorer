# %%
import numpy as np

from pkg.utils import load_manifest

manifest = load_manifest()

ids = [
    864691135163673901,
    864691135617152361,
    864691136090326071,
    864691135565870679,
    864691135510120201,
    864691135214129208,
    864691135759725134,
    864691135256861871,
    864691135759685966,
    864691135946980644,
    864691134941217635,
    864691136275234061,
    864691135741431915,
    864691135361314119,
    864691135777918816,
    864691136137805181,
    864691135737446276,
    864691136451680255,
    864691135468657292,
    864691135578006277,
    864691136452245759,
    864691135916365670,
]

np.isin(ids, manifest.index)

# %%
import re

from cloudfiles import CloudFiles

ground_truth_path = "gs://allen-minnie-phase3/minniephase3-emily-pcg-skeletons/axon_dendrite_classifier/groundtruth_feats_and_class"
ground_truth_cf = CloudFiles(ground_truth_path)

ground_truth_root_ids = [
    int(name.split("_")[0]) for name in ground_truth_cf.list() if name.endswith(".csv")
]


def string_to_list(string):
    string = re.sub("\s+", ",", string)
    if string.startswith("[,"):
        string = "[" + string[2:]
    return eval(string)


# %%
from caveclient import CAVEclient

from pkg.constants import NUCLEUS_TABLE

client = CAVEclient("minnie65_phase3_v1")

nucs = client.materialize.query_table(NUCLEUS_TABLE)
nucs.drop_duplicates("pt_root_id", inplace=True)
nucs.set_index("pt_root_id", inplace=True)

# %%
new_root_map = {}
for root in ground_truth_root_ids[4:]:
    possible_new_roots = client.chunkedgraph.get_latest_roots(root)
    is_targeted = np.isin(possible_new_roots, nucs.index)
    if is_targeted.sum() == 1: 
        new_root_map[root] = possible_new_roots[is_targeted][0]
    else: 
        print('root:', root)
        print(is_targeted)
        print()

#%%
new_root_ids = list(new_root_map.values())


#%%
np.isin(new_root_ids, manifest.index)

#%%
from pkg.constants import PROOFREADING_TABLE

proofreading_df = client.materialize.query_table(PROOFREADING_TABLE)
    
#%%

np.isin(new_root_ids, proofreading_df.pt_root_id)