# %%

from tqdm.auto import tqdm

from pkg.neuronframe import load_neuronframe
from pkg.sequence import create_lumped_time_sequence
from pkg.utils import load_manifest

manifest = load_manifest()
manifest = manifest.query("in_inhibitory_column")

# %%

# for root_id in tqdm(manifest.index):
#     nf = load_neuronframe(root_id)
#     sequence = create_lumped_time_sequence(nf, root_id)
#     break


def run_for_root_id(root_id):
    nf = load_neuronframe(root_id)
    sequence = create_lumped_time_sequence(nf, root_id)


# with tqdm_joblib(total=len(manifest.index)) as pbar:
#     Parallel(n_jobs=4)(
#         delayed(run_for_root_id)(root_id) for root_id in manifest.index
#     )

for root_id in tqdm(manifest.index):
    run_for_root_id(root_id)
