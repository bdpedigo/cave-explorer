from typing import Literal, Optional, Union

import numpy as np
from cloudfiles import CloudFiles
from tqdm.auto import tqdm

from ..io import lazycloud
from ..neuronframe import NeuronFrame, NeuronFrameSequence


@lazycloud(
    cloud_bucket="allen-minnie-phase3",
    folder="edit_sequences",
    file_suffix="time_ordered_sequence.pkl",
    kwarg_keys=["root_id"],
    save_format="pickle",
)
def _create_time_ordered_sequence_dict(
    neuron: NeuronFrame,
    root_id: Optional[int] = None,
    use_cache: bool = True,
    cache_verbose: bool = False,
    only_load: bool = False,
) -> dict:
    root_id
    use_cache
    cache_verbose
    only_load

    neuron_sequence = NeuronFrameSequence(
        neuron, prefix="", edit_label_name="operation_id"
    )
    neuron_sequence.edits.sort_values("time", inplace=True)

    for i in tqdm(range(len(neuron_sequence.edits))):
        operation_id = neuron_sequence.edits.index[i]
        neuron_sequence.apply_edits(operation_id)

    if not neuron_sequence.is_completed:
        raise UserWarning("Neuron is not completed.")

    return neuron_sequence.to_dict()


def create_time_ordered_sequence(
    neuron: NeuronFrame,
    root_id: Optional[int] = None,
    use_cache: bool = True,
    cache_verbose: bool = False,
    only_load: bool = False,
) -> NeuronFrameSequence:
    info = _create_time_ordered_sequence_dict(
        neuron,
        root_id=root_id,
        use_cache=use_cache,
        cache_verbose=cache_verbose,
        only_load=only_load,
    )
    if info is None:
        return None
    else:
        return NeuronFrameSequence.from_dict_and_neuron(info, neuron)


@lazycloud(
    cloud_bucket="allen-minnie-phase3",
    folder="edit_sequences",
    file_suffix="merge_and_clean_sequence.pkl",
    kwarg_keys=["root_id", "order_by", "random_seed"],
    save_format="pickle",
)
def _create_merge_and_clean_sequence_dict(
    neuron,
    root_id: Optional[int] = None,
    order_by: Literal["time", "random"] = "time",
    random_seed: Optional[Union[int, np.integer]] = None,
    use_cache: bool = True,
    cache_verbose: bool = False,
) -> dict:
    root_id
    use_cache
    cache_verbose

    neuron_sequence = NeuronFrameSequence(
        neuron, prefix="meta", edit_label_name="metaoperation_id"
    )
    if order_by == "time":
        neuron_sequence.edits.sort_values(["has_merge", "time"], inplace=True)
    elif order_by == "random":
        rng = np.random.default_rng(random_seed)
        neuron_sequence.edits["random"] = rng.random(len(neuron_sequence.edits))
        neuron_sequence.edits.sort_values(["has_merge", "random"], inplace=True)

    i = 0
    next_operation = True
    pbar = tqdm(total=len(neuron_sequence.edits), desc="Applying edits...")
    while next_operation is not None:
        possible_edit_ids = neuron_sequence.find_incident_edits()
        if len(possible_edit_ids) == 0:
            next_operation = None
        else:
            next_operation = possible_edit_ids[0]
            neuron_sequence.apply_edits(next_operation)
        i += 1
        pbar.update(1)
    pbar.close()

    if not neuron_sequence.is_completed:
        raise UserWarning("Neuron is not completed.")

    return neuron_sequence.to_dict()


def create_merge_and_clean_sequence(
    neuron,
    root_id: Optional[int] = None,
    order_by: Literal["time", "random"] = "time",
    random_seed: Optional[Union[int, np.integer]] = None,
    use_cache: bool = True,
    cache_verbose: bool = False,
) -> NeuronFrameSequence:
    info = _create_merge_and_clean_sequence_dict(
        neuron,
        root_id=root_id,
        order_by=order_by,
        random_seed=random_seed,
        use_cache=use_cache,
        cache_verbose=cache_verbose,
    )
    return NeuronFrameSequence.from_dict_and_neuron(info, neuron)


def load_sequences(root_id, client=None):
    cloud_bucket = "allen-minnie-phase3"
    folder = "edit_sequences"

    cf = CloudFiles(f"gs://{cloud_bucket}/{folder}")

    files = cf.list_files(prefix=f"{root_id}=")
