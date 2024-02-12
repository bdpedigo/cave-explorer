from typing import Literal, Optional, Self, Union

import numpy as np
import pandas as pd

from ..edits import count_synapses_by_sample
from ..utils import find_closest_point
from .neuronframe import NeuronFrame

Hashable = Union[
    str,
    int,
    float,
    bool,
    np.integer,
    tuple[str, ...],
    tuple[int, ...],
    tuple[float, ...],
    tuple[bool, ...],
    tuple[np.integer, ...],
]


class NeuronFrameSequence:
    def __init__(
        self,
        base_neuron: NeuronFrame,
        prefix="",
        edit_label_name=None,
    ):
        self.base_neuron = base_neuron
        self.prefix = prefix
        self.applied_edit_ids = pd.Index([])
        self.unresolved_sequence = {}
        self.resolved_sequence = {}
        self._sequence_info = {}
        self.edit_label_name = edit_label_name
        self.apply_edits(self.applied_edit_ids, label=None)

    def __len__(self) -> int:
        return len(self._sequence_info)

    def __repr__(self) -> str:
        base_repr = self.base_neuron.__repr__()
        new_repr = ""
        for para in base_repr.splitlines():
            new_repr += "\n\t\t" + para
        out = "NeuronFrameSequence(\n"
        out += f"\tbase_neuron={new_repr},\n"
        out += f"\tprefix={self.prefix},\n"
        out += f"\tedit_label_name={self.edit_label_name},\n"
        out += f"\tsequence=({len(self)},)\n"
        out += ")"
        return out

    @property
    def edits(self) -> pd.DataFrame:
        if self.prefix == "meta":
            edits = self.base_neuron.metaedits
        else:
            edits = self.base_neuron.edits
            edits["has_split"] = ~edits["is_merge"]
            edits["has_merge"] = edits["is_merge"]
            edits["n_operations"] = np.ones(len(edits), dtype=int)
        return edits

    # @beartype
    # @edits.setter
    # def edits(self, edits: pd.DataFrame):
    #     self.base_neuron.edits = edits

    @property
    def split_edits(self) -> pd.DataFrame:
        return self.edits.query("has_split")

    @property
    def merge_edits(self) -> pd.DataFrame:
        return self.edits.query("has_merge")

    @property
    def applied_edits(self) -> pd.DataFrame:
        return self.edits.loc[self.applied_edit_ids]

    @property
    def unapplied_edits(self) -> pd.DataFrame:
        return self.edits.loc[~self.edits.index.isin(self.applied_edit_ids)]

    @property
    def latest_label(self) -> Optional[Hashable]:
        if len(self._sequence_info) == 0:
            return None
        else:
            return list(self._sequence_info.keys())[-1]

    @property
    def current_unresolved_neuron(self) -> Self:
        return self.unresolved_sequence[self.latest_label]

    @property
    def current_resolved_neuron(self) -> Self:
        return self.resolved_sequence[self.latest_label]

    def apply_edits(
        self,
        edits: Union[
            list, np.ndarray, pd.Index, pd.Series, int, np.integer, pd.DataFrame
        ],
        label: Optional[Hashable] = None,
    ) -> None:
        if label is None and isinstance(edits, (int, np.integer)):
            label = edits
            edit_ids = edits
        if isinstance(edits, (int, np.integer)):
            edit_ids = pd.Index([edits])
        elif isinstance(edits, (list, np.ndarray)):
            edit_ids = pd.Index(edits)
        elif isinstance(edits, (pd.Series, pd.DataFrame)):
            edit_ids = edits.index
        else:  # pd.Index
            edit_ids = edits

        # TODO add some logic for keeping track of the edit IDs activated at each step
        # perhaps keep track of "resolved" and "unresolved" versions of the neuron?
        is_used = edit_ids.isin(self.applied_edit_ids)
        if is_used.any():
            print(
                f"WARNING: Some edit IDs {list(edit_ids[is_used])} have already been applied."
            )

        self.applied_edit_ids = self.applied_edit_ids.append(edit_ids)

        unresolved_neuron = self.base_neuron.set_edits(
            self.applied_edit_ids, inplace=False, prefix=self.prefix
        )

        resolved_neuron = resolve_neuron(unresolved_neuron, self.base_neuron)

        self.unresolved_sequence[label] = unresolved_neuron
        self.resolved_sequence[label] = resolved_neuron

        self._sequence_info[label] = {
            "edit_ids_added": edit_ids.to_list(),
            "applied_edits": self.applied_edit_ids.to_list(),
            "pre_synapses": resolved_neuron.pre_synapses.index.to_list(),
            "post_synapses": resolved_neuron.post_synapses.index.to_list(),
            "n_nodes": len(resolved_neuron),
            "path_length": resolved_neuron.path_length,
            "order": len(self._sequence_info),
        }

        return None

    @property
    def sequence_info(self) -> pd.DataFrame:
        sequence_info = pd.DataFrame(self._sequence_info).T
        sequence_info.index.name = self.edit_label_name
        sequence_info.index = sequence_info.index.astype("Int64")

        sequence_info["n_pre_synapses"] = sequence_info["pre_synapses"].apply(len)
        sequence_info["n_post_synapses"] = sequence_info["post_synapses"].apply(len)

        sequence_info = sequence_info.join(self.edits)
        sequence_info["cumulative_n_operations"] = sequence_info[
            "n_operations"
        ].cumsum()

        return sequence_info

    @property
    def final_neuron(self):
        final_neuron = self.base_neuron.set_edits(
            self.edits.index, inplace=False, prefix=self.prefix
        )
        final_neuron = resolve_neuron(final_neuron, self.base_neuron)
        return final_neuron

    @property
    def is_completed(self):
        return self.final_neuron == self.current_resolved_neuron

    def synapse_groupby_count(
        self, by: str, which: Literal["pre", "post"]
    ) -> pd.DataFrame:
        if which == "pre":
            synapses = self.base_neuron.pre_synapses
            resolved = self.sequence_info["pre_synapses"]
        else:
            synapses = self.base_neuron.post_synapses
            resolved = self.sequence_info["post_synapses"]

        counts = count_synapses_by_sample(synapses, resolved, by=by)
        counts.index.name = self.edit_label_name
        return counts


def resolve_neuron(unresolved_neuron, base_neuron):
    if base_neuron.nucleus_id in unresolved_neuron.nodes.index:
        resolved_neuron = unresolved_neuron.select_nucleus_component(inplace=False)
    else:
        print("WARNING: Using closest point to nucleus to resolve neuron...")
        point_id = find_closest_point(
            unresolved_neuron.nodes,
            base_neuron.nodes.loc[base_neuron.nucleus_id, ["x", "y", "z"]],
        )
        resolved_neuron = unresolved_neuron.select_component_from_node(
            point_id, inplace=False, directed=False
        )

    resolved_neuron.remove_unused_synapses(inplace=True)
    return resolved_neuron
