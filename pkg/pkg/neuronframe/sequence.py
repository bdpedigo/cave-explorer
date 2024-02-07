from typing import Optional, Union

import numpy as np
import pandas as pd
from beartype import beartype

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
    @beartype
    def __init__(
        self,
        base_neuron: NeuronFrame,
        prefix="",
        edit_label_name=None,
    ):
        self.base_neuron = base_neuron
        self.prefix = prefix
        self.applied_edit_ids = pd.Index([])
        self._edit_ids_added = {}
        self._applied_edit_history = {}
        self.unresolved_sequence = {}
        self.resolved_sequence = {}
        self._resolved_synapses = {}
        self.edit_label_name = edit_label_name

    def __len__(self):
        return len(self.applied_edit_history)

    def __repr__(self) -> str:
        base_repr = self.base_neuron.__repr__()
        new_repr = ""
        for para in base_repr.splitlines():
            new_repr += "\n\t\t" + para
        out = "NeuronFrameSequence(\n"
        out += f"\tbase_neuron={new_repr},\n"
        out += f"\tprefix={self.prefix},\n"
        out += f"\tapplied_edit_ids=({len(self.applied_edit_ids)},)\n"
        out += f"\tsequence=({len(self)},)\n"
        out += ")"
        return out

    @property
    def edits(self):
        if self.prefix == "meta":
            edits = self.base_neuron.metaedits
        else:
            edits = self.base_neuron.edits
            edits["has_split"] = ~edits["is_merge"]
            edits["has_merge"] = edits["is_merge"]

        return edits

    @beartype
    @edits.setter
    def edits(self, edits: pd.DataFrame):
        self.base_neuron.edits = edits

    @property
    def split_edits(self):
        return self.edits.query("has_split")

    @property
    def merge_edits(self):
        return self.edits.query("has_merge")

    @property
    def applied_edits(self):
        return self.edits.loc[self.applied_edit_ids]

    @property
    def unapplied_edits(self):
        return self.edits.loc[~self.edits.index.isin(self.applied_edit_ids)]

    @property
    def latest_label(self):
        return list(self.applied_edit_history.keys())[-1]

    @property
    def current_unresolved_neuron(self):
        return self.unresolved_sequence[self.latest_label]

    @property
    def current_resolved_neuron(self):
        return self.resolved_sequence[self.latest_label]

    @beartype
    def apply_edits(
        self,
        edits: Union[
            list, np.ndarray, pd.Index, pd.Series, int, np.integer, pd.DataFrame
        ],
        label: Optional[Hashable] = None,
    ) -> None:
        if label is None and isinstance(edits, (int, np.integer)):
            label = edits
        if isinstance(edits, (int, np.integer)):
            edit_ids = pd.Index([edits])
        elif isinstance(edit_ids, (list, np.ndarray)):
            edit_ids = pd.Index(edit_ids)
        elif isinstance(edit_ids, (pd.Series, pd.DataFrame)):
            edit_ids = edit_ids.index

        # TODO add some logic for keeping track of the edit IDs activated at each step
        # perhaps keep track of "resolved" and "unresolved" versions of the neuron?
        is_used = edit_ids.isin(self.applied_edit_ids)
        if is_used.any():
            raise UserWarning(
                f"Some edit IDs {is_used[is_used].index.to_list()} have already been applied."
            )

        self.applied_edit_ids = self.applied_edit_ids.append(edit_ids)
        self._edit_ids_added[label] = edit_ids.to_list()
        self._applied_edit_history[label] = self.applied_edit_ids.copy().to_list()

        unresolved_neuron = self.base_neuron.set_edits(
            self.applied_edit_ids, inplace=False, prefix=self.prefix
        )

        if self.base_neuron.nucleus_id in unresolved_neuron.nodes.index:
            resolved_neuron = unresolved_neuron.select_nucleus_component(inplace=False)
        else:
            print("WARNING: Using closest point to nucleus to resolve neuron...")
            point_id = find_closest_point(
                unresolved_neuron.nodes,
                self.base_neuron.nodes.loc[
                    self.base_neuron.nucleus_id, ["x", "y", "z"]
                ],
            )
            resolved_neuron = unresolved_neuron.select_component_from_node(
                point_id, inplace=False, directed=False
            )

        resolved_neuron.remove_unused_synapses(inplace=True)

        self.unresolved_sequence[label] = unresolved_neuron
        self.resolved_sequence[label] = resolved_neuron

        self._resolved_synapses[label] = {
            "pre_synapses": resolved_neuron.pre_synapses.index.to_list(),
            "post_synapses": resolved_neuron.post_synapses.index.to_list(),
        }

        return None

    @property
    def edit_ids_added(self):
        edit_ids_added = pd.Series(self._edit_ids_added, name="edit_ids_added")
        edit_ids_added.index.name = self.edit_label_name
        edit_ids_added = edit_ids_added.to_frame()
        return edit_ids_added

    @property
    def applied_edit_history(self):
        applied_edit_history = pd.Series(
            self._applied_edit_history, name="applied_edits"
        )
        applied_edit_history.index.name = self.edit_label_name
        applied_edit_history = applied_edit_history.to_frame()
        return applied_edit_history

    @property
    def resolved_synapses(self):
        resolved_synapses = pd.DataFrame(self._resolved_synapses).T
        resolved_synapses.index.name = self.edit_label_name
        resolved_synapses["n_pre_synapses"] = resolved_synapses["pre_synapses"].apply(
            len
        )
        resolved_synapses["n_post_synapses"] = resolved_synapses["post_synapses"].apply(
            len
        )
        return resolved_synapses

    @property
    def sequence_info(self):
        sequence_info = pd.concat(
            [
                self.edit_ids_added,
                self.applied_edit_history,
                self.resolved_synapses,
            ],
            axis=1,
        )
        return sequence_info
