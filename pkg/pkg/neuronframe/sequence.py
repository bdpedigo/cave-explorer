from typing import Callable, Literal, Optional, Self, Union

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

SEQUENCE_INFO_COLS = [
    "edit_ids_added",
    "applied_edits",
    "pre_synapses",
    "post_synapses",
    "n_nodes",
    "path_length",
    "order",
]


class NeuronFrameSequence:
    def __init__(
        self,
        base_neuron: NeuronFrame,
        prefix="",
        edit_label_name=None,
        edits=None,
        include_initial_state=True,
        warn_on_missing=True,
    ):
        self.base_neuron = base_neuron
        self.prefix = prefix
        self.applied_edit_ids = pd.Index([])
        self.unresolved_sequence = {}
        self.resolved_sequence = {}
        self._sequence_info = {}
        self.edit_label_name = edit_label_name
        self.tables = {}

        if edits is None:
            if self.prefix == "meta":
                edits = self.base_neuron.metaedits
            else:
                if isinstance(self.base_neuron, str): 
                    print(self.base_neuron)
                edits = self.base_neuron.edits
                edits["has_split"] = ~edits["is_merge"]
                edits["has_merge"] = edits["is_merge"]
                edits["n_operations"] = np.ones(len(edits), dtype=int)
            self.edits = edits.copy()
            if include_initial_state:
                self.apply_edits(
                    self.applied_edit_ids, label=None, warn_on_missing=warn_on_missing
                )
        else:
            self.edits = edits

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
    def current_resolved_neuron(self) -> Self:
        return self.resolved_sequence[self.latest_label]

    def apply_edits(
        self,
        edits: Union[
            list, np.ndarray, pd.Index, pd.Series, int, np.integer, pd.DataFrame
        ],
        label: Optional[Hashable] = None,
        warn_on_reuse: bool = False,
        warn_on_missing: bool = False,
        replace: bool = False,
        only_additions: bool = False,
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
        if is_used.any() and warn_on_reuse:
            print(
                f"WARNING: Some edit IDs {list(edit_ids[is_used])} have already been applied."
            )
        if replace:
            self.applied_edit_ids = edit_ids
        else:
            self.applied_edit_ids = self.applied_edit_ids.append(edit_ids).unique()

        if only_additions:
            unresolved_neuron = self.base_neuron.set_additions(
                self.applied_edit_ids, inplace=False, prefix=self.prefix
            )
        else:
            unresolved_neuron = self.base_neuron.set_edits(
                self.applied_edit_ids, inplace=False, prefix=self.prefix
            )

        self.unresolved_sequence[label] = unresolved_neuron

        resolved_neuron = resolve_neuron(
            unresolved_neuron, self.base_neuron, warn_on_missing=warn_on_missing
        )

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

        sequence_info = sequence_info.join(self.edits, how="left")
        sequence_info["n_operations"].fillna(0, inplace=True)

        sequence_info["cumulative_n_operations"] = sequence_info[
            "n_operations"
        ].cumsum()

        sequence_info["order"] = np.arange(len(sequence_info))

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

    def find_incident_edits(self) -> pd.Index:
        # look at edges that are connected to the current neuron
        current_neuron = self.current_resolved_neuron
        out_edges = self.base_neuron.edges.query(
            "source.isin(@current_neuron.nodes.index) | target.isin(@current_neuron.nodes.index)"
        )
        # ignore those that we already have
        out_edges = out_edges.drop(current_neuron.edges.index)

        possible_edit_ids = out_edges[f"{self.prefix}operation_added"].unique()

        edits = self.edits
        possible_edit_ids = edits.index[edits.index.isin(possible_edit_ids)]

        applied_edit_ids = self.applied_edit_ids
        possible_edit_ids = possible_edit_ids[~possible_edit_ids.isin(applied_edit_ids)]

        return possible_edit_ids

    def apply_to_synapses_by_sample(
        self,
        func: Callable,
        which: Literal["pre", "post"],
        output="series",
        name: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Apply a function which takes in a DataFrame of synapses and returns a DataFrame
        of results.

        Parameters
        ----------
        func
            A function which takes in a DataFrame of synapses and returns a DataFrame
            of results.
        which
            Whether to apply the function to the pre- or post-synaptic synapses.
        kwargs
            Additional keyword arguments to pass to `func`.
        """
        if which == "pre":
            synapses_df = self.base_neuron.pre_synapses
            resolved_synapses = self.sequence_info["pre_synapses"]
        else:
            synapses_df = self.base_neuron.post_synapses
            resolved_synapses = self.sequence_info["post_synapses"]

        results_by_sample = []
        for i, key in enumerate(resolved_synapses.keys()):
            sample_resolved_synapses = resolved_synapses[key]

            input = synapses_df.loc[sample_resolved_synapses]
            result = func(input, **kwargs)
            if output == "dataframe":
                result[self.edit_label_name] = key
            elif output == "series":
                result.name = key
            elif output == "scalar":
                pass

            results_by_sample.append(result)

        if output == "dataframe":
            results_df = pd.concat(results_by_sample, axis=0)
            return results_df
        elif output == "series":
            results_df = pd.concat(results_by_sample, axis=1).T
            results_df.index.name = self.edit_label_name
            return results_df
        else:
            results_df = pd.Series(
                results_by_sample, index=resolved_synapses.keys()
            ).to_frame()
            results_df.index.name = self.edit_label_name
            if name is not None:
                results_df.columns = [name]
            return results_df

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

    def synapse_groupby_metrics(
        self, by: str, which: Literal["pre", "post"], join_sequence_info: bool = True
    ) -> pd.DataFrame:
        counts_table = self.synapse_groupby_count(by=by, which=which)

        edit_label_name = self.edit_label_name

        # melt the counts into long-form
        stats_tidy = counts_table.reset_index().melt(
            var_name=by, value_name="count", id_vars=edit_label_name
        )

        # also compute proportions and do the same melt
        probs = counts_table / counts_table.sum(axis=1).values[:, None]
        probs.fillna(0, inplace=True)
        probs_tidy = probs.reset_index().melt(
            var_name=by, value_name="prop", id_vars=edit_label_name
        )

        # combining tables
        stats_tidy["prop"] = probs_tidy["prop"]

        if join_sequence_info:
            stats_tidy = stats_tidy.join(self.sequence_info, on=edit_label_name)

        return stats_tidy

    # @classmethod
    # def from_sequence(
    #     cls,
    #     base_neuron: NeuronFrame,
    #     sequence_info: pd.DataFrame,
    #     prefix="",
    # ):
    #     edit_label_name = sequence_info.index.name
    #     sequence = cls(
    #         base_neuron,
    #         prefix=prefix,
    #         edit_label_name=edit_label_name,
    #         set_initial_edits=False,
    #     )
    #     for label, row in sequence_info.iterrows():
    #         sequence._sequence_info[label] = row[SEQUENCE_INFO_COLS].to_dict()
    #     return sequence

    def to_dict(self) -> dict:
        out = {
            "prefix": self.prefix,
            "edit_label_name": self.edit_label_name,
            "sequence_info": self.sequence_info[SEQUENCE_INFO_COLS].to_dict(
                orient="index"
            ),
            "edits": self.edits.to_dict(orient="index"),
        }
        return out

    @classmethod
    def from_dict_and_neuron(cls, data: dict, neuron: NeuronFrame) -> Self:
        prefix = data["prefix"]
        edit_label_name = data["edit_label_name"]
        sequence_info = pd.DataFrame(data["sequence_info"]).T
        edits = pd.DataFrame(data["edits"]).T

        out = cls(neuron, prefix=prefix, edit_label_name=edit_label_name, edits=edits)

        for label, row in sequence_info.iterrows():
            out._sequence_info[label] = row.to_dict()

        out.applied_edit_ids = pd.Index(out.sequence_info.iloc[-1]["applied_edits"])

        # TODO set the state of the current neuron to the sequence info's final state?

        return out

    def select(
        self,
        edits: Union[list, np.ndarray, pd.Index, pd.Series, int, np.integer],
    ):
        if isinstance(edits, (int, np.integer)):
            edit_ids = pd.Index([edits])
        elif isinstance(edits, (list, np.ndarray)):
            edit_ids = pd.Index(edits)
        elif isinstance(edits, (pd.Series, pd.DataFrame)):
            edit_ids = edits.index
        else:  # pd.Index
            edit_ids = edits

        for edit_id in edit_ids:
            if edit_id not in self._sequence_info:
                raise ValueError(f"Edit ID {edit_id} not found in sequence.")

        out = self.__class__(
            self.base_neuron,
            prefix=self.prefix,
            edit_label_name=self.edit_label_name,
            edits=self.edits,
        )

        for label, info in self._sequence_info.items():
            if label in edit_ids:
                out._sequence_info[label] = info
        out.applied_edit_ids = pd.Index(out.sequence_info.iloc[-1]["applied_edits"])

        return out

    def select_by_bout(self, by: str, keep: Literal["first", "last"] = "last"):
        """Select a subset of the edit sequence using a bout of activity.

        Parameters
        ----------
        by
            The column to use for grouping the sequence into bouts. This column should
            have boolean values. Nan values will be treated as False. True values will
            denote the start of a new bout of edits.
        keep
            Whether to keep the first or last edit in each bout as the exemplar.
        """
        bouts = self.sequence_info[by].fillna(False).cumsum()
        bouts.name = "bout"
        if keep == "first":
            keep_ind = 0
        else:
            keep_ind = -1
        bout_exemplars = (
            self.sequence_info.index.to_series()
            .groupby(bouts, sort=False)
            .apply(lambda x: x.iloc[keep_ind])
        ).values
        bout_exemplars = pd.Index(bout_exemplars)
        return self.select(bout_exemplars)

    # def reconstruct_neuron_states(self) -> None:
    #     for


def resolve_neuron(unresolved_neuron, base_neuron, warn_on_missing=False):
    if base_neuron.nucleus_id in unresolved_neuron.nodes.index:
        resolved_neuron = unresolved_neuron.select_nucleus_component(inplace=False)
    else:
        if warn_on_missing:
            print("WARNING: Using closest point to nucleus to resolve neuron...")
        point_id = find_closest_point(
            unresolved_neuron.nodes,
            base_neuron.nodes.loc[base_neuron.nucleus_id, ["x", "y", "z"]],
        )
        resolved_neuron = unresolved_neuron.select_component_from_node(
            point_id, inplace=False, directed=False
        )

    resolved_neuron = resolved_neuron.remove_unused_synapses(inplace=False)
    return resolved_neuron
