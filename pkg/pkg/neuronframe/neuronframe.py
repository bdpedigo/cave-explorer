from typing import Literal, Optional, Union

import caveclient as cc
import pandas as pd
from networkframe import NetworkFrame


class NeuronFrame(NetworkFrame):
    def __init__(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        pre_synapses: Optional[pd.DataFrame] = None,
        post_synapses: Optional[pd.DataFrame] = None,
        edits: Optional[pd.DataFrame] = None,
        nucleus_id: Optional[int] = None,
        neuron_id: Optional[int] = None,
        pre_synapse_mapping_col: str = "pre_pt_level2_id",
        post_synapse_mapping_col: str = "post_pt_level2_id",
        **kwargs,
    ):
        super().__init__(nodes, edges, **kwargs)

        if pre_synapses is None:
            pre_synapses = pd.DataFrame()
        if post_synapses is None:
            post_synapses = pd.DataFrame()
        if edits is None:
            edits = pd.DataFrame()

        self.pre_synapses = pre_synapses
        self.post_synapses = post_synapses
        self.edits = edits
        self.nucleus_id = nucleus_id
        self.neuron_id = neuron_id
        self.pre_synapse_mapping_col = pre_synapse_mapping_col
        self.post_synapse_mapping_col = post_synapse_mapping_col

    def __repr__(self) -> str:
        out = (
            "NeuronFrame(\n"
            + f"    neuron_id={self.neuron_id},\n"
            + f"    nodes={self.nodes.shape},\n"
            + f"    edges={self.edges.shape},\n"
            + f"    pre_synapses={self.pre_synapses.shape},\n"
            + f"    post_synapses={self.post_synapses.shape},\n"
            + f"    edits={self.edits.shape},\n"
            + f"    nucleus_id={self.nucleus_id}\n"
            + ")"
        )
        return out

    @property
    def nucleus_id(self) -> int:
        return self._nucleus_id

    @nucleus_id.setter
    def nucleus_id(self, nucleus_id):
        if nucleus_id not in self.nodes.index:
            raise ValueError(f"nucleus_id {nucleus_id} not in nodes table index")
        self._nucleus_id = nucleus_id

    @property
    def has_pre_synapses(self) -> bool:
        return not self.pre_synapses.empty

    @property
    def has_post_synapses(self) -> bool:
        return not self.post_synapses.empty

    @property
    def has_edits(self) -> bool:
        return not self.edits.empty

    @property
    def pre_synapse_mapping_col(self) -> str:
        return self._pre_synapse_mapping_col

    @pre_synapse_mapping_col.setter
    def pre_synapse_mapping_col(self, pre_synapse_mapping_col: str):
        # check if the column exists
        if self.has_pre_synapses and (
            pre_synapse_mapping_col not in self.pre_synapses.columns
        ):
            msg = (
                f"pre_synapse_mapping_col '{pre_synapse_mapping_col}' not in "
                "pre_synapses table columns"
            )
            raise ValueError(msg)

        # check if all elements in the column are in the nodes index
        if (
            self.has_pre_synapses
            and not self.pre_synapses[pre_synapse_mapping_col]
            .isin(self.nodes.index)
            .all()
        ):
            msg = (
                f"pre_synapse_mapping_col '{pre_synapse_mapping_col}' contains "
                "values not in nodes index"
            )
            raise ValueError(msg)

        self._pre_synapse_mapping_col = pre_synapse_mapping_col

    @property
    def post_synapse_mapping_col(self) -> str:
        return self._post_synapse_mapping_col

    @post_synapse_mapping_col.setter
    def post_synapse_mapping_col(self, post_synapse_mapping_col: str):
        # check if the column exists
        if self.has_post_synapses and (
            post_synapse_mapping_col not in self.post_synapses.columns
        ):
            msg = (
                f"post_synapse_mapping_col '{post_synapse_mapping_col}' not in "
                "post_synapses table columns"
            )
            raise ValueError(msg)

        # check if all elements in the column are in the nodes index
        if (
            self.has_post_synapses
            and not self.post_synapses[post_synapse_mapping_col]
            .isin(self.nodes.index)
            .all()
        ):
            msg = (
                f"post_synapse_mapping_col '{post_synapse_mapping_col}' contains "
                "values not in nodes index"
            )
            raise ValueError(msg)

        self._post_synapse_mapping_col = post_synapse_mapping_col

    def set_edits(self, edit_ids: Union[list[int], int], inplace=False):
        if isinstance(edit_ids, int):
            edit_ids = [edit_ids]

        # by convention -1 represents original things
        query = (
            "(operation_added.isin(@edit_ids) | operation_added == -1)"
            " & ~operation_removed.isin(@edit_ids)"
        )
        if inplace:
            self.query_nodes(query, local_dict=locals(), inplace=inplace)
            self.query_edges(query, local_dict=locals(), inplace=inplace)
        else:
            return self.query_nodes(
                query, local_dict=locals(), inplace=inplace
            ).query_edges(query, local_dict=locals(), inplace=inplace)

    def remove_unused_synapses(
        self, which: Literal["both", "pre", "post"] = "both", inplace=False
    ) -> None:
        pre_synapses = self.pre_synapses
        post_synapses = self.post_synapses
        if which == "pre" or which == "both":
            pre_synapses = self.pre_synapses.query(
                f"{self.pre_synapse_mapping_col} in @self.nodes.index"
            )
        if which == "post" or which == "both":
            post_synapses = self.post_synapses.query(
                f"{self.post_synapse_mapping_col} in @self.nodes.index"
            )
        return self._return(
            pre_synapses=pre_synapses, post_synapses=post_synapses, inplace=inplace
        )

    def select_nucleus_component(self, inplace=False):
        return self.select_component_from_node(
            self.nucleus_id, directed=False, inplace=inplace
        )

    def generate_neuroglancer_link(self, client: cc.CAVEclient):
        pass

    def remove_unused_edits(self):
        pass
