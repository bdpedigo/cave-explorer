from itertools import chain, combinations

import networkx as nx
import numpy as np
from grandiso import find_motifs
from tqdm.auto import tqdm


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def _generate_motifs_from_base(motif_base, ignore_isolates=False):
    edges = [(i, j) for i in motif_base.nodes for j in motif_base.nodes if i != j]

    edge_combinations = list(powerset(edges))

    # create a list of all possible subgraphs
    subgraphs = []
    for edge_combination in edge_combinations:
        subgraph = motif_base.copy()
        subgraph.add_edges_from(edge_combination)
        if ignore_isolates and not nx.is_weakly_connected(subgraph):
            continue
        subgraphs.append(subgraph)

    unique_subgraphs = []
    for subgraph1 in subgraphs:
        found_match = False
        for subgraph2 in unique_subgraphs:
            if nx.is_isomorphic(subgraph1, subgraph2):
                found_match = True
                break
        if not found_match:
            unique_subgraphs.append(subgraph1.copy())

    return unique_subgraphs


class MotifCounter:
    def __init__(
        self, orders=[2, 3], backend="auto", ignore_isolates=False, verbose=False
    ):
        self.orders = orders
        self.ignore_isolates = ignore_isolates
        self.backend = backend
        self.verbose = verbose
        self.motifs = self._generate_motifs(orders)

    def _generate_motifs(self, orders):
        motifs = []
        for order in orders:
            motif_base = nx.DiGraph()
            for i in range(order):
                motif_base.add_node(i)

            motifs.extend(
                _generate_motifs_from_base(
                    motif_base, ignore_isolates=self.ignore_isolates
                )
            )
        return motifs

    def _count_motif_isomorphisms_networkx(self, graph):
        counts = []
        for motif in tqdm(self.motifs, disable=not self.verbose):
            counts.append(
                len(
                    list(
                        nx.isomorphism.DiGraphMatcher(
                            graph, motif
                        ).subgraph_isomorphisms_iter()
                    )
                )
            )
        return counts

    def count_motif_isomorphisms(self, graph):
        if self.backend == "networkx" or self.backend == "auto":
            return self._count_motif_isomorphisms_networkx(graph)
        else:
            raise NotImplementedError

    def _count_motif_monomorphisms_networkx(self, graph):
        counts = []
        for motif in tqdm(self.motifs, disable=not self.verbose):
            counts.append(
                len(
                    list(
                        nx.isomorphism.DiGraphMatcher(
                            graph, motif
                        ).subgraph_monomorphisms_iter()
                    )
                )
            )
        return counts

    def _count_motif_monomorphisms_grandiso(self, graph):
        counts = []
        for motif in tqdm(self.motifs, disable=not self.verbose):
            if not nx.is_weakly_connected(motif):
                counts.append(np.nan)
            else:
                counts.append(find_motifs(motif, graph, count_only=True))
        return counts

    def count_motif_monomorphisms(self, graph):
        if self.backend == "grandiso" or self.backend == "auto":
            return self._count_motif_monomorphisms_grandiso(graph)
        elif self.backend == "networkx":
            return self._count_motif_monomorphisms_networkx(graph)
        else:
            raise NotImplementedError
