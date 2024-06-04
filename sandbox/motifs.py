# %%
import networkx as nx

# %%


motif = nx.DiGraph()
motif.add_node(0)
motif.add_node(1)
# motif.add_node(2)

# create a list of all possible edges
edges = [(i, j) for i in motif.nodes for j in motif.nodes if i != j]

# create all possible combinations in power set of edges
from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


edge_combinations = list(powerset(edges))

# create a list of all possible subgraphs
subgraphs = []
for edge_combination in edge_combinations:
    subgraph = motif.copy()
    subgraph.add_edges_from(edge_combination)
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

# %%
import numpy as np
from matplotlib import pyplot as plt

scale = 0.75
fig, axs = plt.subplots(
    1,
    len(unique_subgraphs),
    figsize=(len(unique_subgraphs) * scale, scale),
    constrained_layout=True,
)

pos = {0: (-1, 0), 1: (1, 0), 2: (0, np.sqrt(2))}
pad = 0.3
for i, subgraph in enumerate(unique_subgraphs):
    nx.draw(subgraph, ax=axs[i], node_size=50, pos=pos, width=3)
    axs[i].set_axis_off()
    axs[i].autoscale()
    axs[i].set(xlim=(-1 - pad, 1 + pad), ylim=(-pad, np.sqrt(2) + pad))
    # axs[i].axis("equal")

# %%
unique_subgraphs[7]

# %%
G = nx.erdos_renyi_graph(150, 0.2, directed=True)
for i in range(len(unique_subgraphs)):
    print(
        len(
            list(
                nx.isomorphism.DiGraphMatcher(
                    G, unique_subgraphs[i]
                ).subgraph_isomorphisms_iter()
            )
        )
    )

# %%


# create all possible combinations in power set of edges
from itertools import chain, combinations

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
                counts.append(len(find_motifs(motif, graph, count_only=False)))
        return counts

    def count_motif_monomorphisms(self, graph):
        if self.backend == "grandiso" or self.backend == "auto":
            return self._count_motif_monomorphisms_grandiso(graph)
        elif self.backend == "networkx":
            return self._count_motif_monomorphisms_networkx(graph)
        else:
            raise NotImplementedError


# %%
G = nx.erdos_renyi_graph(160, 0.2, directed=True)
# %%
G = nx.erdos_renyi_graph(150, 0.2, directed=True)
motif_counter = MotifCounter(
    orders=[2, 3], backend="networkx", verbose=True, ignore_isolates=True
)
counts = motif_counter.count_motif_isomorphisms(G)


# %%
from grandiso import find_motifs

find_motifs(unique_subgraphs[2], G, count_only=True)

# %%
import networkx as nx
from grandiso import find_motifs

host = nx.fast_gnp_random_graph(10, 0.5, directed=True)

motif = nx.DiGraph()
motif.add_node(0)
motif.add_node(1)
motif.add_node(2)
motif.add_edge(0, 1)
motif.add_edge(1, 0)

grandiso_motifs = find_motifs(motif, host, count_only=False)

nx_motifs = list(
    nx.isomorphism.DiGraphMatcher(host, motif).subgraph_monomorphisms_iter()
)
# 
# print(len(grandiso_motifs))
print(len(nx_motifs))

# %%
import seaborn as sns

sns.heatmap(nx.to_numpy_array(host))
