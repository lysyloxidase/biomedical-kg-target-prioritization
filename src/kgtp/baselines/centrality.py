"""Network centrality baseline in the disease neighborhood."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import networkx as nx

from kgtp.eval.metrics import Triple


class CentralityBaseline:
    """Score target genes by PageRank plus degree centrality."""

    def __init__(self) -> None:
        self.pagerank: dict[str, float] = {}
        self.degree: dict[str, float] = {}

    def fit(self, triples: Sequence[Triple]) -> CentralityBaseline:
        """Fit centrality values on an undirected KG projection."""

        graph = nx.Graph()
        graph.add_edges_from((head, tail) for head, _, tail in triples)
        if graph.number_of_nodes() == 0:
            self.pagerank = {}
            self.degree = {}
            return self
        self.pagerank = _pagerank(graph)
        self.degree = {str(node): float(value) for node, value in graph.degree()}
        return self

    def score(self, triple: Triple) -> float:
        """Return target centrality score."""

        _, _, tail = triple
        return float(self.pagerank.get(tail, 0.0) + self.degree.get(tail, 0.0))


def _pagerank(
    graph: nx.Graph, *, damping: float = 0.85, iterations: int = 50
) -> dict[str, float]:
    nodes = sorted(str(node) for node in graph.nodes)
    if not nodes:
        return {}
    neighbors: Mapping[str, set[str]] = {
        node: {str(neighbor) for neighbor in graph.neighbors(node)} for node in nodes
    }
    scores = dict.fromkeys(nodes, 1.0 / len(nodes))
    for _ in range(iterations):
        updated: defaultdict[str, float] = defaultdict(float)
        base = (1.0 - damping) / len(nodes)
        for node in nodes:
            updated[node] += base
        for node in nodes:
            node_neighbors = neighbors[node]
            if not node_neighbors:
                share = damping * scores[node] / len(nodes)
                for target in nodes:
                    updated[target] += share
            else:
                share = damping * scores[node] / len(node_neighbors)
                for target in node_neighbors:
                    updated[target] += share
        scores = dict(updated)
    return {node: float(score) for node, score in scores.items()}
