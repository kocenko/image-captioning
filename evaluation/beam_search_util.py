from typing import Union
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
import numpy as np

COLOR_HIGHLIGHT = "#27b051"
COLOR_DARKER = "#4e1599"


@dataclass
class CandidateNode:
    id: int
    tokens: list[int]
    probability: float
    best: bool
    last: bool


@dataclass
class CandidateGraph:
    nodes: dict
    edges: defaultdict


def scale_params(
    sizes: list[float],
    max_size: Union[int, float] = 1000,
    min_size: Union[int, float] = 10,
    which_type: type = int,
    use_log: bool = True,
) -> list[int]:
    if use_log:
        sizes = np.log10(sizes)
    max_elem_size = max(sizes)
    min_elem_size = min(sizes)
    divisor = max_elem_size - min_elem_size
    sizes = [which_type((size - min_elem_size) * (max_size - min_size) / divisor + min_size) for size in sizes]
    return sizes


def truncate_graph(nodes: dict, edges: dict) -> tuple[dict, dict, set]:
    nodes = nodes.copy()
    edges = edges.copy()
    truncated = set()

    parents = [parent_id for parent_id, children in edges.items() if len(children) > 1]
    all_children = [child for parent in parents for child in edges[parent] if len(edges[child]) == 1]

    for child in all_children:
        while len(edges[child]):
            truncated.add(child)
            next_in_line = edges[child][0]
            edges[child] = edges[next_in_line]
            nodes[child].tokens = nodes[next_in_line].tokens
            nodes[child].best = nodes[next_in_line].best
            nodes[child].last = nodes[next_in_line].last
            nodes[child].probability = nodes[next_in_line].probability

    parents.extend([child for child in all_children if len(edges[child]) == 1])

    # Resetting numbering
    new_nodes = {}
    new_edges = {}
    all_nodes = set([parent for parent in parents] + [child for parent in parents for child in edges[parent]])
    mapping = {node: i for i, node in enumerate(sorted(list(all_nodes)))}
    for old_id, new_id in mapping.items():
        node = nodes[old_id]
        node.id = new_id
        new_nodes[new_id] = node
        edge = edges[old_id]
        new_edges[new_id] = [mapping[elem] for elem in edge]

    truncated = set([mapping[node] for node in truncated])
    return new_nodes, new_edges, truncated


def unravel_graph(candidate_graph: CandidateGraph) -> tuple[nx.DiGraph, dict]:
    nodes, edges = candidate_graph.nodes, candidate_graph.edges
    new_nodes, new_edges, truncated = truncate_graph(nodes, edges)

    graph = nx.DiGraph()
    params = {}

    alpha_min, alpha_max = 0.3, 1.0
    probability_min = sorted([node.probability for node in new_nodes.values()], reverse=True)[-3]
    labels = {}
    node_alphas = []
    node_colors = []
    node_outline_colors = []
    node_outline_width = []
    edge_alphas = []
    edge_colors = []
    edge_styles = []

    for node in new_nodes.values():
        graph.add_node(node.id, font_color="w" if node.best else "k")
        labels[node.id] = "[{}] P={:.2e}".format(node.id, node.probability)
        node_alphas.append(node.probability if node.id not in truncated else probability_min)
        node_colors.append(COLOR_DARKER if node.best else "black")
        node_outline_colors.append(COLOR_HIGHLIGHT if node.last else "black")
        node_outline_width.append(5 if node.last else 0)

    for parent_id, children_ids in new_edges.items():
        for child_id in children_ids:
            child = new_nodes[child_id]
            graph.add_edge(parent_id, child_id)
            edge_colors.append(COLOR_DARKER if child.best else "black")
            edge_alphas.append(child.probability if child.id not in truncated else probability_min)
            edge_styles.append("--" if child_id in truncated else "-")

    params["labels"] = labels
    params["node_alphas"] = scale_params(node_alphas, alpha_max, alpha_min, float)
    params["node_colors"] = node_colors
    params["node_outline_colors"] = node_outline_colors
    params["node_outline_width"] = node_outline_width
    params["edge_alphas"] = scale_params(edge_alphas, alpha_max, alpha_min, float)
    params["edge_colors"] = edge_colors
    params["edge_styles"] = edge_styles
    params["nodes"] = new_nodes
    params["edges"] = edges

    return graph, params
