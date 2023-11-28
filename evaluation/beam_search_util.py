from typing import Union
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
import numpy as np


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


def scale_params(sizes: list[float], max_size: Union[int, float] = 1000, min_size: Union[int, float] = 10, which_type: type = int, use_log: bool = True) -> list[int]:
    if use_log:
        sizes = np.log10(sizes)
    max_elem_size = max(sizes)
    min_elem_size = min(sizes)
    divisor = (max_elem_size - min_elem_size)
    sizes = [
        which_type((size - min_elem_size) * (max_size - min_size) / divisor + min_size)
        for size in sizes
    ]
    return sizes


def truncate_graph(nodes: dict, edges: dict) -> tuple[dict, dict]:
    nodes = nodes.copy()
    edges = edges.copy()

    parents = [parent_id for parent_id, children in edges.items() if len(children) > 1]
    all_children = [child for parent in parents for child in edges[parent] if len(edges[child]) == 1]

    for child in all_children:
        while len(edges[child]):
            edges[child] = edges[edges[child][0]]

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

    return new_nodes, new_edges


def unravel_graph(candidate_graph: CandidateGraph) -> tuple[nx.DiGraph, dict]:
    nodes, edges = candidate_graph.nodes, candidate_graph.edges
    new_nodes, new_edges = truncate_graph(nodes, edges)

    graph = nx.DiGraph()
    params = {}

    alpha_min, alpha_max = 0.3, 1.0
    labels = {}
    node_alphas = []
    node_colors = []
    node_outline_colors = []
    edge_alphas = []
    edge_colors = []

    for node in new_nodes.values():
        graph.add_node(node.id, font_color='w' if node.best else 'k')
        labels[node.id] = '[{}] P={:.2e}'.format(node.id, node.probability)
        node_alphas.append(node.probability)
        node_colors.append('indigo' if node.best else 'black')
        node_outline_colors.append('red' if node.last else 'black')

    for parent_id, children_ids in new_edges.items():
        for child_id in children_ids:
            child = new_nodes[child_id]
            graph.add_edge(parent_id, child_id)
            edge_colors.append('indigo' if child.best else 'black')
            edge_alphas.append(child.probability)

    params['labels'] = labels
    params['node_alphas'] = scale_params(node_alphas, alpha_max, alpha_min, float)
    params['node_colors'] = node_colors
    params['node_outline_colors'] = node_outline_colors
    params['edge_alphas'] = scale_params(edge_alphas, alpha_max, alpha_min, float)
    params['edge_colors'] = edge_colors
    params['nodes'] = new_nodes
    params['edges'] = edges

    return graph, params
