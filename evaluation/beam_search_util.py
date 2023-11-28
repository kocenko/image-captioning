from typing import Union
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx



@dataclass
class CandidateNode:
    id: int
    tokens: list[int]
    probability: float
    best: bool


@dataclass
class CandidateGraph:
    nodes: dict
    edges: defaultdict


def scale_params(sizes: list[float], max_size: Union[int, float] = 1000, min_size: Union[int, float] = 10, which_type: type = int) -> list[int]:
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
    new_nodes = {parent: nodes[parent] for parent in parents} | {child: nodes[child] for parent in parents for child in edges[parent]}
    new_edges = {parent: edges[parent] for parent in parents}
    return new_nodes, new_edges


def unravel_graph(candidate_graph: CandidateGraph) -> tuple[nx.DiGraph, dict]:
    nodes, edges = candidate_graph.nodes, candidate_graph.edges
    new_nodes, new_edges = truncate_graph(nodes, edges)

    graph = nx.DiGraph()
    params = {}

    # Add nodes
    node_sizes = []
    node_alphas = []

    edge_colors = []
    for node in new_nodes.values():
        graph.add_node(node.id)
        node_sizes.append(node.probability)
        node_alphas.append(node.probability)

    for parent_id, children_ids in new_edges.items():
        for child_id in children_ids:
            child = nodes[child_id]
            graph.add_edge(parent_id, child_id)
            edge_colors.append('green' if child.best else 'black')

    params['node_sizes'] = scale_params(node_sizes)
    params['node_alphas'] = scale_params(node_alphas, 1.0, 0.3, float)
    params['edge_colors'] = edge_colors
    params['nodes'] = new_nodes
    params['edges'] = edges

    return graph, params
