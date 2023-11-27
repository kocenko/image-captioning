from typing import Callable
from model.transformer import CandidateNode


def unravel_graph(root_node: CandidateNode, hashing_fun: Callable):
    def go_through(node: CandidateNode):
        if node.children:
            return [go_through(child) for child in node.children]
        else:
            return hashing_fun(node.parent.tokens), hashing_fun(node.tokens)

    output = go_through(root_node, output)
    for li in output:
        print(li)
