from graphviz import Digraph
from Micrograd.Micrograd import Value

class Graph:
    def __init__(self, root: Value) -> None:
        self.root = root
    
    def trace(self) -> tuple:
        nodes, edges = set(), set()
        def build_trace(node: Value) -> None:
            if node not in nodes:
                nodes.add(node)
                for node_children in node._prev:
                    edges.add((node_children, node))
                    build_trace(node_children)
                    
        build_trace(self.root)
        return nodes, edges
    
    def draw(self, format='svg', rankdir = 'LR') -> Digraph:
        assert rankdir in ['LR', 'TB'], 'rankdir must be one of {LR, TB}'
        nodes, edges = self.trace()
        
        dot = Digraph(format = format, graph_attr = {'rankdir': rankdir})
        
        for node in nodes:
            dot.node(name=str(id(node)), label = "{label | data %.4f | grad %.4f }" % (node.label, node.data, node.grad), shape='record')
            if node._op:
                dot.node(name=str(id(node)), label = node._op)
                dot.edge(str(id(node)) + node._op, str(id(node)))
        
        for node_child, node_parent in edges:
            dot.edge(str(id(node_child)), str(id(node_parent)) + node_parent._op)
        
        return dot
       
        