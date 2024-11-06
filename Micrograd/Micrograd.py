import numpy as np   

class Value:
    """
        Value is a class that represents a value in the computation graph. 
        It is the base class for all the other classes in the computation graph.
    """
    def __init__(self, data, _children = () , _op = '', label = '') -> None:
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f'Value(data = {self.data}, grad = {self.grad}, label = {self.label})'
   
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += output.grad
            other.grad += output.grad
        
        output._backward = _backward
        return output
    
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data - other.data, (self, other), '-')
        
        def _backward():
            self.grad += output.grad
            other.grad -= output.grad
        
        output._backward = _backward
        return output
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        
        output._backward = _backward
        return output
    
    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data ** other.data, (self, other), '**')
        
        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * output.grad
            other.grad += (self.data ** other.data) * np.log(self.data) * output.grad
            
        output._backward = _backward
        return output
    
    def __neg__(self):
        return Value(-self.data, (self,), '-')
    
    def __radd__(self, other):
        return self + other
    
    def __rsub__(self, other):
        return self - other
    
    def __rmul__(self, other):
        return self * other
    
    def __rpow__(self, other):
        return self ** other
    
    def __truediv__(self, other):
        return self * (other ** -1)
    
    def __rtruediv__(self, other):
        return other * (self ** -1)
    
    def backward(self):
        topoList = []
        visited = set()
        def build_topoList(node):
            if v not in visited:
                visited.add(v)
                for node_children in v._prev:
                    build_topoList(node_children)
                topoList.append(v)
                
        build_topoList(self)
        self.grad = 1.0
        for v in reversed(topoList):
            v._backward()