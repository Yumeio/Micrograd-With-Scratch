import numpy as np   

class Value:
    """
        Value is a class that represents a value in the computation graph. 
        It is the base class for all the other classes in the computation graph.
    """
    
    # Constructor
    def __init__(self, data, _children = () , _op = '', label = '') -> None:
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self._backward = lambda: None

    # String representation of the class
    def __repr__(self) -> str:
        return f'Value(data = {self.data}, grad = {self.grad}, label = {self.label})'
   
    # Overloading the addition(+) operator
    def __add__(self, other) -> 'Value':
        other = other if isinstance(other, Value) else Value
        out = Value(data = self.data + other.data, _children = (self, other), _op = '+')
        
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other) -> 'Value':
        return self + other
    
    # Overloading the subtraction(-) operator
    def __sub__(self, other) -> 'Value':
        other = other if isinstance(other, Value) else Value
        out = Value(data = self.data - other.data, _children = (self, other), _op = '-')
        
        def _backward():
            self.grad += 1 * out.grad
            other.grad -= 1 * out.grad
        
        out._backward = _backward
        return out
    
    def __rsub__(self, other) -> 'Value':
        return self - other
    
    # Overloading the multiplication(*) operator
    def __mul__(self, other) -> 'Value':
        other = other if isinstance(other, Value) else Value
        out = Value(data = self.data * other.data, _children = (self, other), _op = '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __rmul__(self, other) -> 'Value':
        return self * other
    
    # Overloading the power(**) operator
    def __pow__(self, other) -> 'Value':
        other = other if isinstance(other, Value) else Value
        out = Value(data = self.data ** other.data, _children = (self, other), _op = '**')
        
        def _backward():
            self.grad += other.data * (self.data ** (other.data - 1)) * out.grad
            other.grad += (self.data ** other.data) * np.log(self.data) * out.grad
    
        out._backward = _backward
        return out
    
    def __rpow__(self, other) -> 'Value':
        return self ** other
    
    # Overloading the division(/) operator
    def __truediv__(self, other) -> 'Value':
        other = other if isinstance(other, Value) else Value
        out = Value(data = self.data / other.data, _children = (self, other), _op = '/')
        
        def _backward():
            self.grad += (1 / other.data) * out.grad
            other.grad -= (self.data / (other.data ** 2)) * out.grad
        
        out._backward = _backward
        return out

    def __rtruediv__(self, other) -> 'Value':
        return self / other
    
    # Overloading the negation operator    
    def __neg__(self) -> 'Value':
        out = Value(data = -self.data, _children = (self), _op = 'neg')
        
        def _backward():
            self.grad += -1 * out.grad
        
        out._backward = _backward
        return out
    
    # Overloading the absolute value operator
    def __abs__(self) -> 'Value':
        out = Value(data = abs(self.data), _children = (self), _op = 'abs')
        
        def _backward():
            self.grad += np.sign(self.data) * out.grad
        
        out._backward = _backward
        return out
    
    # Overloading the matrix multiplication(@) operator
    def __matmul__(self, other) -> 'Value':
        other = other if isinstance(other, Value) else Value
        out = Value(data = self.data @ other.data, _children = (self, other), _op = '@')
        
        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
            
        out._backward = _backward
        return out

    def __rmatmul__(self, other) -> 'Value':
        return self @ other
    
    # Sigmoid function
    def sigmoid(self) -> 'Value':
        t = np.exp(-self.data)
        out = Value(data = 1/(1 + t), _children = (self), _op = 'sigmoid')
        
        def _backward():
            self.grad += (1 - out.data) * out.data * out.grad
        
        out._backward = _backward
        return out
            
    # Trigonometric functions
    def cos(self) -> 'Value':
        out = Value(data = np.cos(self.data), _children = (self), _op = 'cos')
        
        def _backward():
            self.grad += -np.sin(self.data) * out.grad
        
        out._backward = _backward
        return out
    
    def sin(self) -> 'Value':
        out = Value(data = np.sin(self.data), _children = (self), _op = 'sin')

        def _backward():
            self.grad += np.cos(self.data) * out.grad
            
        out._backward = _backward
        return out
    
    def tan(self) -> 'Value':
        out = Value(data = np.tan(self.data), _children = (self), _op = 'tan')
        
        def _backward():
            self.grad += (1/np.cos(self.data) ** 2) * out.grad
            
        out._backward = _backward
        return out
    
    def cotan(self) -> 'Value':
        out = Value(data = 1/np.tan(self.data), _children = (self), _op = 'cotan')
        
        def _backward():
            self.grad += -(1/np.sin(self.data) ** 2) * out.grad 
        
        out._backward = _backward
        
    
    # Hyperbolic functions
    def cosh(self) -> 'Value':
        t = np.exp(self.data) + np.exp(-self.data)
        out = Value(data = t/2, _children = (self), _op = 'cosh')
        
        def _backward():
            self.grad += np.sinh(self.data) * out.grad
        
        out._backward = _backward
        return out
    
    def sinh(self) -> 'Value':
        t = np.exp(self.data) - np.exp(-self.data)
        out = Value(data = t/2, _children = (self), _op = 'sinh')
        
        def _backward():
            self.grad += np.cosh(self.data) * out.grad
            
        out._backward = _backward
        return out
    
    def tanh(self) -> 'Value':
        cosh = (np.exp(self.data) + np.exp(-self.data)) / 2
        sinh = (np.exp(self.data) - np.exp(-self.data)) / 2
        out = Value(data = sinh/cosh, _children = (self), _op = 'tanh')
        
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        
        out._backward = _backward
        return out
    
    def coth(self) -> 'Value':
        cosh = (np.exp(self.data) + np.exp(-self.data)) / 2
        sinh = (np.exp(self.data) - np.exp(-self.data)) / 2
        out = Value(data = cosh/sinh, _children = (self), _op = 'coth')

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        
        out._backward = _backward
        return out
    
    def sech(self) -> 'Value':
        cosh = (np.exp(self.data) + np.exp(-self.data)) / 2
        out = Value(data = 1/cosh, _children = (self), _op = 'sech')
        
        def _backward():
            self.grad += -np.tanh(self.data) * out.grad
            
        out._backward = _backward
        return out
    
    def csch(self) -> 'Value':
        sinh = (np.exp(self.data) - np.exp(-self.data)) / 2
        out = Value(data = 1/sinh, _children = (self), _op = 'csch')

        def _backward():
            self.grad += -np.coth(self.data) * out.grad
        
        out._backward = _backward
        return out

    # Inverse trigonometric functions
    def acos(self) -> 'Value':
        out = Value(data = np.arccos(self.data), _children = (self), _op = 'acos')
        
        def _backward():
            self.grad += -(1/np.sqrt(1 - self.data ** 2)) * out.grad
        
        out._backward = _backward
        return out
    
    def asin(self) -> 'Value':
        out = Value(data = np.arcsin(self.data), _children = (self), _op = 'asin')
        
        def _backward():
            self.grad += (1/np.sqrt(1 - self.data ** 2)) * out.grad
        
        out._backward = _backward
        return out
    
    def atan(self) -> 'Value':
        out = Value(data = np.arctan(self.data), _children = (self), _op = 'atan')
        
        def _backward():
            self.grad += (1/(1 + self.data ** 2)) * out.grad
        
        out._backward = _backward
        return out
    
    def acotan(self) -> 'Value':
        out = Value(data = np.arctan(1/self.data), _children = (self), _op = 'acotan')
        
        def _backward():
            self.grad += -(1/(1 + self.data ** 2)) * out.grad
        
        out._backward = _backward
        return out
    
    def asec(self) -> 'Value':
        out = Value(data = np.arccos(1/self.data), _children = (self), _op = 'asec')
        
        def _backward():
            self.grad += -(1/(np.abs(self.data) * np.sqrt(self.data ** 2 - 1))) * out.grad
        
        out._backward = _backward
        return out
    
    def acsc(self) -> 'Value':
        out = Value(data = np.arcsin(1/self.data), _children = (self), _op = 'acsc')
        
        def _backward():
            self.grad += (1/(np.abs(self.data) * np.sqrt(self.data ** 2 - 1))) * out.grad
        
        out._backward = _backward
        return out
    
    # Inverse hyperbolic functions
    def acosh(self) -> 'Value':
        out = Value(data = np.arccosh(self.data), _children = (self), _op = 'acosh')
        
        def _backward():
            self.grad += (1/np.sqrt(self.data ** 2 - 1)) * out.grad
        
        out._backward = _backward
        return out
    
    def asinh(self) -> 'Value':
        out = Value(data = np.arcsinh(self.data), _children = (self), _op = 'asinh')
        
        def _backward():
            self.grad += (1/np.sqrt(self.data ** 2 + 1)) * out.grad
        
        out._backward = _backward
        return out
    
    def atanh(self) -> 'Value':
        out = Value(data = np.arctanh(self.data), _children = (self), _op = 'atanh')
        
        def _backward():
            self.grad += (1/(1 - self.data ** 2)) * out.grad
        
        out._backward = _backward
        return out
    
    def acoth(self) -> 'Value':
        out = Value(data = np.arctanh(1/self.data), _children = (self), _op = 'acoth')
        
        def _backward():
            self.grad += -(1/(1 - self.data ** 2)) * out.grad
        
        out._backward = _backward
        return out
        
    # Exponential and logarithmic functions
    def exp(self) -> 'Value':
        out = Value(data = np.exp(self.data), _children = (self), _op = 'exp')
        
        def _backward():
            self.grad += np.exp(self.data) * out.grad
        
        out._backward = _backward
        return out
    
    def log(self) -> 'Value':
        out = Value(data = np.log(self.data), _children = (self), _op = 'log')

        def _backward():
            self.grad += (1/self.data) * out.grad
        
        out._backward = _backward
        return out
    
    def log_2(self) -> 'Value':
        out = Value(data = np.log2(self.data), _children = (self), _op = 'log_2')
        
        def _backward():
            self.grad += (1/(self.data * np.log(2))) * out.grad
        
        out._backward = _backward
        return out
    
    def log_10(self, other) -> 'Value':
        out = Value(data = np.log10(self.data), _children = (self), _op = 'log_10')

        def _backward():
            self.grad += (1/(self.data * np.log(10))) * out.grad
        
        out._backward = _backward
        return out
    
    def log_n(self, n: int) -> 'Value':
        out = Value(data = np.log(self.data) / np.log(n), _children = (self), _op = f'log_{n}' ) 
        
        def _backward():
            self.grad += (1/(self.data * np.log(n))) * out.grad
        
        out._backward = _backward
        return out
    
    # SQRT and CBRT functions
    def sqrt(self, other) -> 'Value':
        out = Value(data = np.sqrt(self.data), _children = (self), _op = 'sqrt')
        
        def _backward():
            self.grad += (1/(2 * np.sqrt(self.data))) * out.grad
            
        out._backward = _backward
        return out
    
    def cbrt(self, other) -> 'Value':
        out = Value(data = np.cbrt(self.data), _children = (self), _op = 'cbrt')
        
        def _backward():
            self.grad += (1/(3 * self.data ** (2/3))) * out.grad
        
        out._backward = _backward
        return out
    
    # radius and degrees functions
    def rad(self) -> 'Value':
        out = Value(data = np.radians(self.data), _children = (self), _op = 'rad')
        
        def _backward():
            self.grad += np.radians(1) * out.grad
            
        out._backward = _backward
        return out
    
    def deg(self) -> 'Value':
        out = Value(data = np.degrees(self.data), _children = (self), _op = 'deg')
    
        def _backward():
            self.grad += np.degrees(1) * out.grad
        
        out._backward = _backward
        return out
    
    # ReLU function to calculate the rectified linear unit
    def relu(self, other):
        out = Value(data = np.maximum(0, self.data), _children = (self), _op = 'relu')
        
        def _backward():
            self.grad += (self.data > 0) * out.grad
        
        out._backward = _backward
        return out
    
    # Backward function to calculate the gradients of the computation graph
    def backward(self): 
        visited = set() # Set to keep track of visited nodes
        topo = [] # List to keep track of the topological order of the nodes
        
        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        
        build(self)
        
        # Set the gradient of the self node to 1
        self.grad = 1
        
        for v in reversed(topo):
            v._backward()
    
    # Forward function to calculate the forward pass of the computation graph
    def forward(self):
        visited = set()
        topo = []
        
        def build(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
                
        build(self)