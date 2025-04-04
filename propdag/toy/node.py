__docformat__ = "restructuredtext"
__all__ = ["ToyNode"]

from propdag.template import *


class ToyNode(TNode):
    def forward(self):
        print(f"{self.name}: Forward pass")

    def backward(self):
        print(f"{self.name}: Backward pass")

    def clear(self):
        print(f"{self.name}: Clear forward cache")

    def clear_forward_cache(self):
        print(f"{self.name}: Clear forward cache")

    def clear_backward_cache(self):
        print(f"{self.name}: Clear backward cache")
