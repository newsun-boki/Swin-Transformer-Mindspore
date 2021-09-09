# from mindspore import Tensor
from mindspore.common.tensor import Tensor
import mindspore 
import mindspore.nn as nn
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import composite as C
# from numpy.core.fromnumeric import transpose
from numpy.lib.shape_base import expand_dims
from scipy import special

class MaskedFill(nn.Cell):
    r"""implement of torch.Tensor.masked_fill by mindspore
    Args:
        value: Value to replace the input Tensor
    """
    def __init__(self):
        super().__init__()
        self.minusend = Tensor([1],mindspore.float32)
        self.mul = P.Mul()
        self.sub = P.Sub()
    def construct(self, inputs:Tensor, mask:Tensor,value:float):
        masked = self.sub(self.minusend,mask)
        adder =value * mask
        inputs= self.mul(inputs,masked)
        outputs =inputs + adder
        return outputs

x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
reshape = P.Reshape()
expand_dims = P.ExpandDims()
# print(reshape(x,(-1,)))
# print(expand_dims(reshape(x,(-1,)),0))
# print(expand_dims(reshape(x,(-1,)),1))
transpose = P.Transpose()
y = Tensor(np.array([[1,2,3,4]]), mindspore.float32)
y_ = transpose(y,(1,0))
mm = nn.MatMul()
value = Tensor([2],mindspore.float32)
mul = P.Mul()
masked_fill = MaskedFill()
yy = masked_fill(y_,y_!=4.,88)
# print(yy)

# print(mm(y,y_))
# zeros = P.Zeros()
# img_mask = zeros((1, 5, 5, 1),mindspore.float32)
# print(img_mask)