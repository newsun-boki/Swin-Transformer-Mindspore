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
class Roll(nn.Cell):
    r"""simply implement of torch.roll by mindspore,only support dim=0,1,2
    """
    def __init__(self):
        super().__init__()
    def construct(self, x:Tensor, shift_size:int, dim:int):
        if shift_size > 0:
            if dim == 0:
                for shift in range(shift_size):
                    tmp = x[-1]
                    for i in range(x.shape[dim]-1,0,-1):
                        x[i] = x[i-1]
                    x[0] = tmp
            elif dim == 1:
                for shift in range(shift_size):
                    for dim0 in range(x.shape[0]):
                        x_1 = x[dim0]
                        tmp = x_1[-1]
                        for i in range(x_1.shape[0]-1,0,-1):
                            x_1[i] = x_1[i-1]
                        x_1[0] = tmp
                        x[dim0] = x_1
            elif dim == 2:
                for shift in range(shift_size):
                    for dim0 in range(x.shape[0]):
                        x_1 = x[dim0]
                        for dim1 in range(x_1.shape[0]):
                            x_2 = x_1[dim1]
                            tmp = x_2[-1]
                            for i in range(x_2.shape[0]-1,0,-1):
                                x_2[i] = x_2[i-1]
                            x_2[0] = tmp
                            x_1[dim1] = x_2
                        x[dim0] = x_1
        elif shift_size < 0:
            shift_size = -shift_size
            if dim == 0:
                for shift in range(shift_size):
                    tmp = x[0]
                    for i in range(0,x.shape[0]-1):
                        x[i] = x[i + 1]
                    x[-1] = tmp
            elif dim == 1:
                for shift in range(shift_size):
                    for dim0 in range(x.shape[0]):
                        x_1 = x[dim0]
                        tmp = x_1[0]
                        for i in range(0,x_1.shape[0]-1):
                            x_1[i] = x_1[i+1]
                        x_1[-1] = tmp
                        x[dim0] = x_1
            elif dim == 2:
                for shift in range(shift_size):
                    for dim0 in range(x.shape[0]):
                        x_1 = x[dim0]
                        for dim1 in range(x_1.shape[0]):
                            x_2 = x_1[dim1]
                            tmp = x_2[0]
                            for i in range(0,x_2.shape[0]-1):
                                x_2[i] = x_2[i-1]
                            x_2[-1] = tmp
                            x_1[dim1] = x_2
                        x[dim0] = x_1
        return x


x = Tensor(np.array([[1,2,3,4],[5,6,7,8]]),dtype=mindspore.float32)
roll = Roll()
# x = roll(roll(x,-1,1),1,0)
# print(x)
class AdaptiveAvgPool1d_1(nn.Cell):
    def __init__(self):
        super().__init__()
    def construct(self, x):
        mean = P.ReduceMean(keep_dims=True)
        x = mean(x,-1)
        return x
avgpool = AdaptiveAvgPool1d_1()
print(avgpool(x))