/*
 * @Author: liboyu 
 * @Date: 2021-08-13 22:12:20 
 * @Last Modified by:   liboyu 
 * @Last Modified time: 2021-08-13 22:12:20 
 */


# from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# img_size=224
# # img_size = to_2tuple(img_size)
# img_size = (img_size, img_size)
# print(img_size)
import numpy as np

from mindspore.common.initializer import initializer
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.communication.management import get_group_size
from mindspore.context import ParallelMode
from mindspore import context
from mindspore.ops import operations as ops

shape = (1,2, 2)
ones = ops.Ones()
output = ones(shape, mstype.float32)
print(output)

zeros = ops.Zeros()
output = zeros((1, 3, 4),mstype.int32)
print(output)