# from mindspore import Tensor
from mindspore.common.tensor import Tensor
from mindspore import dtype as mstype
import mindspore
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from scipy import special
import math
coords_h = np.arange(5)
coords_w = np.arange(5)
c1,c2 = np.meshgrid(coords_h, coords_w)
c1 = c1.flatten()
c2 = c2.flatten()
coords_flatten = np.stack(np.stack((c2,c1)))   # 2, Wh*Ww
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
# print(relative_coords)
# transpose = P.Transpose()
# relative_coords = transpose(relative_coords,(1,2,0))
relative_coords = relative_coords.transpose(1,2,0)
# print(relative_coords)
# print(relative_coords)
# # relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
relative_coords[:, :, 0] += 5 - 1  # shift to start from 0
relative_coords[:, :, 1] += 5 - 1
relative_coords[:, :, 0] *= 2 * 5 - 1
# print(relative_coords)
relative_position_index = Tensor(relative_coords.sum(-1))
print(relative_position_index)
# n = Tensor(np.array([[1,0],[2,5]]),mstype.float32)
# print(n.sum(-1))






