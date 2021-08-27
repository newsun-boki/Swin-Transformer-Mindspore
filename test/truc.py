# from mindspore import Tensor
from mindspore.common.tensor import Tensor
import mindspore
import numpy as np
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from scipy import special
import math
t = Tensor(np.arange(5))
u = Tensor(special.erfinv(np.random.uniform(0.,1.,t.shape)))
mul = P.Mul()
print(u)
tensor = mul(u,2)
add = P.Add()
print(tensor)
min_value = Tensor(5, mindspore.float32)
max_value = Tensor(20, mindspore.float32)
x = Tensor(np.array([[1., 25., 5., 7.], [4., 11., 6., 21.]]), mindspore.float32)
output = C.clip_by_value(x, min_value, max_value)
print(output)