from mindspore import Tensor
import mindspore
import numpy as np

t = Tensor(np.arange(5))
u = Tensor(np.random.uniform(1.,5.,(2.,3)))
coords_h = np.arange(10)
coords_w = np.arange(10)
c1,c2 = np.meshgrid(coords_h, coords_w)
c1 = c1.flatten()
c2 = c2.flatten()
coords_flatten = Tensor(np.stack(np.stack((c2,c1)))) 
from mindspore.ops import operations as P
div = P.Div()
mul = P.Mul()
y = div(t,0.5)

# x = Tensor(np.random.rand(3,3)).astype(y.dtype)
shape = (y.shape[0],) + (1,) * (y.ndim - 1)
print(shape)
print(type(shape[0]))
x = Tensor(np.random.rand(shape[0])).astype(y.dtype)
# floor =P.Floor()
print(x)
print(y)
print(y*x)
# x = Tensor(np.random.rand(1, 1, 4, 1024).astype(np.float32))

