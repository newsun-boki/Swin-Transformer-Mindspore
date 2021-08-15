import torch

# A = torch.tensor([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],[[13,14,15,16],[17,18,19,20],[21,22,23,24]]])
# B = A.flatten(2)
# print(A)
# print(B)
# for i in range(4):
#     print(i)

# print([i for i in range(10)])
from mindspore.ops import operations as ops
from mindspore import dtype as mstype
shape = (2, 2,2)
ones = ops.Ones()
output = ones(shape, mstype.float32)
print(output)

zeros = ops.Zeros()
output = zeros(shape, mstype.float32)
print(type(output))
# print(torch.zeros((2,2,2)))