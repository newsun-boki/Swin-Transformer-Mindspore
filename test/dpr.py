import torch
import numpy as np
dpr = [x.item() for x in torch.linspace(0, 0.1, 4)]
x = np.linspace(0,0.1,4)
x = x.tolist();
# dpr = [ for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
print(dpr)
print(type(x))
print(np.linspace(0, 0.1, 4).tolist())