from model import CBOWModel
import numpy as np
import torch

sample = [[1,4,5,2],\
        [2,3,1,2]]

model = CBOWModel(10, 10, 10)

sample = np.asarray(sample)

x = torch.from_numpy((sample)).type(torch.LongTensor)

print(x)
print(x.shape)

out = model(x)

print(out)
print("out shape ", out.shape)
