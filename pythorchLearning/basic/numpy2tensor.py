import torch
import numpy as np

if __name__ == "__main__":
    a = np.array([2, 3.3])
    print(torch.from_numpy(a))
    b = np.ones([2, 3])
    print(torch.from_numpy(b))
    print(torch.tensor([1.2, 3]).type())
    torch.set_default_tensor_type(torch.DoubleTensor)
    print(torch.tensor([1.2, 3]).type())
