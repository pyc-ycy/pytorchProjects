import torch

if __name__ == "__main__":
    t = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])
    print(torch.is_tensor(t))
    print(t)
