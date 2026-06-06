import torch


print("CUDA is available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("My GPU:", torch.cuda.get_device_name(0))
