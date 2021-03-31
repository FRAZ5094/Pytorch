import torch

if torch.cuda.is_available():
    device=torch.device("cuda:0")
    print(f"running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device=torch.device("cpu")
    print("running on cpu")
