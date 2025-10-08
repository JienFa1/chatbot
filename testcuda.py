import torch 
print("Torch CUDA available:", torch.cuda.is_available(), "CUDA:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
