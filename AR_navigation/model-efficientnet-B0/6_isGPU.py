import torch

print("===== Device Check =====")
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(3, 3).to(device)

print("device:", device)
print(x)
print("tensor device:", x.device)