import torch

print("===== Device Check =====")
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_id = 0

    print(f"GPU {gpu_id}:", torch.cuda.get_device_name(gpu_id))

    props = torch.cuda.get_device_properties(gpu_id)
    total_mem_gb = props.total_memory / (1024 ** 3)

    allocated_gb = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
    reserved_gb = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
    free_estimated_gb = total_mem_gb - reserved_gb

    print(f"총 GPU 메모리: {total_mem_gb:.2f} GB")
    print(f"현재 할당된 메모리: {allocated_gb:.2f} GB")
    print(f"현재 예약된 메모리: {reserved_gb:.2f} GB")
    print(f"예약 기준 남은 메모리(대략): {free_estimated_gb:.2f} GB")

    # 텐서 하나 올려보기
    x = torch.randn(3, 3).to(device)
    print("\n테스트 텐서 device:", x.device)

    # 텐서 올린 뒤 다시 확인
    allocated_gb_after = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
    reserved_gb_after = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)

    print("\n===== After Tensor Allocation =====")
    print(f"할당된 메모리: {allocated_gb_after:.4f} GB")
    print(f"예약된 메모리: {reserved_gb_after:.4f} GB")

else:
    print("현재 사용 가능 장치: CPU")

'''
===== Device Check =====
torch version: 2.5.1+cu121
CUDA available: True
CUDA device count: 1
GPU 0: NVIDIA GeForce RTX 4050 Laptop GPU
총 GPU 메모리: 6.00 GB
현재 할당된 메모리: 0.00 GB
현재 예약된 메모리: 0.00 GB
예약 기준 남은 메모리(대략): 6.00 GB

테스트 텐서 device: cuda:0

===== After Tensor Allocation =====
할당된 메모리: 0.0000 GB
예약된 메모리: 0.0020 GB
'''