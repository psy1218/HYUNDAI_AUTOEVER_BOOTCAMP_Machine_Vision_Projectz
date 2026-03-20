import time
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# -------------------------------------------------
# 1. 로그 출력 함수
# -------------------------------------------------
def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_device_info():
    print_section("Device Check")

    print(f"torch version           : {torch.__version__}")
    print(f"CUDA available          : {torch.cuda.is_available()}")
    print(f"CUDA device count       : {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        gpu_id = 0
        print(f"GPU {gpu_id} name           : {torch.cuda.get_device_name(gpu_id)}")

        props = torch.cuda.get_device_properties(gpu_id)
        total_mem_gb = props.total_memory / (1024 ** 3)
        allocated_gb = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)

        print(f"Total GPU memory        : {total_mem_gb:.2f} GB")
        print(f"Allocated memory        : {allocated_gb:.4f} GB")
        print(f"Reserved memory         : {reserved_gb:.4f} GB")
        print(f"Estimated free memory   : {total_mem_gb - reserved_gb:.4f} GB")
    else:
        print("Using CPU only")


def print_gpu_memory(stage=""):
    if torch.cuda.is_available():
        gpu_id = 0
        allocated_gb = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)

        print(f"[GPU Memory] {stage}")
        print(f"  Allocated             : {allocated_gb:.4f} GB")
        print(f"  Reserved              : {reserved_gb:.4f} GB")


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# -------------------------------------------------
# 2. 메인
# -------------------------------------------------
def main():
    start_time = time.time()

    # 네 클래스 개수
    num_classes = 28

    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2-1. 장치 정보 출력
    print_device_info()

    # 2-2. 모델 로드 시작
    print_section("Load EfficientNet-B0")
    load_start = time.time()

    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    load_end = time.time()
    print(f"Pretrained model loaded : EfficientNet-B0")
    print(f"Load time               : {load_end - load_start:.2f} sec")

    # 2-3. 마지막 층 확인 및 교체
    print_section("Replace Final Layer")

    print("Before replacement:")
    print(model.classifier)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    print("\nAfter replacement:")
    print(model.classifier)

    print(f"\nInput features          : {in_features}")
    print(f"Output classes          : {num_classes}")

    # 2-4. 모델을 device로 이동
    print_section("Move Model To Device")
    move_start = time.time()

    model = model.to(device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    move_end = time.time()
    print(f"Using device            : {device}")
    print(f"Move time               : {move_end - move_start:.2f} sec")

    # 2-5. 파라미터 수 확인
    print_section("Model Parameter Info")
    total_params, trainable_params = count_parameters(model)

    print(f"Total parameters        : {total_params:,}")
    print(f"Trainable parameters    : {trainable_params:,}")

    # 2-6. classifier만 학습하고 싶으면 freeze
    print_section("Optional Freeze Example")
    print("지금은 freeze를 실제로 적용하지 않았습니다.")
    print("처음 실험에서 classifier만 학습하려면 아래 코드 사용:")
    print("-" * 60)
    print("for param in model.features.parameters():")
    print("    param.requires_grad = False")
    print("-" * 60)

    # 2-7. 더미 입력으로 forward test
    print_section("Forward Pass Test")
    test_start = time.time()

    dummy_input = torch.randn(8, 3, 384, 256).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    test_end = time.time()

    print(f"Dummy input shape       : {tuple(dummy_input.shape)}")
    print(f"Output shape            : {tuple(output.shape)}")
    print(f"Forward test time       : {test_end - test_start:.2f} sec")

    # 2-8. GPU 메모리 출력
    print_section("Memory Status After Forward Test")
    print_gpu_memory(stage="after dummy forward")

    # 2-9. 전체 소요 시간
    end_time = time.time()
    print_section("Summary")
    print(f"Model setup complete")
    print(f"Final device            : {device}")
    print(f"Final output classes    : {num_classes}")
    print(f"Total elapsed time      : {end_time - start_time:.2f} sec")


if __name__ == "__main__":
    main()