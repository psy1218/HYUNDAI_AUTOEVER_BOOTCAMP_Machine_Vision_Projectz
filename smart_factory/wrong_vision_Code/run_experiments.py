import subprocess
import shlex

PROJECT_ROOT = "/home/psy1218/projects/2_pro/YOLOv11"
DATA_YAML = f"{PROJECT_ROOT}/data.yaml"
RUNS_DIR = f"{PROJECT_ROOT}/runs"

EXPERIMENTS = [
    {
        "name": "stage1_noaug_n_960_e40",
        "model": "yolo11n.pt",
        "epochs": 40,
        "batch": 8,
        "imgsz": 960,
        "extra_args": {
            "optimizer": "AdamW",
            "patience": 15,
            "cos_lr": "True",
            "save": "True",
            "save_period": 10,
            "plots": "True",
            "verbose": "True",
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "erasing": 0.0,
        },
    },
    {
        "name": "stage1_lightaug_n_960_e40",
        "model": "yolo11n.pt",
        "epochs": 40,
        "batch": 8,
        "imgsz": 960,
        "extra_args": {
            "optimizer": "AdamW",
            "patience": 15,
            "cos_lr": "True",
            "save": "True",
            "save_period": 10,
            "plots": "True",
            "verbose": "True",
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "degrees": 3.0,
            "translate": 0.03,
            "scale": 0.05,
            "shear": 0.0,
            "perspective": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "hsv_h": 0.005,
            "hsv_s": 0.15,
            "hsv_v": 0.10,
            "erasing": 0.0,
        },
    },
    {
        "name": "stage1_lightaug_n_832_e40",
        "model": "yolo11n.pt",
        "epochs": 40,
        "batch": 8,
        "imgsz": 832,
        "extra_args": {
            "optimizer": "AdamW",
            "patience": 15,
            "cos_lr": "True",
            "save": "True",
            "save_period": 10,
            "plots": "True",
            "verbose": "True",
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "degrees": 3.0,
            "translate": 0.03,
            "scale": 0.05,
            "shear": 0.0,
            "perspective": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "hsv_h": 0.005,
            "hsv_s": 0.15,
            "hsv_v": 0.10,
            "erasing": 0.0,
        },
    },
    {
        "name": "stage1_lightaug_s_960_e40",
        "model": "yolo11s.pt",
        "epochs": 40,
        "batch": 4,
        "imgsz": 960,
        "extra_args": {
            "optimizer": "AdamW",
            "patience": 15,
            "cos_lr": "True",
            "save": "True",
            "save_period": 10,
            "plots": "True",
            "verbose": "True",
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "degrees": 3.0,
            "translate": 0.03,
            "scale": 0.05,
            "shear": 0.0,
            "perspective": 0.0,
            "fliplr": 0.0,
            "flipud": 0.0,
            "hsv_h": 0.005,
            "hsv_s": 0.15,
            "hsv_v": 0.10,
            "erasing": 0.0,
        },
    },
]


def build_command(exp: dict) -> str:
    cmd_parts = [
        "yolo detect train",
        f"model={exp['model']}",
        f"data={DATA_YAML}",
        f"epochs={exp['epochs']}",
        f"batch={exp['batch']}",
        f"imgsz={exp['imgsz']}",
        "device=0",
        "workers=8",
        f"project={RUNS_DIR}",
        f"name={exp['name']}",
    ]

    for k, v in exp["extra_args"].items():
        cmd_parts.append(f"{k}={v}")

    return " ".join(cmd_parts)


def run_experiment(exp: dict):
    cmd = build_command(exp)
    print("\n" + "=" * 100)
    print(f"[START] {exp['name']}")
    print(cmd)
    print("=" * 100 + "\n")

    result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        print(f"\n[FAILED] {exp['name']} (returncode={result.returncode})\n")
    else:
        print(f"\n[DONE] {exp['name']}\n")


def main():
    for exp in EXPERIMENTS:
        run_experiment(exp)

    print("\n모든 stage1 실험 종료")
    print("runs 폴더에서 results.csv, results.png, weights/best.pt 확인해서 최고 성능 실험을 고르면 됨.")


if __name__ == "__main__":
    main()