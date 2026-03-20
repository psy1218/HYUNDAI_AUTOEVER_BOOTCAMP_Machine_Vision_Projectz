import os
import csv
import time
import subprocess
from datetime import datetime

import pandas as pd

ROOT = "/home/psy1218/projects/2_pro/YOLOv11"
DATA_YAML = f"{ROOT}/data.yaml"
RUNS_DIR = f"{ROOT}/runs"
LOG_DIR = f"{ROOT}/experiment_logs"
PER_RUN_DIR = f"{LOG_DIR}/per_run_csv"
INTEGRATED_CSV = f"{LOG_DIR}/integrated_summary.csv"

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PER_RUN_DIR, exist_ok=True)

CSV_COLUMNS = [
    "phase",
    "Exp 이름",
    "변경점",
    "model",
    "epochs",
    "batch",
    "imgsz",
    "optimizer",
    "best epoch",
    "Precision",
    "Recall",
    "mAP50",
    "mAP50-95",
    "run_dir",
    "best_pt",
    "results_csv",
    "status",
    "start_time",
    "end_time",
    "duration_min",
]

# -----------------------------
# 1차 스크리닝 실험 목록
# -----------------------------
SCREEN_EXPERIMENTS = [
    {
        "name": "screen01",
        "changes": "baseline-like / AdamW lr0=0.001",
        "params": {
            "model": "yolo11n.pt",
            "epochs": 40,
            "batch": 8,
            "imgsz": 960,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0005,
        },
    },
    {
        "name": "screen02",
        "changes": "AdamW lr0=0.0005",
        "params": {
            "model": "yolo11n.pt",
            "epochs": 40,
            "batch": 8,
            "imgsz": 960,
            "optimizer": "AdamW",
            "lr0": 0.0005,
            "lrf": 0.01,
            "weight_decay": 0.0005,
        },
    },
    {
        "name": "screen03",
        "changes": "AdamW lr0=0.0003",
        "params": {
            "model": "yolo11n.pt",
            "epochs": 40,
            "batch": 8,
            "imgsz": 960,
            "optimizer": "AdamW",
            "lr0": 0.0003,
            "lrf": 0.01,
            "weight_decay": 0.0005,
        },
    },
    {
        "name": "screen04",
        "changes": "SGD lr0=0.01",
        "params": {
            "model": "yolo11n.pt",
            "epochs": 40,
            "batch": 8,
            "imgsz": 960,
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "weight_decay": 0.0005,
        },
    },
    {
        "name": "screen05",
        "changes": "imgsz 640 / batch 16",
        "params": {
            "model": "yolo11n.pt",
            "epochs": 40,
            "batch": 16,
            "imgsz": 640,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0005,
        },
    },
    {
        "name": "screen06",
        "changes": "imgsz 800 / batch 12",
        "params": {
            "model": "yolo11n.pt",
            "epochs": 40,
            "batch": 12,
            "imgsz": 800,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0005,
        },
    },
    {
        "name": "screen07",
        "changes": "imgsz 960 / batch 8",
        "params": {
            "model": "yolo11n.pt",
            "epochs": 40,
            "batch": 8,
            "imgsz": 960,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0005,
        },
    },
    {
        "name": "screen08",
        "changes": "imgsz 1280 / batch 4",
        "params": {
            "model": "yolo11n.pt",
            "epochs": 40,
            "batch": 4,
            "imgsz": 1280,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0005,
        },
    },
    {
        "name": "screen09",
        "changes": "weight_decay 0.001",
        "params": {
            "model": "yolo11n.pt",
            "epochs": 40,
            "batch": 8,
            "imgsz": 960,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.001,
        },
    },
    {
        "name": "screen10",
        "changes": "weight_decay 0.0001",
        "params": {
            "model": "yolo11n.pt",
            "epochs": 40,
            "batch": 8,
            "imgsz": 960,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0001,
        },
    },
    {
        "name": "screen11",
        "changes": "rect=True",
        "params": {
            "model": "yolo11n.pt",
            "epochs": 40,
            "batch": 8,
            "imgsz": 960,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "rect": True,
        },
    },
    {
        "name": "screen12",
        "changes": "multi_scale=True",
        "params": {
            "model": "yolo11n.pt",
            "epochs": 40,
            "batch": 8,
            "imgsz": 960,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "multi_scale": True,
        },
    },
    {
        "name": "screen13",
        "changes": "close_mosaic=10 / mosaic=0.2",
        "params": {
            "model": "yolo11n.pt",
            "epochs": 40,
            "batch": 8,
            "imgsz": 960,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0005,
            "close_mosaic": 10,
            "mosaic": 0.2,
            "mixup": 0.0,
            "cutmix": 0.0,
        },
    },
    {
        "name": "screen14",
        "changes": "yolo11s / imgsz 960 / batch 4",
        "params": {
            "model": "yolo11s.pt",
            "epochs": 40,
            "batch": 4,
            "imgsz": 960,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "weight_decay": 0.0005,
        },
    },
]

# 공통 설정
COMMON_PARAMS = {
    "data": DATA_YAML,
    "device": 0,
    "workers": 8,
    "patience": 8,      # screen 단계는 빨리 끊기게
    "cos_lr": True,
    "save": True,
    "save_period": -1,
    "plots": False,
    "verbose": True,
    "project": RUNS_DIR,
    "cache": "disk",
}


def ensure_integrated_csv():
    if not os.path.exists(INTEGRATED_CSV):
        with open(INTEGRATED_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()


def build_command(params):
    cmd = ["yolo", "detect", "train"]
    for k, v in params.items():
        cmd.append(f"{k}={v}")
    return cmd


def parse_results(run_dir):
    results_csv = os.path.join(run_dir, "results.csv")
    best_pt = os.path.join(run_dir, "weights", "best.pt")

    if not os.path.exists(results_csv):
        raise FileNotFoundError(f"results.csv not found: {results_csv}")

    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    def find_col(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    epoch_col = find_col(["epoch"])
    p_col = find_col(["metrics/precision(B)", "metrics/precision"])
    r_col = find_col(["metrics/recall(B)", "metrics/recall"])
    m50_col = find_col(["metrics/mAP50(B)", "metrics/mAP50"])
    m95_col = find_col(["metrics/mAP50-95(B)", "metrics/mAP50-95"])

    if m95_col is None:
        raise ValueError(f"mAP50-95 column not found. columns={df.columns.tolist()}")

    best_idx = df[m95_col].idxmax()
    best_row = df.loc[best_idx]

    best_epoch = int(best_row[epoch_col]) if epoch_col else int(best_idx) + 1

    # epoch 값이 0-based면 +1 보정
    if epoch_col and best_epoch == 0:
        best_epoch = 1
    elif epoch_col and best_epoch < len(df):
        # Ultralytics results.csv는 보통 epoch가 1-based로 저장되지만
        # 버전 차이 방지를 위해 여기서는 값 그대로 두고, 0이면만 보정
        pass

    return {
        "best epoch": best_epoch,
        "Precision": float(best_row[p_col]) if p_col else "",
        "Recall": float(best_row[r_col]) if r_col else "",
        "mAP50": float(best_row[m50_col]) if m50_col else "",
        "mAP50-95": float(best_row[m95_col]) if m95_col else "",
        "results_csv": results_csv,
        "best_pt": best_pt if os.path.exists(best_pt) else "",
    }


def save_per_run_csv(row, exp_name):
    out_csv = os.path.join(PER_RUN_DIR, f"{exp_name}_summary.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def append_integrated_csv(row):
    with open(INTEGRATED_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(row)


def run_experiment(phase, exp_name, changes, params):
    run_dir = os.path.join(RUNS_DIR, exp_name)
    start_dt = datetime.now()
    start_ts = time.time()

    row = {
        "phase": phase,
        "Exp 이름": exp_name,
        "변경점": changes,
        "model": params.get("model", ""),
        "epochs": params.get("epochs", ""),
        "batch": params.get("batch", ""),
        "imgsz": params.get("imgsz", ""),
        "optimizer": params.get("optimizer", ""),
        "best epoch": "",
        "Precision": "",
        "Recall": "",
        "mAP50": "",
        "mAP50-95": "",
        "run_dir": run_dir,
        "best_pt": "",
        "results_csv": "",
        "status": "running",
        "start_time": start_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "end_time": "",
        "duration_min": "",
    }

    try:
        merged = {}
        merged.update(COMMON_PARAMS)
        merged.update(params)
        merged["name"] = exp_name

        cmd = build_command(merged)

        print("\n" + "=" * 120)
        print(f"[{phase}] START: {exp_name}")
        print("CMD:", " ".join(cmd))
        print("=" * 120)

        subprocess.run(cmd, check=True)

        parsed = parse_results(run_dir)

        end_dt = datetime.now()
        duration_min = round((time.time() - start_ts) / 60.0, 2)

        row.update({
            "best epoch": parsed["best epoch"],
            "Precision": parsed["Precision"],
            "Recall": parsed["Recall"],
            "mAP50": parsed["mAP50"],
            "mAP50-95": parsed["mAP50-95"],
            "best_pt": parsed["best_pt"],
            "results_csv": parsed["results_csv"],
            "status": "success",
            "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_min": duration_min,
        })

    except Exception as e:
        end_dt = datetime.now()
        duration_min = round((time.time() - start_ts) / 60.0, 2)
        row.update({
            "status": f"fail: {str(e)}",
            "end_time": end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_min": duration_min,
        })

    save_per_run_csv(row, exp_name)
    append_integrated_csv(row)
    return row


def select_top_k_from_integrated(k=3):
    df = pd.read_csv(INTEGRATED_CSV)
    df.columns = [c.strip() for c in df.columns]

    screen_df = df[(df["phase"] == "screen") & (df["status"] == "success")].copy()
    if len(screen_df) == 0:
        return []

    screen_df["mAP50-95"] = pd.to_numeric(screen_df["mAP50-95"], errors="coerce")
    screen_df["Recall"] = pd.to_numeric(screen_df["Recall"], errors="coerce")

    screen_df = screen_df.sort_values(
        by=["mAP50-95", "Recall"],
        ascending=[False, False]
    )

    return screen_df.head(k).to_dict("records")


def build_refine_experiment(screen_row, idx):
    base_name = screen_row["Exp 이름"]
    refine_name = f"refine{idx}_{base_name}"

    model = screen_row["model"]
    batch = int(float(screen_row["batch"]))
    imgsz = int(float(screen_row["imgsz"]))
    optimizer = screen_row["optimizer"]

    original_changes = screen_row["변경점"]

    params = {
        "model": model,
        "epochs": 80,
        "batch": batch,
        "imgsz": imgsz,
        "optimizer": optimizer,
    }

    # screen 실험 설정에서 핵심 차이를 복구
    # 이름 규칙 기반으로 반영
    if "lr0=0.0005" in original_changes:
        params["lr0"] = 0.0005
        params["lrf"] = 0.01
        params["weight_decay"] = 0.0005
    elif "lr0=0.0003" in original_changes:
        params["lr0"] = 0.0003
        params["lrf"] = 0.01
        params["weight_decay"] = 0.0005
    elif "SGD" in original_changes:
        params["lr0"] = 0.01
        params["lrf"] = 0.01
        params["weight_decay"] = 0.0005
    else:
        params["lr0"] = 0.001
        params["lrf"] = 0.01

    if "weight_decay 0.001" in original_changes:
        params["weight_decay"] = 0.001
    elif "weight_decay 0.0001" in original_changes:
        params["weight_decay"] = 0.0001
    else:
        params.setdefault("weight_decay", 0.0005)

    if "rect=True" in original_changes:
        params["rect"] = True

    if "multi_scale=True" in original_changes:
        params["multi_scale"] = True

    if "close_mosaic=10" in original_changes:
        params["close_mosaic"] = 10
        params["mosaic"] = 0.2
        params["mixup"] = 0.0
        params["cutmix"] = 0.0

    changes = f"REFINE from {base_name} / {original_changes}"
    return refine_name, changes, params


def main():
    ensure_integrated_csv()

    print("\n=== 1차 스크리닝 시작 ===")
    for exp in SCREEN_EXPERIMENTS:
        run_experiment(
            phase="screen",
            exp_name=exp["name"],
            changes=exp["changes"],
            params=exp["params"],
        )

    print("\n=== 스크리닝 종료, 상위 3개 선택 ===")
    top3 = select_top_k_from_integrated(k=3)

    if len(top3) == 0:
        print("성공한 screen 실험이 없어서 refine 단계는 건너뜀.")
        return

    for i, row in enumerate(top3, start=1):
        print(
            f"TOP{i}: {row['Exp 이름']} | mAP50-95={row['mAP50-95']} | "
            f"Recall={row['Recall']} | changes={row['변경점']}"
        )

    print("\n=== 2차 정밀 재학습 시작 ===")
    for i, row in enumerate(top3, start=1):
        refine_name, changes, params = build_refine_experiment(row, i)

        # refine 단계는 조금 더 천천히, 기록도 그림 저장
        global COMMON_PARAMS
        old_common = COMMON_PARAMS.copy()

        COMMON_PARAMS["patience"] = 20
        COMMON_PARAMS["plots"] = True
        COMMON_PARAMS["save_period"] = 10

        run_experiment(
            phase="refine",
            exp_name=refine_name,
            changes=changes,
            params=params,
        )

        COMMON_PARAMS = old_common

    print("\n=== 전체 완료 ===")
    print(f"통합 CSV: {INTEGRATED_CSV}")
    print(f"개별 CSV 폴더: {PER_RUN_DIR}")


if __name__ == "__main__":
    main()