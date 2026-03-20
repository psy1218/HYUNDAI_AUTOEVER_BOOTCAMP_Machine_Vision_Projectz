import os
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO


# =========================
# 기본 설정
# =========================
ROOT = "/home/psy1218/projects/2_pro/YOLOv11"
RUNS_DIR = os.path.join(ROOT, "runs")
DATA_YAML = os.path.join(ROOT, "data.yaml")
TEST_IMAGES_DIR = os.path.join(ROOT, "images", "test")
TEST_LABELS_DIR = os.path.join(ROOT, "labels", "test")
OUTPUT_ROOT = os.path.join(ROOT, "test_eval_all_runs")

IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.25
SAVE_PRED_IMAGES = True
SAVE_WRONG_IMAGES = True

CLASS_NAMES = {
    0: "deform",
    1: "hole",
    2: "open",
    3: "retain",
    4: "tear",
}
NUM_CLASSES = len(CLASS_NAMES)


# =========================
# 유틸
# =========================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def xywhn_to_xyxy(box, img_w, img_h):
    x_c, y_c, w, h = box
    x1 = (x_c - w / 2.0) * img_w
    y1 = (y_c - h / 2.0) * img_h
    x2 = (x_c + w / 2.0) * img_w
    y2 = (y_c + h / 2.0) * img_h
    return [x1, y1, x2, y2]


def clip_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(img_w - 1, x1))
    y1 = max(0, min(img_h - 1, y1))
    x2 = max(0, min(img_w - 1, x2))
    y2 = max(0, min(img_h - 1, y2))
    return [x1, y1, x2, y2]


def box_iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def read_gt_labels(label_path, img_w, img_h):
    gts = []
    if not os.path.exists(label_path):
        return gts

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        box_xywhn = list(map(float, parts[1:5]))
        box_xyxy = xywhn_to_xyxy(box_xywhn, img_w, img_h)
        box_xyxy = clip_box(box_xyxy, img_w, img_h)
        gts.append({
            "cls_id": cls_id,
            "cls_name": CLASS_NAMES.get(cls_id, str(cls_id)),
            "box_xyxy": box_xyxy
        })
    return gts


def draw_boxes(image, gt_boxes, pred_boxes, title_text=None):
    vis = image.copy()

    for gt in gt_boxes:
        x1, y1, x2, y2 = map(int, gt["box_xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"GT:{gt['cls_name']}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    for pred in pred_boxes:
        x1, y1, x2, y2 = map(int, pred["box_xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            vis,
            f"PR:{pred['cls_name']} {pred['conf']:.2f}",
            (x1, min(vis.shape[0] - 10, y2 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )

    if title_text:
        cv2.putText(
            vis,
            title_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )
    return vis


def greedy_match(gt_list, pred_list, iou_threshold=0.5):
    """
    class 상관없이 IoU 최대 기준 매칭 후,
    class_match / iou_match / final_match 따로 기록
    """
    matches = []
    used_gt = set()
    used_pred = set()

    candidates = []
    for gi, gt in enumerate(gt_list):
        for pi, pred in enumerate(pred_list):
            iou = box_iou_xyxy(gt["box_xyxy"], pred["box_xyxy"])
            candidates.append((iou, gi, pi))

    candidates.sort(reverse=True, key=lambda x: x[0])

    for iou, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        if iou <= 0:
            continue
        used_gt.add(gi)
        used_pred.add(pi)

        gt = gt_list[gi]
        pred = pred_list[pi]

        class_match = int(gt["cls_id"] == pred["cls_id"])
        iou_match = int(iou >= iou_threshold)
        final_match = int(class_match == 1 and iou_match == 1)

        matches.append({
            "gt_idx": gi,
            "pred_idx": pi,
            "gt_class": gt["cls_name"],
            "pred_class": pred["cls_name"],
            "confidence": pred["conf"],
            "iou": iou,
            "class_match": class_match,
            "iou_match": iou_match,
            "final_match": final_match,
            "match_status": "matched" if final_match == 1 else "matched_but_wrong"
        })

    missed_gt = [i for i in range(len(gt_list)) if i not in used_gt]
    extra_pred = [i for i in range(len(pred_list)) if i not in used_pred]

    return matches, missed_gt, extra_pred


def compute_detection_metrics(prediction_df):
    tp = int((prediction_df["final_match"] == 1).sum())
    fp = int((prediction_df["match_status"].isin(["extra_pred", "matched_but_wrong"])).sum())
    fn = int((prediction_df["match_status"].isin(["missed_gt", "matched_but_wrong"])).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    matched_iou = prediction_df.loc[prediction_df["final_match"] == 1, "iou"]
    mean_iou = matched_iou.mean() if len(matched_iou) > 0 else 0.0

    return precision, recall, f1, mean_iou, tp, fp, fn


def build_confusion_matrix_rows(prediction_df):
    labels = list(CLASS_NAMES.values())
    labels_with_bg = labels + ["background"]
    idx_map = {name: i for i, name in enumerate(labels_with_bg)}

    cm = np.zeros((len(labels_with_bg), len(labels_with_bg)), dtype=int)

    for _, row in prediction_df.iterrows():
        status = row["match_status"]

        if status in ["matched", "matched_but_wrong"]:
            gt = row["gt_class"]
            pred = row["pred_class"]
            cm[idx_map[gt], idx_map[pred]] += 1
        elif status == "missed_gt":
            gt = row["gt_class"]
            cm[idx_map[gt], idx_map["background"]] += 1
        elif status == "extra_pred":
            pred = row["pred_class"]
            cm[idx_map["background"], idx_map[pred]] += 1

    cm_df = pd.DataFrame(cm, index=labels_with_bg, columns=labels_with_bg)
    return cm_df


def save_confusion_matrix_png(cm_df, out_png, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_df.values, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm_df.columns))
    plt.xticks(tick_marks, cm_df.columns, rotation=45, ha="right")
    plt.yticks(tick_marks, cm_df.index)

    thresh = cm_df.values.max() / 2.0 if cm_df.values.max() > 0 else 0.5
    for i in range(cm_df.shape[0]):
        for j in range(cm_df.shape[1]):
            plt.text(
                j, i, format(cm_df.values[i, j], "d"),
                ha="center", va="center",
                color="white" if cm_df.values[i, j] > thresh else "black"
            )

    plt.ylabel("GT")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def get_best_weights_from_runs(runs_dir):
    runs = []
    for run_name in sorted(os.listdir(runs_dir)):
        run_path = os.path.join(runs_dir, run_name)
        if not os.path.isdir(run_path):
            continue
        best_pt = os.path.join(run_path, "weights", "best.pt")
        if os.path.exists(best_pt):
            runs.append({
                "run_name": run_name,
                "run_path": run_path,
                "best_pt": best_pt
            })
    return runs


def evaluate_single_run(run_info):
    run_name = run_info["run_name"]
    best_pt = run_info["best_pt"]

    print(f"\n[INFO] Evaluating run: {run_name}")
    print(f"[INFO] best.pt: {best_pt}")

    run_out_dir = os.path.join(OUTPUT_ROOT, run_name)
    pred_img_dir = os.path.join(run_out_dir, "pred_images")
    wrong_img_dir = os.path.join(run_out_dir, "wrong_case_images")

    ensure_dir(run_out_dir)
    ensure_dir(pred_img_dir)
    ensure_dir(wrong_img_dir)

    model = YOLO(best_pt)

    # YOLO 기본 test metric
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=960,
        batch=8,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=0,
        plots=False,
        save_json=False,
        verbose=False
    )

    yolo_precision = float(metrics.box.mp) if hasattr(metrics.box, "mp") else 0.0
    yolo_recall = float(metrics.box.mr) if hasattr(metrics.box, "mr") else 0.0
    yolo_map50 = float(metrics.box.map50) if hasattr(metrics.box, "map50") else 0.0
    yolo_map50_95 = float(metrics.box.map) if hasattr(metrics.box, "map") else 0.0

    classwise_rows = []
    ap50_list = metrics.box.ap50 if hasattr(metrics.box, "ap50") else None
    ap_list = metrics.box.ap if hasattr(metrics.box, "ap") else None
    p_list = metrics.box.p if hasattr(metrics.box, "p") else None
    r_list = metrics.box.r if hasattr(metrics.box, "r") else None

    for cid in range(NUM_CLASSES):
        classwise_rows.append({
            "run_name": run_name,
            "class_id": cid,
            "class_name": CLASS_NAMES[cid],
            "precision": float(p_list[cid]) if p_list is not None else np.nan,
            "recall": float(r_list[cid]) if r_list is not None else np.nan,
            "map50": float(ap50_list[cid]) if ap50_list is not None else np.nan,
            "map50_95": float(ap_list[cid]) if ap_list is not None else np.nan,
        })

    image_paths = sorted([
        os.path.join(TEST_IMAGES_DIR, x)
        for x in os.listdir(TEST_IMAGES_DIR)
        if x.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    ])

    prediction_rows = []
    image_summary_rows = []

    for img_path in image_paths:
        image_name = os.path.basename(img_path)
        stem = Path(image_name).stem
        label_path = os.path.join(TEST_LABELS_DIR, f"{stem}.txt")

        image = cv2.imread(img_path)
        if image is None:
            continue

        img_h, img_w = image.shape[:2]
        gt_list = read_gt_labels(label_path, img_w, img_h)

        result = model.predict(
            source=img_path,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=0,
            verbose=False
        )[0]

        pred_list = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()

            for box, cls_id, conf in zip(boxes_xyxy, classes, confs):
                pred_list.append({
                    "cls_id": int(cls_id),
                    "cls_name": CLASS_NAMES.get(int(cls_id), str(cls_id)),
                    "box_xyxy": box.tolist(),
                    "conf": float(conf)
                })

        matches, missed_gt, extra_pred = greedy_match(
            gt_list, pred_list, iou_threshold=IOU_THRESHOLD
        )

        matched_gt_idx = set()
        matched_pred_idx = set()

        for m in matches:
            matched_gt_idx.add(m["gt_idx"])
            matched_pred_idx.add(m["pred_idx"])

            prediction_rows.append({
                "run_name": run_name,
                "split": "test",
                "image_name": image_name,
                "gt_class": m["gt_class"],
                "pred_class": m["pred_class"],
                "confidence": m["confidence"],
                "iou": m["iou"],
                "class_match": m["class_match"],
                "iou_match": m["iou_match"],
                "final_match": m["final_match"],
                "match_status": m["match_status"],
            })

        for gi in missed_gt:
            gt = gt_list[gi]
            prediction_rows.append({
                "run_name": run_name,
                "split": "test",
                "image_name": image_name,
                "gt_class": gt["cls_name"],
                "pred_class": "",
                "confidence": np.nan,
                "iou": 0.0,
                "class_match": 0,
                "iou_match": 0,
                "final_match": 0,
                "match_status": "missed_gt",
            })

        for pi in extra_pred:
            pred = pred_list[pi]
            prediction_rows.append({
                "run_name": run_name,
                "split": "test",
                "image_name": image_name,
                "gt_class": "",
                "pred_class": pred["cls_name"],
                "confidence": pred["conf"],
                "iou": 0.0,
                "class_match": 0,
                "iou_match": 0,
                "final_match": 0,
                "match_status": "extra_pred",
            })

        wrong_count = sum(1 for x in prediction_rows if x["image_name"] == image_name and x["match_status"] != "matched")
        image_correct = int(wrong_count == 0)

        image_summary_rows.append({
            "run_name": run_name,
            "image_name": image_name,
            "num_gt": len(gt_list),
            "num_pred": len(pred_list),
            "num_correct_match": sum(1 for m in matches if m["final_match"] == 1),
            "num_wrong": wrong_count,
            "image_correct": image_correct
        })

        if SAVE_PRED_IMAGES:
            vis = draw_boxes(image, gt_list, pred_list, title_text=run_name)
            cv2.imwrite(os.path.join(pred_img_dir, image_name), vis)

        if SAVE_WRONG_IMAGES and image_correct == 0:
            wrong_vis = draw_boxes(image, gt_list, pred_list, title_text=f"{run_name} WRONG")
            cv2.imwrite(os.path.join(wrong_img_dir, image_name), wrong_vis)

    prediction_df = pd.DataFrame(prediction_rows)
    image_summary_df = pd.DataFrame(image_summary_rows)
    classwise_df = pd.DataFrame(classwise_rows)

    if len(prediction_df) == 0:
        print(f"[WARN] No predictions/GT rows for {run_name}")
        return None

    precision, recall, f1, mean_iou, tp, fp, fn = compute_detection_metrics(prediction_df)
    image_accuracy = image_summary_df["image_correct"].mean() if len(image_summary_df) > 0 else 0.0

    cm_df = build_confusion_matrix_rows(prediction_df)

    summary_df = pd.DataFrame([{
        "run_name": run_name,
        "num_test_images": len(image_paths),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "image_accuracy": image_accuracy,
        "yolo_precision": yolo_precision,
        "yolo_recall": yolo_recall,
        "yolo_map50": yolo_map50,
        "yolo_map50_95": yolo_map50_95
    }])

    wrong_only_df = prediction_df[prediction_df["match_status"] != "matched"].copy()

    # 저장
    summary_df.to_csv(os.path.join(run_out_dir, "summary_metrics.csv"), index=False, encoding="utf-8-sig")
    classwise_df.to_csv(os.path.join(run_out_dir, "classwise_metrics.csv"), index=False, encoding="utf-8-sig")
    prediction_df.to_csv(os.path.join(run_out_dir, "prediction_analysis.csv"), index=False, encoding="utf-8-sig")
    wrong_only_df.to_csv(os.path.join(run_out_dir, "wrong_only.csv"), index=False, encoding="utf-8-sig")
    image_summary_df.to_csv(os.path.join(run_out_dir, "image_summary.csv"), index=False, encoding="utf-8-sig")
    cm_df.to_csv(os.path.join(run_out_dir, "confusion_matrix_test.csv"), encoding="utf-8-sig")
    save_confusion_matrix_png(
        cm_df,
        os.path.join(run_out_dir, "confusion_matrix_test.png"),
        title=f"{run_name} Test Confusion Matrix"
    )

    print(f"[DONE] {run_name}")
    return {
        "summary_df": summary_df,
        "classwise_df": classwise_df,
        "wrong_only_df": wrong_only_df,
        "image_summary_df": image_summary_df
    }


def main():
    ensure_dir(OUTPUT_ROOT)

    runs = get_best_weights_from_runs(RUNS_DIR)
    if len(runs) == 0:
        print("[ERROR] runs 폴더에서 best.pt를 찾지 못했어.")
        return

    print(f"[INFO] Found {len(runs)} runs with best.pt")

    all_summary = []
    all_classwise = []
    all_wrong_only = []
    all_image_summary = []

    for run_info in runs:
        out = evaluate_single_run(run_info)
        if out is None:
            continue
        all_summary.append(out["summary_df"])
        all_classwise.append(out["classwise_df"])
        all_wrong_only.append(out["wrong_only_df"])
        all_image_summary.append(out["image_summary_df"])

    if len(all_summary) > 0:
        pd.concat(all_summary, ignore_index=True).to_csv(
            os.path.join(OUTPUT_ROOT, "all_runs_test_summary.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    if len(all_classwise) > 0:
        pd.concat(all_classwise, ignore_index=True).to_csv(
            os.path.join(OUTPUT_ROOT, "all_runs_classwise_metrics.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    if len(all_wrong_only) > 0:
        pd.concat(all_wrong_only, ignore_index=True).to_csv(
            os.path.join(OUTPUT_ROOT, "all_runs_wrong_only.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    if len(all_image_summary) > 0:
        pd.concat(all_image_summary, ignore_index=True).to_csv(
            os.path.join(OUTPUT_ROOT, "all_runs_image_summary.csv"),
            index=False,
            encoding="utf-8-sig"
        )

    print("\n[ALL DONE]")
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()