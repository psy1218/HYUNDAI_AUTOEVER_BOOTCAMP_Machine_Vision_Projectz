import os
import glob
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO


# =========================
# 사용자 설정
# =========================
MODEL_PATH = "/home/psy1218/projects/2_pro/YOLOv11/runs/exp_noaug/weights/best.pt"
DATASET_ROOT = "/home/psy1218/projects/2_pro/YOLOv11"
SPLIT = "test"

IMAGE_DIR = os.path.join(DATASET_ROOT, "images", SPLIT)
LABEL_DIR = os.path.join(DATASET_ROOT, "labels", SPLIT)

CLASS_NAMES = {
    0: "deform",
    1: "hole",
    2: "open",
    3: "retain",
    4: "tear",
}
NUM_CLASSES = len(CLASS_NAMES)

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp"]
CONF_THRES = 0.25
IOU_MATCH_THRES = 0.50
IMGSZ = 960
DEVICE = 0  # GPU 사용, CPU면 "cpu"


# =========================
# 유틸 함수
# =========================
def yolo_to_xyxy(line, img_w, img_h):
    parts = line.strip().split()
    if len(parts) != 5:
        return None

    cls_id, xc, yc, w, h = map(float, parts)
    cls_id = int(cls_id)

    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    x2 = (xc + w / 2) * img_w
    y2 = (yc + h / 2) * img_h
    return [cls_id, x1, y1, x2, y2]


def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def load_gt_labels(label_path, img_w, img_h):
    gts = []
    if not os.path.exists(label_path):
        return gts

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parsed = yolo_to_xyxy(line, img_w, img_h)
        if parsed is None:
            continue
        cls_id, x1, y1, x2, y2 = parsed
        gts.append({
            "cls_id": cls_id,
            "bbox": [x1, y1, x2, y2]
        })
    return gts


def calc_prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def line(char="=", n=66):
    return char * n


# =========================
# 메인 평가
# =========================
def main():
    print(line())
    print("테스트 평가 시작")
    print(f"MODEL_PATH : {MODEL_PATH}")
    print(f"IMAGE_DIR  : {IMAGE_DIR}")
    print(f"LABEL_DIR  : {LABEL_DIR}")
    print(line())

    model = YOLO(MODEL_PATH)

    image_paths = []
    for ext in IMG_EXTS:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, f"*{ext}")))
    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        print("테스트 이미지가 없습니다.")
        return

    # 전체 통계
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # 클래스별 통계
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    # mAP50 비슷하게 보일 IoU 평균용
    matched_ious_all = []
    matched_ious_by_class = defaultdict(list)

    # 치명적 오류 집계
    false_alarm_normal_to_defect = 0   # 정상인데 불량으로 예측
    miss_defect_to_normal = 0          # 불량인데 하나도 못 찾음

    total_gt_defects = 0

    for idx, image_path in enumerate(image_paths):
        image_name = os.path.basename(image_path)
        stem = Path(image_path).stem
        label_path = os.path.join(LABEL_DIR, stem + ".txt")

        img = cv2.imread(image_path)
        if img is None:
            print(f"[경고] 이미지 읽기 실패: {image_path}")
            continue

        img_h, img_w = img.shape[:2]
        gts = load_gt_labels(label_path, img_w, img_h)
        total_gt_defects += len(gts)

        results = model.predict(
            source=image_path,
            conf=CONF_THRES,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False
        )

        preds = []
        if len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)

                for b, c, cls_id in zip(xyxy, confs, clss):
                    preds.append({
                        "cls_id": int(cls_id),
                        "conf": float(c),
                        "bbox": b.tolist()
                    })

        # 정상/불량 기반 치명적 오류
        if len(gts) == 0 and len(preds) > 0:
            false_alarm_normal_to_defect += 1

        if len(gts) > 0 and len(preds) == 0:
            miss_defect_to_normal += len(gts)

        # GT-Pred greedy matching
        matched_gt = set()
        matched_pred = set()
        candidates = []

        for gt_i, gt in enumerate(gts):
            for pred_i, pred in enumerate(preds):
                iou = compute_iou(gt["bbox"], pred["bbox"])
                candidates.append((iou, gt_i, pred_i))

        candidates.sort(reverse=True, key=lambda x: x[0])

        for iou, gt_i, pred_i in candidates:
            if iou < IOU_MATCH_THRES:
                continue
            if gt_i in matched_gt or pred_i in matched_pred:
                continue

            gt = gts[gt_i]
            pred = preds[pred_i]

            matched_gt.add(gt_i)
            matched_pred.add(pred_i)

            if gt["cls_id"] == pred["cls_id"]:
                total_tp += 1
                class_tp[gt["cls_id"]] += 1
                matched_ious_all.append(iou)
                matched_ious_by_class[gt["cls_id"]].append(iou)
            else:
                # IoU는 맞지만 클래스가 틀리면
                total_fp += 1
                total_fn += 1
                class_fp[pred["cls_id"]] += 1
                class_fn[gt["cls_id"]] += 1

        # 매칭 실패 GT -> FN
        for gt_i, gt in enumerate(gts):
            if gt_i not in matched_gt:
                total_fn += 1
                class_fn[gt["cls_id"]] += 1

        # 매칭 실패 Pred -> FP
        for pred_i, pred in enumerate(preds):
            if pred_i not in matched_pred:
                total_fp += 1
                class_fp[pred["cls_id"]] += 1

        if (idx + 1) % 20 == 0 or (idx + 1) == len(image_paths):
            print(f"[{idx+1}/{len(image_paths)}] 처리 완료")

    # 전체 지표
    precision, recall, f1 = calc_prf(total_tp, total_fp, total_fn)
    map50_like = np.mean(matched_ious_all) if len(matched_ious_all) > 0 else 0.0

    # =========================
    # 터미널 출력
    # =========================
    print("\n" + line())
    print("🏆 [모델 최종 종합 성적표] 🏆")
    print(line())
    print(f"▶정밀도 (Precision) : {precision:.4f} (불량이라고 찍은 것 중 진짜 불량 비율)")
    print(f"▶재현율 (Recall)    : {recall:.4f} (실제 불량 중 모델이 찾아낸 비율)")
    print(f"▶F1-Score           : {f1:.4f} (정밀도와 재현율의 밸런스 점수)")
    print(f"▶mAP@50             : {map50_like:.4f} (IoU 0.50 기준 평균 매칭 품질)")
    print("-" * 66)
    print("🚨 [혼동 행렬 기반 치명적 오류 요약]")
    print(f"▶오탐지 (정상 -> 불량 착각) : {false_alarm_normal_to_defect}건")
    print(f"▶미탐지 (불량 -> 정상 놓침) : {miss_defect_to_normal}건 / (총 불량 {total_gt_defects}개 중)")
    print(line())

    print("\n📦 [불량 종류별 상세 지표]")
    print(f"{'클래스명':<10} | {'정밀도(P)':<10} | {'재현율(R)':<10} | {'F1-Score':<10} | {'mAP@50':<10}")
    print("-" * 66)

    for cls_id in range(NUM_CLASSES):
        cls_name = CLASS_NAMES[cls_id]
        tp = class_tp[cls_id]
        fp = class_fp[cls_id]
        fn = class_fn[cls_id]

        p, r, f1_c = calc_prf(tp, fp, fn)
        map50_c = np.mean(matched_ious_by_class[cls_id]) if len(matched_ious_by_class[cls_id]) > 0 else 0.0

        print(f"{cls_name:<10} | {p:<10.4f} | {r:<10.4f} | {f1_c:<10.4f} | {map50_c:<10.4f}")

    print(line())
    print("평가 완료")


if __name__ == "__main__":
    main()