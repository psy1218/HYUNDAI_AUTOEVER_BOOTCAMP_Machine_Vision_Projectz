import os
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def evaluate_random_on_28class_dataset(dataset_root):
    """
    dataset_root 예:
    /mnt/c/Users/한국전파진흥협회/Downloads/test_dataset/test_dataset

    폴더 구조 예:
    test_dataset/
        1_E/
            img1.jpg
            img2.jpg
        1_W/
            img3.jpg
        ...
        7_N/
            img100.jpg
    """

    # 1. 28개 클래스 정의
    directions = ['E', 'W', 'S', 'N']
    class_labels = []

    for zone in range(1, 8):
        for d in directions:
            class_labels.append(f"{zone}_{d}")

    # 폴더 실제 존재하는 클래스만 써도 되지만
    # 혼동행렬을 28개 고정으로 보기 위해 전체 28개를 유지
    y_true = []
    y_pred = []
    records = []

    # 2. 데이터셋 순회
    for class_name in sorted(os.listdir(dataset_root)):
        class_path = os.path.join(dataset_root, class_name)

        if not os.path.isdir(class_path):
            continue

        # 28개 클래스 형식이 아닌 폴더는 건너뜀
        if class_name not in class_labels:
            continue

        for filename in os.listdir(class_path):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            image_path = os.path.join(class_path, filename)

            # 정답: 폴더명
            true_label = class_name

            # 예측: 아직 모델 없으므로 28개 클래스 중 랜덤
            pred_label = random.choice(class_labels)

            y_true.append(true_label)
            y_pred.append(pred_label)

            records.append({
                '파일명': filename,
                '정답 클래스': true_label,
                '예측 클래스': pred_label
            })

    # 3. 데이터가 없을 경우
    if len(y_true) == 0:
        print("평가할 이미지가 없습니다.")
        return None

    # 4. 전체 metric 계산
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average='macro', zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=class_labels, average='weighted', zero_division=0
    )

    # 5. classification report
    report = classification_report(
        y_true,
        y_pred,
        labels=class_labels,
        zero_division=0,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()

    # 6. confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    # 7. 결과 요약 출력
    print("\n" + "=" * 60)
    print("28클래스 랜덤 예측 평가 결과")
    print("=" * 60)
    print(f"총 이미지 수: {len(y_true)}")
    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Precision(macro): {precision_macro:.4f}")
    print(f"Recall(macro)   : {recall_macro:.4f}")
    print(f"F1-score(macro) : {f1_macro:.4f}")
    print(f"Precision(weighted): {precision_weighted:.4f}")
    print(f"Recall(weighted)   : {recall_weighted:.4f}")
    print(f"F1-score(weighted) : {f1_weighted:.4f}")

    # 8. 일부 예측 결과 표
    results_df = pd.DataFrame(records)
    print("\n[예측 결과 예시 상위 10개]")
    print(results_df.head(10))

    print("\n[클래스별 Precision / Recall / F1 일부]")
    print(report_df[['precision', 'recall', 'f1-score', 'support']].head(10))

    # 9. 혼동행렬 시각화
    plt.figure(figsize=(18, 18))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(
        cmap='Blues',
        xticks_rotation=90,
        values_format='d'
    )
    plt.title('28-Class Confusion Matrix (Random Prediction)')
    plt.tight_layout()
    plt.show()

    return {
        'results_df': results_df,
        'report_df': report_df,
        'confusion_matrix': cm,
        'class_labels': class_labels,
        'summary': {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }
    }


# 실행 예시
dataset_root = r"/mnt/c/Users/한국전파진흥협회/Downloads/test_dataset/test_dataset"
result = evaluate_random_on_28class_dataset(dataset_root)