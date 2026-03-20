import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyze_confusion_matrix(
    cm,
    class_names,
    save_dir="./cm_analysis",
    cm_csv_name="confusion_matrix_table.csv",
    cm_xlsx_name="confusion_matrix_table.xlsx",
    cm_png_name="confusion_matrix_heatmap.png",
    cm_norm_png_name="confusion_matrix_normalized_heatmap.png",
    metrics_csv_name="classification_metrics.csv",
    metrics_xlsx_name="classification_metrics.xlsx",
    summary_csv_name="summary_metrics.csv"
):
    """
    cm          : 2D list or numpy array (confusion matrix)
    class_names : class label list
    save_dir    : folder to save outputs
    """

    os.makedirs(save_dir, exist_ok=True)

    cm = np.array(cm, dtype=np.int64)

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"혼동행렬은 정사각 2차원 배열이어야 합니다. 현재 shape={cm.shape}")

    n = cm.shape[0]
    if len(class_names) != n:
        raise ValueError(f"class_names 길이({len(class_names)})와 혼동행렬 크기({n})가 다릅니다.")

    # ---------------------------------
    # 1) confusion matrix 표 만들기
    # ---------------------------------
    row_sum = cm.sum(axis=1)      # support
    col_sum = cm.sum(axis=0)
    diag = np.diag(cm)
    total = cm.sum()

    df_cm = pd.DataFrame(
        cm,
        index=[f"True_{c}" for c in class_names],
        columns=[f"Pred_{c}" for c in class_names]
    )

    df_cm["Row_Total"] = row_sum
    df_cm["Correct"] = diag
    df_cm["Row_Recall"] = np.divide(
        diag, row_sum, out=np.zeros_like(diag, dtype=float), where=row_sum != 0
    )

    bottom_row = list(col_sum) + [total, diag.sum(), (diag.sum() / total if total > 0 else 0.0)]
    df_cm.loc["Column_Total"] = bottom_row

    cm_csv_path = os.path.join(save_dir, cm_csv_name)
    df_cm.to_csv(cm_csv_path, encoding="utf-8-sig")

    cm_xlsx_path = os.path.join(save_dir, cm_xlsx_name)
    with pd.ExcelWriter(cm_xlsx_path, engine="openpyxl") as writer:
        df_cm.to_excel(writer, sheet_name="ConfusionMatrix")
        ws = writer.book["ConfusionMatrix"]

        for col in ws.columns:
            max_len = 0
            col_letter = col[0].column_letter
            for cell in col:
                val = "" if cell.value is None else str(cell.value)
                max_len = max(max_len, len(val))
            ws.column_dimensions[col_letter].width = min(max_len + 2, 22)

        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.0000"

    # ---------------------------------
    # 2) precision / recall / f1 계산
    # ---------------------------------
    precision = np.divide(
        diag, col_sum, out=np.zeros_like(diag, dtype=float), where=col_sum != 0
    )
    recall = np.divide(
        diag, row_sum, out=np.zeros_like(diag, dtype=float), where=row_sum != 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(diag, dtype=float),
        where=(precision + recall) != 0
    )
    support = row_sum

    df_metrics = pd.DataFrame({
        "class_name": class_names,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support,
        "predicted_total": col_sum,
        "correct": diag
    })

    accuracy = diag.sum() / total if total > 0 else 0.0

    macro_precision = precision.mean() if n > 0 else 0.0
    macro_recall = recall.mean() if n > 0 else 0.0
    macro_f1 = f1.mean() if n > 0 else 0.0

    weighted_precision = np.average(precision, weights=support) if support.sum() > 0 else 0.0
    weighted_recall = np.average(recall, weights=support) if support.sum() > 0 else 0.0
    weighted_f1 = np.average(f1, weights=support) if support.sum() > 0 else 0.0

    df_summary = pd.DataFrame([
        ["accuracy", accuracy],
        ["macro_precision", macro_precision],
        ["macro_recall", macro_recall],
        ["macro_f1", macro_f1],
        ["weighted_precision", weighted_precision],
        ["weighted_recall", weighted_recall],
        ["weighted_f1", weighted_f1],
        ["total_samples", int(total)]
    ], columns=["metric", "value"])

    metrics_csv_path = os.path.join(save_dir, metrics_csv_name)
    df_metrics.to_csv(metrics_csv_path, index=False, encoding="utf-8-sig")

    metrics_xlsx_path = os.path.join(save_dir, metrics_xlsx_name)
    with pd.ExcelWriter(metrics_xlsx_path, engine="openpyxl") as writer:
        df_metrics.to_excel(writer, sheet_name="PerClassMetrics", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)

        ws1 = writer.book["PerClassMetrics"]
        ws2 = writer.book["Summary"]

        for ws in [ws1, ws2]:
            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col:
                    val = "" if cell.value is None else str(cell.value)
                    max_len = max(max_len, len(val))
                ws.column_dimensions[col_letter].width = min(max_len + 2, 20)

            for row in ws.iter_rows():
                for cell in row:
                    if isinstance(cell.value, float):
                        cell.number_format = "0.0000"

    summary_csv_path = os.path.join(save_dir, summary_csv_name)
    df_summary.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")

    # ---------------------------------
    # 3) heatmap 그리기
    # ---------------------------------
    def plot_cm(matrix, labels, title, save_path, normalize=False):
        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            plot_data = np.divide(
                matrix, row_sums,
                out=np.zeros_like(matrix, dtype=float),
                where=row_sums != 0
            )
        else:
            plot_data = matrix

        plt.figure(figsize=(max(12, len(labels) * 0.6), max(10, len(labels) * 0.5)))
        plt.imshow(plot_data, interpolation="nearest", aspect="auto", cmap="Blues")
        plt.title(title)
        plt.colorbar()

        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels, rotation=90)
        plt.yticks(ticks, labels)

        threshold = plot_data.max() / 2 if plot_data.size > 0 else 0
        for i in range(plot_data.shape[0]):
            for j in range(plot_data.shape[1]):
                text_val = f"{plot_data[i, j]:.2f}" if normalize else f"{int(plot_data[i, j])}"
                plt.text(
                    j, i, text_val,
                    ha="center", va="center",
                    color="white" if plot_data[i, j] > threshold else "black",
                    fontsize=7
                )

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()

    cm_png_path = os.path.join(save_dir, cm_png_name)
    cm_norm_png_path = os.path.join(save_dir, cm_norm_png_name)

    plot_cm(cm, class_names, "Confusion Matrix", cm_png_path, normalize=False)
    plot_cm(cm, class_names, "Normalized Confusion Matrix", cm_norm_png_path, normalize=True)

    # ---------------------------------
    # 4) 콘솔 출력
    # ---------------------------------
    print("\n===== SUMMARY =====")
    print(f"Accuracy            : {accuracy:.4f}")
    print(f"Macro Precision     : {macro_precision:.4f}")
    print(f"Macro Recall        : {macro_recall:.4f}")
    print(f"Macro F1-score      : {macro_f1:.4f}")
    print(f"Weighted Precision  : {weighted_precision:.4f}")
    print(f"Weighted Recall     : {weighted_recall:.4f}")
    print(f"Weighted F1-score   : {weighted_f1:.4f}")
    print(f"Total Samples       : {total}")

    print("\n[저장 완료]")
    print(f"CM CSV        : {cm_csv_path}")
    print(f"CM XLSX       : {cm_xlsx_path}")
    print(f"CM PNG        : {cm_png_path}")
    print(f"CM NORM PNG   : {cm_norm_png_path}")
    print(f"METRICS CSV   : {metrics_csv_path}")
    print(f"METRICS XLSX  : {metrics_xlsx_path}")
    print(f"SUMMARY CSV   : {summary_csv_path}")

    return df_cm, df_metrics, df_summary


# =========================================================
# 사용 예시
# =========================================================
if __name__ == "__main__":
    class_names = [
        "1_E", "1_N", "1_S", "1_W",
        "2_E", "2_N", "2_S", "2_W",
        "3_E", "3_N", "3_S", "3_W",
        "4_E", "4_N", "4_S", "4_W",
        "5_E", "5_N", "5_S", "5_W",
        "6_E", "6_N", "6_S", "6_W",
        "7_E", "7_N", "7_S", "7_W"
    ]

    cm = [
        [6, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 9, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 12, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 28, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 6, 0, 0, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 22, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 0, 1, 0, 0, 0, 0, 0, 0, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 3, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 11, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 13],
    ]

    df_cm, df_metrics, df_summary = analyze_confusion_matrix(cm, class_names)

    print("\n===== PER-CLASS METRICS =====")
    print(df_metrics)