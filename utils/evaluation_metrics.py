"""
Module: evaluation_metrics.py
Description: 提供類別穩定性計算 (IoU/ID based) 與標準 MOT 指標 (MOTA, IDF1) 計算函式。
"""
import re
from pathlib import Path
import pandas as pd
import numpy as np
import motmetrics as mm


# ==========================================================
# 輔助函式: Intersection over Union (IoU)
# ==========================================================
def _bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    計算兩個邊界框 [x1, y1, x2, y2] 的 IoU。
    
    Args:
        box1 (np.ndarray): 第一個邊界框。
        box2 (np.ndarray): 第二個邊界框。
        
    Returns:
        float: IoU 值 (0.0 ~ 1.0)。
    """
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = max(box1[2], box2[2])
    y_bottom = max(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    if union_area == 0:
        return 0.0

    return intersection_area / union_area


# ==========================================================
# 評估函式 1: 類別穩定性 (基於 IoU 匹配)
# ==========================================================
def compute_class_stability(detection_csv_path: str | Path, iou_threshold: float = 0.3) -> dict:
    """
    分析基於 IoU 匹配的連續幀之間的類別變化 (用於偵測器基準線評估)。

    Args:
        detection_csv_path (str | Path): 標準化偵測結果 CSV 路徑。
        iou_threshold (float): IoU 匹配閾值。

    Returns:
        dict: 包含 'class_instability_rate' 等指標。
    """
    df = pd.read_csv(detection_csv_path)
    if df.empty:
        return {"class_instability_rate": 0.0, "total_class_changes": 0, "total_matched_pairs": 0}

    frame_ids = sorted(df["frame_id"].unique())
    total_class_changes = 0
    total_matched_pairs = 0

    for i in range(len(frame_ids) - 1):
        frame_t = df[df["frame_id"] == frame_ids[i]].reset_index(drop=True)
        frame_t1 = df[df["frame_id"] == frame_ids[i + 1]].reset_index(drop=True)

        boxes_t = frame_t[["x1", "y1", "x2", "y2"]].values
        classes_t = frame_t["cls_id"].values
        boxes_t1 = frame_t1[["x1", "y1", "x2", "y2"]].values
        classes_t1 = frame_t1["cls_id"].values

        if len(boxes_t) == 0 or len(boxes_t1) == 0:
            continue

        matched_indices = []

        for idx_t, box_t in enumerate(boxes_t):
            best_iou = -1.0
            best_idx_t1 = -1

            for idx_t1, box_t1 in enumerate(boxes_t1):
                if idx_t1 in matched_indices:
                    continue
                iou = _bbox_iou(box_t, box_t1)
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_idx_t1 = idx_t1

            if best_idx_t1 != -1:
                total_matched_pairs += 1
                matched_indices.append(best_idx_t1)
                if classes_t[idx_t] != classes_t1[best_idx_t1]:
                    total_class_changes += 1

    rate = total_class_changes / total_matched_pairs if total_matched_pairs > 0 else 0.0
    return {
        "class_instability_rate": rate,
        "total_class_changes": total_class_changes,
        "total_matched_pairs": total_matched_pairs,
    }


# ==========================================================
# 評估函式 2: 類別穩定性 (基於追蹤 ID)
# ==========================================================
def compute_class_stability_with_id(tracking_csv_path: str | Path) -> dict:
    """
    分析基於追蹤 ID 的類別穩定性 (用於追蹤器結果評估)。

    Args:
        tracking_csv_path (str | Path): 包含 obj_id 的追蹤結果 CSV 路徑。

    Returns:
        dict: 包含 'class_instability_rate_id' 等指標。
    """
    path = Path(tracking_csv_path)
    if not path.exists():
        return {"class_instability_rate_id": 0.0, "total_class_changes_id": 0, "total_comparisons_id": 0}

    df = pd.read_csv(path)
    if df.empty:
        return {"class_instability_rate_id": 0.0, "total_class_changes_id": 0, "total_comparisons_id": 0}

    df_tracked = df[df["obj_id"] != -1].copy()
    total_changes = 0
    total_detections = 0

    for _, group in df_tracked.groupby("obj_id"):
        group = group.sort_values(by="frame_id")
        total_detections += len(group)
        class_ids = group["cls_id"].values
        changes = np.diff(class_ids)
        total_changes += np.count_nonzero(changes)

    total_comparisons = total_detections - len(df_tracked["obj_id"].unique())
    rate = total_changes / total_comparisons if total_comparisons > 0 else 0.0

    return {
        "class_instability_rate_id": rate,
        "total_class_changes_id": total_changes,
        "total_comparisons_id": total_comparisons,
    }


def compute_class_accuracy(df_pred: pd.DataFrame, df_gt: pd.DataFrame) -> float:
    """
    計算分類準確率 (Accuracy)，假設 ID 匹配正確。

    Args:
        df_pred: 預測 DataFrame (需含 FrameId, Id, ClassId)。
        df_gt: 真值 DataFrame (需含 FrameId, Id, ClassId)。

    Returns:
        float: 準確率。
    """
    required = ["FrameId", "Id", "ClassId"]
    for col in required:
        if col not in df_pred.columns or col not in df_gt.columns:
            return 0.0

    merged = pd.merge(df_pred, df_gt, on=["FrameId", "Id"], suffixes=("_pred", "_gt"), how="inner")
    if merged.empty:
        return 0.0

    correct = (merged["ClassId_pred"] == merged["ClassId_gt"]).sum()
    return correct / len(merged)


# ==========================================================
# 評估函式 3: 標準 MOT 指標 (MOTA, IDF1, etc.)
# ==========================================================
def _load_mot_gt_to_mot_format(gt_path: str | Path) -> pd.DataFrame:
    """讀取 MOTChallenge 格式的 gt.txt 並標準化欄位名稱。"""
    cols = ["frameid", "id", "x", "y", "w", "h", "confidence", "class", "visibility", "other"]
    df = pd.read_csv(gt_path, header=None, names=cols, index_col=False)
    
    # 保留需要的欄位並重新命名
    df = df[["frameid", "id", "x", "y", "w", "h", "class"]].copy()
    df.rename(columns={
        "frameid": "FrameId", "id": "Id", "x": "X", "y": "Y", 
        "w": "Width", "h": "Height", "class": "ClassId"
    }, inplace=True)
    return df

def compute_mot_metrics(prediction: str | Path | pd.DataFrame, gt_path: str | Path) -> dict | None:
    """
    計算標準 MOT 指標 (MOTA, IDF1, IDSW) 及分類準確率。

    Args:
        prediction: 預測結果 (CSV 路徑或 DataFrame)。
        gt_path: GT 檔案路徑。

    Returns:
        dict | None: 包含各項指標的字典，若失敗則回傳 None。
    """
    gt_path = Path(gt_path)

    # 1. 讀取 GT
    try:
        df_gt_full = _load_mot_gt_to_mot_format(gt_path)
        df_gt_mot = df_gt_full[["FrameId", "Id", "X", "Y", "Width", "Height"]].copy()
    except Exception as e:
        print(f"⚠️ 無法讀取 GT: {e}")
        return None

    # 2. 讀取 Prediction
    try:
        if isinstance(prediction, (str, Path)):
            df_pred = pd.read_csv(prediction)
        elif isinstance(prediction, pd.DataFrame):
            df_pred = prediction.copy()
        else:
            print(f"❌ 錯誤: 不支援的預測輸入格式 {type(prediction)}")
            return None
    except Exception as e:
        print(f"⚠️ 無法讀取預測資料: {e}")
        return None

    # 3. 資料前處理
    # 補齊寬高 (若只有 x1, y1, x2, y2)
    if "w" not in df_pred.columns and "x2" in df_pred.columns:
        df_pred["w"] = df_pred["x2"] - df_pred["x1"]
        df_pred["h"] = df_pred["y2"] - df_pred["y1"]
        df_pred["x"], df_pred["y"] = df_pred["x1"], df_pred["y1"]
    
    # 補齊左上角座標
    if "x" not in df_pred.columns and "x1" in df_pred.columns:
        df_pred["x"] = df_pred["x1"]
    if "y" not in df_pred.columns and "y1" in df_pred.columns:
        df_pred["y"] = df_pred["y1"]

    # 欄位重新命名以符合 py-motmetrics 需求
    rename_map = {
        "frame_id": "FrameId", "obj_id": "Id", "cls_id": "ClassId",
        "x": "X", "y": "Y", "w": "Width", "h": "Height", "conf": "Conf"
    }
    df_pred.rename(columns=rename_map, inplace=True)

    # Frame ID 對齊 (0-based -> 1-based)
    if not df_pred.empty and df_pred["FrameId"].min() == 0 and df_gt_mot["FrameId"].min() == 1:
        df_pred["FrameId"] += 1

    # 4. 計算指標
    class_acc = compute_class_accuracy(df_pred, df_gt_full)

    try:
        acc = mm.MOTAccumulator(auto_id=True)
        all_frames = sorted(list(set(df_gt_mot["FrameId"]) | set(df_pred["FrameId"])))

        for fid in all_frames:
            gt = df_gt_mot[df_gt_mot["FrameId"] == fid]
            pred = df_pred[df_pred["FrameId"] == fid]

            dist = mm.distances.iou_matrix(
                gt[["X", "Y", "Width", "Height"]].values,
                pred[["X", "Y", "Width", "Height"]].values,
                max_iou=0.5
            )
            acc.update(gt["Id"].values, pred["Id"].values, dist)

        mh = mm.metrics.create()
        summary = mh.compute(acc, metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses"], name="acc")

        return {
            "MOTA": float(summary["mota"].iloc[0]),
            "IDF1": float(summary["idf1"].iloc[0]),
            "IDSW": int(summary["num_switches"].iloc[0]),
            "FP": int(summary["num_false_positives"].iloc[0]),
            "FN": int(summary["num_misses"].iloc[0]),
            "Cls_Acc": float(class_acc)
        }
    except Exception as e:
        print(f"⚠️ MOTMetrics 計算錯誤: {e}")
        return None