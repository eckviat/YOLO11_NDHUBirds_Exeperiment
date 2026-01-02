"""
Module: yolo_detector.py
Description: 封裝 YOLOv11 模型操作，包含載入、推論、驗證與結果輸出。
"""
import os
import time
from pathlib import Path
from ultralytics import YOLO
from output_formatter import save_results


# ==========================================================
# 模型載入
# ==========================================================
def load_model(weights_path):
    """載入 YOLO 模型。"""
    return YOLO(str(weights_path))


# ==========================================================
# 推論模式
# ==========================================================
def run_detection(model, source_path, output_dir, imgsz=960, conf=0.4, iou=0.3, stream=False, half=False):
    """
    執行 YOLO 推論並儲存標準化結果。

    Returns:
        tuple: (CSV路徑, 平均延遲 ms)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_name = f"{source_path.parent.name}.csv" if source_path.is_dir() else f"{source_path.stem}.csv"
    csv_path = output_dir / csv_name
    if csv_path.exists(): csv_path.unlink()
    
    t_start = time.perf_counter()
    results = model.predict(
        source=str(source_path), imgsz=imgsz, conf=conf, iou=iou,
        save=True, save_conf=False, save_txt=False,
        project=str(output_dir), name="predict",
        verbose=False, agnostic_nms=True, stream=stream, half=half
    )
    if stream: results = list(results)
    t_end = time.perf_counter()
    
    print(f"\n✅ 正在儲存結果: {csv_path}")
    for i, r in enumerate(results):
        save_results(r, i, csv_path, source=source_path, save_mode="auto")

    frames = len(results)
    latency = (t_end - t_start) * 1000 / frames if frames > 0 else 0
    print(f"   ⏱️ 偵測耗時: {(t_end-t_start)*1000:.2f} ms (平均 {latency:.2f} ms/frame)")
    
    return csv_path, latency


# ==========================================================
# 驗證模式
# ==========================================================
def run_validation(model, data_yaml, output_dir, imgsz=960, conf=0.4, iou=0.3):
    """執行 YOLO val()。"""
    return model.val(
        data=str(data_yaml), imgsz=imgsz, conf=conf, iou=iou,
        save_json=True, project=str(output_dir), name="val", verbose=True
    )

def run_predict_on_validation_set(model, source_path, output_dir, imgsz=960, conf=0.4, iou=0.3, half=False):
    """在驗證集上執行推論 (不含計算 mAP，僅產生 CSV)。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_name = f"{source_path.parent.name}_predictions.csv" if source_path.is_dir() else f"{source_path.stem}_predictions.csv"
    csv_path = output_dir / csv_name
    if csv_path.exists(): csv_path.unlink()
        
    t_start = time.perf_counter()
    results = model.predict(
        source=str(source_path), imgsz=imgsz, conf=conf, iou=iou,
        save=False, project=str(output_dir), name="val_predict",
        verbose=False, agnostic_nms=True, stream=False, half=half
    )
    t_end = time.perf_counter()

    print(f"\n✅ 儲存驗證集結果: {csv_path}")
    for i, r in enumerate(results):
        save_results(r, i, csv_path, source=source_path, save_mode="auto")

    frames = len(results)
    latency = (t_end - t_start) * 1000 / frames if frames > 0 else 0
    print(f"   ⏱️ 偵測耗時: {(t_end-t_start)*1000:.2f} ms")
    
    return csv_path, latency