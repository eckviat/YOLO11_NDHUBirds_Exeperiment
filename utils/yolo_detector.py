"""
Module: yolo_detector.py
Function: Run YOLO detection and save both original YOLO outputs and standardized CSV.
"""
from ultralytics import YOLO
from pathlib import Path
import os
from output_formatter import save_results

def load_model(weights_path):
    """
    Load YOLOv11 model from weights.
    
    Args:
        weights_path (str or Path): Path to the YOLO model file (.pt)
    Returns:
        model: YOLO model instance
    """
    model = YOLO(str(weights_path))
    return model

def run_detection(model, source_path, output_dir, imgsz=960, conf=0.3, iou=0.3):
    """
    Run YOLO detection on an image folder or video file and save results.

    Args:
        model: Loaded YOLO model.
        source_path (str or Path): Image folder or video file.
        output_dir (str or Path): Directory to save detection results.
        imgsz (int): Image size for inference.
        conf (float): Confidence threshold.
        iou (float): IoU threshold.
    Returns:
        Results object (from Ultralytics)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # -----------------------------
    # 判斷 CSV 檔名
    # -----------------------------
    if source_path.is_dir():
        csv_name = f"{source_path.parent.name}.csv"  # e.g. IMG_0004/images -> IMG_0004.csv
    else:
        csv_name = f"{source_path.stem}.csv"         # e.g. video.mp4 -> video.csv

    csv_path = output_dir / csv_name
    
    # ============================================
    # 1. YOLO 原始輸出 (txt, 圖片)
    # ============================================
    results = model.predict(
        source=str(source_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        save=True,
        save_conf=True,
        save_txt=True,
        project=str(output_dir),
        name="predict",
        verbose=False
    )

    # ============================================
    # 2. 標準化 CSV 輸出
    # ============================================
    for frame_id, r in enumerate(results, 1):
        save_results(r, frame_id, csv_path, source=source_path, save_mode="auto")
    
    return csv_path