"""
Module: tracking_launcher.py
Description: 執行 YOLO (Ultralytics) 內建追蹤並將結果整合為 DataFrame。
"""
import os
import time
import pandas as pd
from pathlib import Path
from ultralytics import YOLO


# ==========================================================
# 輔助函式: 結果格式化
# ==========================================================
def _format_results(r, frame_id):
    """
    將單幀 Ultralytics 結果轉為 DataFrame。

    Args:
        r: Ultralytics Result 物件。
        frame_id (int): 幀編號。

    Returns:
        pd.DataFrame: 格式化後的資料。
    """
    if r.boxes.id is None or len(r.boxes) == 0:
        return pd.DataFrame()
        
    boxes = r.boxes.xyxy.cpu().numpy()
    confs = r.boxes.conf.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy()
    obj_ids = r.boxes.id.cpu().numpy()

    return pd.DataFrame({
        'frame_id': frame_id,
        'obj_id': obj_ids.astype(int),
        'x1': boxes[:, 0], 'y1': boxes[:, 1], 'x2': boxes[:, 2], 'y2': boxes[:, 3],
        'conf': confs, 'cls_id': cls_ids.astype(int),
    })


# ==========================================================
# 執行追蹤
# ==========================================================
def run_tracker(model, tracker_type, source_path, output_dir, imgsz=960, conf=0.4, iou=0.3, stream=False, half=False):
    """
    執行 YOLO 追蹤並整合結果 (In-Memory)。

    Args:
        model: YOLO 模型實例。
        tracker_type (str): 追蹤器名稱 (如 botsort, bytetrack)。
        source_path: 影像來源。
        output_dir: 輸出根目錄。
        stream (bool): 是否使用 Generator 模式。

    Returns:
        tuple: (pd.DataFrame, avg_latency_ms)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
     
    project_dir = output_dir / f"track_{tracker_type}"
    print(f"\n✅ 正在執行 {tracker_type} 追蹤，來源: {source_path}...")

    if model is None:
        raise ValueError("必須傳入 YOLO 模型實例")

    # 1. 開始計時
    t_start = time.perf_counter()
    
    # 2. 初始化 Generator
    results_generator = model.track(
        source=str(source_path),
        tracker=f"../trackers/{tracker_type}.yaml", 
        imgsz=imgsz,
        conf=conf, iou=iou,
        save=False, save_conf=False, save_txt=False,
        project=str(project_dir.parent), name=str(project_dir.name),
        verbose=False, agnostic_nms=True,
        stream=stream, half=half
    )

    print("\n🔄 正在執行追蹤迴圈 (Detect + Track)...")
    df_list = []
    
    # 3. 執行迴圈
    for frame_id, r in enumerate(results_generator, 0):
        df_frame = _format_results(r, frame_id)
        if not df_frame.empty:
            df_list.append(df_frame)
            
        # 顯示進度條
        if frame_id % 10 == 0:
             print(f"   Processing frame {frame_id}...", end='\r')

    # 4. 停止計時
    t_end = time.perf_counter()
            
    if not df_list:
        return pd.DataFrame(), 0.0

    df_raw = pd.concat(df_list, ignore_index=True)
    
    total_frames = df_raw['frame_id'].nunique()
    if total_frames == 0: total_frames = frame_id + 1 # Fallback
    
    # 計算平均延遲
    total_time_ms = (t_end - t_start) * 1000
    latency = total_time_ms / total_frames if total_frames > 0 else 0.0
    
    print(f"\n   ⏱️ 追蹤總運算耗時: {total_time_ms:.2f} ms")
    
    return df_raw, latency