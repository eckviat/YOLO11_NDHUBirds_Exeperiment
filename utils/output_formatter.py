"""
Module: output_formatter.py
Function: Standardize and save YOLO detection/tracking results to CSV format.
"""
import pandas as pd
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def save_results(result, frame_id, save_path, source=None, save_mode="auto"):
    """
    Args:
        results (ultralytics.engine.results.Results): YOLO detection/tracking output for one frame.
        frame_id (int): Frame index.
        save_path (str or Path): CSV file path to store results.
        source (str or Path, optional): 資料來源 (影片、資料夾、影像)
        save_mode (str): "auto" | "video" | "folder" | "image" | "none"
            - "auto": 根據 source 自動判斷模式
                * mp4/avi/mov -> "video"
                * 資料夾 -> "folder"
                * 圖片(jpg/png/jpeg) -> "image"
            - "video": 使用影片檔名，逐幀追加
            - "folder": 使用資料夾名稱，逐幀追加
            - "image": 使用影像檔名，每次覆寫
            - "none": 不存檔，只回傳 DataFrame
    
    Output Schema (CSV):
        source_name (str): Name of the image or the video
        frame_id (int): Frame index
        obj_id (int): Tracker-assigned ID (-1 if no tracking)
        cls_id (int): Predicted class index
        conf (float): Confidence score
        x1,y1,x2,y2 (float): Bounding box coordinates
    """
    
    source = Path(source) if source else None

    # -------------------------
    # Auto 判斷 save_mode
    # -------------------------
    if save_mode == "auto":
        if source is None:
            save_mode = "none"
        elif source.is_dir():
            save_mode = "folder"
        else:
            ext = source.suffix.lower()
            if ext in VIDEO_EXTS:
                save_mode = "video"
            elif ext in IMAGE_EXTS:
                save_mode = "image"
            else:
                save_mode = "none"
            
    # -------------------------
    # 決定名稱與寫檔策略
    # -------------------------
    if save_mode == "video" and source:
        name, write_mode = source.stem, "append"
    elif save_mode == "folder" and source:
        name, write_mode = source.name, "append"
    elif save_mode == "image" and source:
        name, write_mode = source.stem, "overwrite"
    elif save_mode == "none":
        name, write_mode = "unknown", None
    else:
        name, write_mode = "unknown", "append"

    # -------------------------
    # 建立 DataFrame
    # -------------------------
    rows = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        obj_id = int(box.id[0]) if hasattr(box, "id") and box.id is not None else -1

        rows.append({
            "source_name": name,
            "frame_id": frame_id,
            "obj_id": obj_id,
            "cls_id": cls_id,
            "conf": conf,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })
    df = pd.DataFrame(rows)
    
    # -------------------------
    # None 模式：直接回傳，不存檔
    # -------------------------
    if save_mode == "none":
        return df

    # -------------------------
    # 存檔
    # -------------------------
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if write_mode == "append" and save_path.exists():
        df.to_csv(save_path, mode="a", index=False, header=False)
    elif write_mode == "overwrite":
        df.to_csv(save_path, index=False, header=True)
    else:  # auto fallback
        df.to_csv(save_path, mode="a" if save_path.exists() else "w",
                  index=False, header=not save_path.exists())

    return df