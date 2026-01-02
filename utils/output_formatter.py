"""
Module: output_formatter.py
Description: 負責標準化偵測與追蹤結果，並處理檔案命名與儲存。
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ==========================================================
# 結果儲存函式
# ==========================================================
def save_results(result, frame_id, save_path, source=None, save_mode="auto"):
    """
    將單幀 Ultralytics 結果標準化並寫入 CSV。

    Args:
        result: Ultralytics Result 物件。
        frame_id (int): 當前幀 ID。
        save_path (Path): 目標 CSV 路徑。
        source (Path): 原始輸入來源。
        save_mode (str): "auto", "video", "folder", "image"。

    Returns:
        pd.DataFrame: 該幀的資料 DataFrame。
    """
    source = Path(source) if source else None

    # 1. 決定儲存模式
    if save_mode == "auto":
        if source is None: save_mode = "none"
        elif source.is_dir(): save_mode = "folder"
        elif source.suffix.lower() in VIDEO_EXTS: save_mode = "video"
        elif source.suffix.lower() in IMAGE_EXTS: save_mode = "image"
        else: save_mode = "none"

    if save_mode == "none":
        return pd.DataFrame()

    # 決定 source_name 與寫入模式
    if save_mode in ["video", "folder"] and source:
        name = source.parent.name if source.name.lower() in ["images", "labels", "gt"] else source.name
        write_mode = "append"
    elif save_mode == "image" and source:
        name, write_mode = source.stem, "overwrite"
    else:
        name, write_mode = "unknown", "append"

    # 2. 提取資料建立 DataFrame
    rows = []
    if hasattr(result, 'boxes') and result.boxes.xyxy.numel() > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            
            obj_id = int(box.id[0]) if hasattr(box, "id") and box.id is not None else -1

            rows.append({
                "source_name": name,
                "frame_id": frame_id,
                "obj_id": obj_id,
                "cls_id": int(box.cls[0]),
                "conf": float(box.conf[0]),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "w": w, "h": h, "x": x1, "y": y1
            })
    
    df = pd.DataFrame(rows)

    # 3. 寫入 CSV
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["source_name", "frame_id", "obj_id", "cls_id", "conf", "x1", "y1", "x2", "y2", "w", "h"]

    if write_mode == "overwrite":
        df.to_csv(save_path, index=False, columns=cols)
    else:
        if not save_path.exists():
            pd.DataFrame(columns=cols).to_csv(save_path, index=False)
        if not df.empty:
            df.reindex(columns=cols, fill_value=0).to_csv(save_path, mode="a", index=False, header=False)
            
    return df


# ==========================================================
# 檔名生成工具
# ==========================================================
def get_detailed_name(seq, stage, tracker, method=None, win=None, realtime=None, ext=".csv"):
    """
    生成標準化的實驗檔案名稱。
    格式: {序列}_{階段}_{追蹤器}_{方法}_w{窗口}_{RT模式}_{時間}.csv
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [seq, stage, tracker]
    
    if method: parts.append(method)
    if win: parts.append(f"w{win}" if win != 'all' else "wall")
    if realtime is not None: parts.append("RT" if realtime else "RTOff")
        
    parts.append(timestamp)
    return "_".join(parts) + ext