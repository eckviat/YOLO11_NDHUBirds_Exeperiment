"""
Module: authentic_tracking_launcher.py
Description: 負責執行正宗 (Authentic) 的追蹤器演算法 (SORT, DeepSORT)，
             直接操作追蹤器物件而非透過 Ultralytics 介面。
"""
import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# --- 環境變數設定 (防止 Windows 下 YOLO 與 DeepSORT 搶資源死鎖) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

IMPORTS_OK = True

# ==========================================================
# 嘗試載入追蹤器函式庫
# ==========================================================
# 1. 載入 SORT (優先使用本地實作)
try:
    from sort_tracker import Sort
except ImportError:
    print("❌ 無法載入本地 SORT，請確認 utils/sort_tracker.py 存在")
    IMPORTS_OK = False

# 2. 載入 DeepSORT (外部庫)
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError as e:
    print(f"❌ 無法載入 DeepSORT: {e}")
    IMPORTS_OK = False

# 支援的檔案副檔名
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


# ==========================================================
# 追蹤器初始化
# ==========================================================
def _initialize_tracker(tracker_type, conf, iou):
    """
    根據類型初始化追蹤器物件。

    Args:
        tracker_type (str): 'SORT' 或 'DeepSORT'。
        conf (float): 信心分數閾值 (部分追蹤器可能用到)。
        iou (float): IoU 閾值。

    Returns:
        object: 初始化後的追蹤器實例。

    Raises:
        ImportError: 若相關套件未安裝。
        ValueError: 若指定了未知的追蹤器類型。
    """
    if not IMPORTS_OK:
        raise ImportError("追蹤器套件未就緒")

    if tracker_type == 'SORT':
        return Sort(max_age=30, min_hits=3, iou_threshold=iou)
    
    elif tracker_type == 'DeepSORT':
        # 參數參考主流實作優化
        return DeepSort(
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0, 
            max_cosine_distance=0.2, 
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet", # 使用輕量級模型加速
            half=True,            # 開啟半精度
            bgr=True,
            embedder_gpu=False    # Windows 下設為 False 以避免死鎖
        )
    else:
        raise ValueError(f"未知的追蹤器: {tracker_type}")


# ==========================================================
# 核心執行邏輯
# ==========================================================
def run_authentic_tracker(model, tracker_type, source_path, output_dir, conf=0.4, iou=0.3, half=True, imgsz=640):
    """
    執行正宗追蹤器 (SORT / DeepSORT) 並回傳結果。

    Args:
        model: 已載入的 YOLO 模型。
        tracker_type (str): 'SORT' 或 'DeepSORT'。
        source_path (str | Path): 影像來源路徑。
        output_dir (str | Path): 輸出目錄 (此函式目前僅回傳 DataFrame，未直接存檔)。
        conf (float): YOLO 偵測信心閾值。
        iou (float): NMS/Matching IoU 閾值。
        half (bool): 是否使用半精度推論。

    Returns:
        tuple: (pd.DataFrame, float) -> (追蹤結果, 總耗時秒數)。
    """
    if not IMPORTS_OK:
        return pd.DataFrame(), 0.0

    print(f"🔄 初始化 {tracker_type}...")
    try:
        tracker = _initialize_tracker(tracker_type, conf, iou)
    except Exception as e:
        print(f"❌ 初始化錯誤: {e}")
        return pd.DataFrame(), 0.0

    # --- 路徑處理與來源識別 ---
    source_path = Path(source_path)
    files = []
    cap = None
    
    if source_path.is_file():
        print(f"🎥 讀取影片: {source_path.name}")
        cap = cv2.VideoCapture(str(source_path))
    elif source_path.is_dir():
        # 優先搜尋 images 子目錄
        target_img_dir = source_path / "images"
        search_dir = target_img_dir if target_img_dir.exists() else source_path
        
        # 抓取圖片檔案
        files = sorted([p for p in search_dir.glob('*') if p.suffix.lower() in IMAGE_EXTS])
        
        if not files:
            # 若無圖片，嘗試搜尋影片檔
            vid_files = [p for p in source_path.glob('*') if p.suffix.lower() in VIDEO_EXTS]
            if vid_files:
                print(f"🎥 切換為影片模式: {vid_files[0].name}")
                cap = cv2.VideoCapture(str(vid_files[0]))
            else:
                print(f"❌ 錯誤: {source_path} 內無影像資料")
                return pd.DataFrame(), 0.0
        else:
            print(f"🖼️ 讀取圖片序列: 共 {len(files)} 張")
    else:
        print(f"❌ 路徑錯誤: {source_path}")
        return pd.DataFrame(), 0.0

    # --- 開始追蹤迴圈 ---
    results_list = []
    frame_id = 0
    t_start = time.perf_counter()

    print(f"🚀 開始追蹤 ({tracker_type}) - 請耐心等候...")

    while True:
        # 1. 讀取影像
        if cap:
            ret, frame = cap.read()
            if not ret: break
        else:
            if frame_id >= len(files): break
            frame = cv2.imread(str(files[frame_id]))
        
        if frame is None: break

        # 2. YOLO 推論 (Detect)
        yolo_results = model.predict(frame, conf=conf, iou=iou, verbose=False, half=half, imgsz=imgsz)
        
        # 3. 資料格式轉換 (Data Formatting)
        dets_to_track = []
        
        for r in yolo_results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                if tracker_type == 'DeepSORT':
                    # DeepSORT 格式: ([left, top, w, h], conf, class_id)
                    w, h = x2 - x1, y2 - y1
                    dets_to_track.append(([x1, y1, w, h], score, cls_id))
                    
                elif tracker_type == 'SORT':
                    # SORT 格式: [x1, y1, x2, y2, score] (np.array)
                    dets_to_track.append([x1, y1, x2, y2, score])

        # 4. 更新追蹤器 (Update Tracker)
        final_tracks = [] # 格式: [x1, y1, x2, y2, id, cls, conf]

        if tracker_type == 'DeepSORT':
            tracks = tracker.update_tracks(dets_to_track, frame=frame)
            for track in tracks:
                if not track.is_confirmed(): continue
                ltrb = track.to_ltrb()
                track_id = int(track.track_id)
                class_id = int(track.det_class) if track.det_class is not None else 0
                conf_val = track.det_conf if track.det_conf is not None else -1.0
                final_tracks.append([ltrb[0], ltrb[1], ltrb[2], ltrb[3], track_id, class_id, conf_val])

        elif tracker_type == 'SORT':
            np_dets = np.array(dets_to_track) if len(dets_to_track) > 0 else np.empty((0, 5))
            tracks = tracker.update(np_dets)
            for t in tracks:
                # SORT 回傳: [x1, y1, x2, y2, id]
                final_tracks.append([t[0], t[1], t[2], t[3], int(t[4]), 0, -1.0])

        # 5. 收集結果
        for ft in final_tracks:
            results_list.append({
                'frame_id': frame_id,
                'obj_id': ft[4],
                'x1': ft[0], 'y1': ft[1], 'x2': ft[2], 'y2': ft[3],
                'conf': ft[6], 'cls_id': ft[5]
            })

        # 進度顯示 (每 10 幀更新一次)
        if frame_id % 10 == 0:
            elapsed = time.perf_counter() - t_start
            fps = frame_id / elapsed if elapsed > 0 else 0
            print(f"   Frame {frame_id} | Dets: {len(dets_to_track)} | FPS: {fps:.2f}", end='\r')

        frame_id += 1

    t_end = time.perf_counter()
    if cap: cap.release()
    print(f"\n   ✅ 完成。共 {frame_id} 幀，總耗時 {t_end - t_start:.2f} 秒。")
    
    df = pd.DataFrame(results_list)
    return df, (t_end - t_start)