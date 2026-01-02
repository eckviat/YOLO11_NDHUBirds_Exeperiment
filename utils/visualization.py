"""
Module: visualization.py
Description: 將 CSV 追蹤結果繪製回影片，支援不同模式 (Raw, Voted, Detection)。
"""
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ==========================================================
# 繪圖輔助
# ==========================================================
def draw_text_with_bg(img, text, x, y, font_scale=0.5, thickness=1, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """在圖片上繪製帶有背景框的文字。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = int(x), int(y)
    img_h, img_w = img.shape[:2]
    
    if y - text_h - 5 < 0: y = text_h + 10
    if x + text_w > img_w: x = img_w - text_w
    
    cv2.rectangle(img, (x, y - text_h - baseline - 4), (x + text_w, y + 2), bg_color, -1)
    cv2.putText(img, text, (x, y - 2), font, font_scale, text_color, thickness)


# ==========================================================
# 影片生成主函式
# ==========================================================
def generate_video_from_csv(source_path, csv_path, output_video_path, mode='tracking', fps=30):
    """
    根據 CSV 結果生成可視化影片。

    Args:
        source_path (Path): 原始圖片資料夾。
        csv_path (Path): 追蹤結果 CSV。
        output_video_path (Path): 輸出影片路徑。
        mode (str): 'detection' (灰), 'raw' (紅), 'voted' (綠)。
    """
    source_path = Path(source_path)
    csv_path = Path(csv_path)
    output_video_path = Path(output_video_path)
    
    if not csv_path.exists():
        print(f"⚠️ CSV 不存在: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"⚠️ CSV 為空: {csv_path}")
        return

    # 尋找圖片
    img_dir = source_path / "images" if (source_path / "images").exists() else source_path
    img_files = sorted([f for f in img_dir.glob('*.jpg')] + [f for f in img_dir.glob('*.png')])
    if not img_files:
        print(f"❌ 找不到圖片: {img_dir}")
        return

    # 初始化影片寫入器
    first_img = cv2.imread(str(img_files[0]))
    h, w = first_img.shape[:2]
    out = cv2.VideoWriter(str(output_video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    colors = {
        'detection': (192, 192, 192), # 灰
        'raw': (0, 0, 200),           # 紅
        'voted': (0, 200, 0)          # 綠
    }
    box_color = colors.get(mode, (0, 255, 255))

    print(f"🎬 正在生成影片 ({mode}): {output_video_path.name}")
    grouped = df.groupby('frame_id')

    for i, img_file in enumerate(tqdm(img_files, desc="Rendering")):
        frame_idx = i
        img = cv2.imread(str(img_file))
        
        if frame_idx in grouped.groups:
            frame_data = grouped.get_group(frame_idx)
            for _, row in frame_data.iterrows():
                if 'x1' in row:
                    x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                elif 'w' in row:
                    x1, y1 = int(row['x']), int(row['y'])
                    x2, y2 = x1 + int(row['w']), y1 + int(row['h'])
                else: continue

                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
                
                conf = row.get('conf', 1.0)
                cls = int(row['cls_id'])
                oid = int(row.get('obj_id', -1))
                
                label = f"Cls:{cls} {conf:.2f}" if mode == 'detection' else f"ID:{oid} C:{cls}"
                draw_text_with_bg(img, label, x1, y1, bg_color=box_color)

        out.write(img)

    out.release()
    print(f"✅ 影片已儲存。\n")