"""
Module: source_utils.py
Description: 處理輸入來源路徑，自動判斷檔案型態及尋找驗證資料 (data.yaml)。
"""
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ==========================================================
# 來源偵測
# ==========================================================
def detect_source_type(source_path):
    """
    自動判斷來源型態 (影片/影像資料夾/單張影像)。

    Args:
        source_path (str | Path): 輸入路徑。

    Returns:
        Path: 可用的來源路徑 (若是資料夾則可能指向 images 子目錄)。

    Raises:
        FileNotFoundError: 路徑不存在。
        ValueError: 檔案類型不支援。
    """
    p = Path(source_path)

    if not p.exists():
        raise FileNotFoundError(f"來源不存在: {p}")

    if p.is_file():
        if p.suffix.lower() in VIDEO_EXTS or p.suffix.lower() in IMAGE_EXTS:
            return p
        raise ValueError(f"不支援的檔案類型: {p.suffix}")

    if p.is_dir():
        # 優先回傳 images 子目錄
        images_dir = p / "images"
        if images_dir.exists():
            return images_dir
        return p

    raise ValueError(f"無法判斷來源型態: {p}")


# ==========================================================
# 設定檔偵測
# ==========================================================
def detect_validation_yaml(base_path):
    """
    自動尋找 validation 用的 data.yaml。
    搜尋順序: 當前目錄 -> 父目錄 -> 子目錄。

    Args:
        base_path (str | Path): 搜尋起點。

    Returns:
        Path: data.yaml 的路徑。
    """
    p = Path(base_path)
    if not p.exists():
        raise FileNotFoundError(f"資料路徑不存在: {p}")

    # 1. 檢查當前路徑
    if (p / "data.yaml").exists():
        return p / "data.yaml"

    # 2. 檢查上層目錄
    if (p.parent / "data.yaml").exists():
        return p.parent / "data.yaml"

    # 3. 檢查子資料夾
    candidates = list(p.glob("*/data.yaml"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"❌ 無法在 {p} 或其周邊找到 data.yaml")