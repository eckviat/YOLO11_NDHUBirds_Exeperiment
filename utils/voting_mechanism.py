"""
Module: voting_mechanism.py
Description: 實作類別投票修正機制 (Majority/Weighted Voting)。
"""
import time
import pandas as pd
import numpy as np


# ==========================================================
# 投票策略
# ==========================================================
def _vote_majority(cls_ids, confs=None):
    """簡單多數決投票。"""
    if len(cls_ids) == 0: return -1
    modes = pd.Series(cls_ids).mode()
    return int(modes[0])

def _vote_weighted(cls_ids, confs):
    """信心分數加權投票。"""
    if len(cls_ids) == 0: return -1
    df = pd.DataFrame({"cls": cls_ids, "conf": confs})
    return int(df.groupby("cls")["conf"].sum().idxmax())


# ==========================================================
# 核心邏輯
# ==========================================================
def apply_voting_logic(df, window_size, strategy, is_realtime=False):
    """
    對每個追蹤 ID 進行滑動窗口投票修正。

    Args:
        df: 原始 DataFrame。
        window_size: 窗口大小 (int 或 'all')。
        strategy: 'majority' 或 'weighted'。
        is_realtime: True 為因果窗口 (Causal)，False 為中心窗口 (Centered)。
    """
    df_out = df.copy()
    obj_ids = df_out[df_out["obj_id"] != -1]["obj_id"].unique()
    mode_str = "Real-time" if is_realtime else "Offline"
    print(f"   ... 對 {len(obj_ids)} 個目標進行 [{strategy}] 投票 (Win: {window_size}, Mode: {mode_str})")

    for obj_id in obj_ids:
        mask = df_out["obj_id"] == obj_id
        track = df_out[mask].sort_values("frame_id")

        orig_cls = track["cls_id"].values
        orig_conf = track["conf"].values
        new_cls = orig_cls.copy()
        n = len(track)

        # 全局投票 ('all')
        if window_size == "all":
            if not is_realtime:
                final = _vote_weighted(orig_cls, orig_conf) if strategy == "weighted" else _vote_majority(orig_cls)
                new_cls[:] = final
                df_out.loc[mask, "cls_id"] = new_cls
                continue
        
        # 滑動窗口
        win_int = int(window_size) if window_size != "all" else None
        
        for i in range(n):
            if is_realtime:
                start = 0 if window_size == "all" else max(0, i - win_int + 1)
                end = i + 1
            else:
                half = win_int // 2
                start = max(0, i - half)
                end = min(n, i + half + 1)

            w_cls = orig_cls[start:end]
            w_conf = orig_conf[start:end]

            if strategy == "majority":
                new_cls[i] = _vote_majority(w_cls)
            elif strategy == "weighted":
                new_cls[i] = _vote_weighted(w_cls, w_conf)

        df_out.loc[mask, "cls_id"] = new_cls

    return df_out


# ==========================================================
# 執行入口
# ==========================================================
def run_class_voting(input_df, method="weighted", window_size=5, is_realtime=True):
    """
    執行類別投票並計算耗時。

    Returns:
        tuple: (修正後 DataFrame, 平均延遲 ms)
    """
    if input_df.empty or "obj_id" not in input_df.columns:
        print("⚠️ 無資料或 ID，跳過投票。")
        return input_df, 0.0
    
    t_start = time.perf_counter()
    df_out = apply_voting_logic(input_df, window_size, method, is_realtime)
    t_end = time.perf_counter()
    
    total_ms = (t_end - t_start) * 1000
    frames = input_df['frame_id'].nunique()
    latency = total_ms / frames if frames > 0 else 0
    
    print(f"   ⏱️ 投票運算耗時: {total_ms:.2f} ms (平均 {latency:.4f} ms/frame)")
    return df_out, latency