"""
Module: evaluate_class_stability.py
Function: Analyze frame-by-frame class stability for object detection results.
"""
import time

def compute_class_stability(label_dir):
    """
    Args:
        label_dir (str or Path): Path to the directory containing YOLO detection results without ID.
    Returns:
        dict: Dictionary of per-video class change statistics.
    """
    pass

def compute_latency(frame):
    start = time.time()
    results = model(frame)
    end = time.time()
    latency_ms = (end - start) * 1000