"""
Module: tracking_launcher.py
Function: Unified tracker execution for SORT, DeepSORT, BoT-SORT.
"""

def run_tracker(tracker_type, detection_result_dir, image_dir, output_dir, config=None):
    """
    Args:
        tracker_type (str): Type of tracker to use ('sort', 'deepsort', 'botsort')
        detection_result_dir (str or Path): Path to YOLO output (no ID)
        image_dir (str or Path): Corresponding image folder
        output_dir (str or Path): Directory to save tracking results
        config (dict): Optional tracker configuration parameters
    """
    pass
