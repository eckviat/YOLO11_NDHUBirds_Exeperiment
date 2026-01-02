"""
Module: sort_tracker.py
Description: SORT (Simple Online and Realtime Tracking) 演算法實作。
             包含 Kalman Filter 狀態預測與 Hungarian Algorithm 關聯匹配。
"""
import numpy as np
from filterpy.kalman import KalmanFilter


# ==========================================================
# 關聯匹配演算法 (Hungarian Algorithm)
# ==========================================================
def linear_assignment(cost_matrix):
    """
    解決線性指派問題 (Linear Assignment Problem)。
    優先嘗試使用 lap 函式庫，若無則降級使用 scipy。
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i],i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


# ==========================================================
# IoU 計算
# ==========================================================
def iou_batch(bb_test, bb_gt):
    """
    計算批次邊界框之間的 IoU 矩陣。
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)
    
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1]) +
        (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return(o)


# ==========================================================
# Kalman Filter 狀態追蹤器
# ==========================================================
class KalmanBoxTracker(object):
    """
    代表單一追蹤目標的 Kalman Filter 狀態。
    狀態向量: [u, v, s, r, u', v', s'] (中心x, 中心y, 面積, 寬高比, 及其速度)
    """
    count = 0
    def __init__(self, bbox):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # 狀態轉移矩陣 F
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        # 觀測矩陣 H
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. 
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01
        
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """使用新的觀測框更新狀態"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(self.convert_bbox_to_z(bbox))

    def predict(self):
        """推進狀態至下一時刻"""
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.convert_bbox_to_z(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """回傳當前預測的邊界框 [x1, y1, x2, y2]"""
        return self.convert_x_to_bbox(self.kf.x)

    def convert_bbox_to_z(self, bbox):
        """轉換 [x1,y1,x2,y2] 為 [x,y,s,r]"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h
        r = w / float(h)
        return np.array([x, y, s, r]).reshape((4, 1))

    def convert_x_to_bbox(self, x, score=None):
        """轉換 [x,y,s,r] 回 [x1,y1,x2,y2]"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / w
        if score is None:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2.]).reshape((1,4))
        else:
            return np.array([x[0]-w/2., x[1]-h/2., x[0]+w/2., x[1]+h/2., score]).reshape((1,5))


# ==========================================================
# SORT 主程式
# ==========================================================
class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Args:
            max_age (int): 追蹤目標消失幾幀後刪除。
            min_hits (int): 初始化軌跡所需的最小偵測次數。
            iou_threshold (float): 關聯匹配的 IoU 閾值。
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        更新追蹤器狀態。
        
        Args:
            dets (np.array): 偵測框 [[x1,y1,x2,y2,score], ...]
            
        Returns:
            np.array: 追蹤結果 [[x1,y1,x2,y2,obj_id], ...]
        """
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        
        # 1. 預測現有軌跡
        for t, trk in enumerate(trks):
            self.trackers[t].predict()
            d = self.trackers[t].get_state()[0]
            trk[:] = [d[0], d[1], d[2], d[3], 0]
            if np.any(np.isnan(d)):
                to_del.append(t)
                
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # 2. 關聯匹配 (Detection <-> Track)
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # 3. 更新匹配成功的軌跡
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # 4. 為未匹配的偵測建立新軌跡
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i,:])
            self.trackers.append(trk)
            
        # 5. 輸出結果與清理過期軌跡
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0,5))

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold):
        """使用 Hungarian Algorithm 進行 IoU 匹配。"""
        if len(trackers) == 0:
            return np.empty((0,2), dtype=int), np.arange(len(detections)), np.empty((0,5), dtype=int)
            
        iou_matrix = iou_batch(detections, trackers)
        
        if min(iou_matrix.shape) > 0:
            a = (iou_matrix > iou_threshold).astype(np.int32)
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
                matched_indices = np.stack(np.where(a), axis=1)
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty((0,2))
            
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:,0]:
                unmatched_detections.append(d)
                
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:,1]:
                unmatched_trackers.append(t)
                
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
                
        if len(matches) == 0:
            matches = np.empty((0,2), dtype=int)
        else:
            matches = np.concatenate(matches, 0)
            
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)