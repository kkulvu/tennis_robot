import cv2
import numpy as np
from collections import deque
import math
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class TennisShotDetector:
    def __init__(self):
        # 1. 加载模型
        # 注意：如果显存允许，将两个模型合并为一个训练好的模型会更快
        self.ball_model = YOLO('D:/pytorch/tennis_robot_new/best_ball.pt')
        self.racket_model = YOLO('D:/pytorch/tennis_robot_new/best_racket.pt')
        
        # 2. 改进 DeepSort 配置
        # max_age: 遮挡或丢失后保留多久，网球速度快，设小一点避免错误重连
        # n_init: 确认目标需要的帧数，设小一点以便快速捕捉发球
        self.tracker = DeepSort(
            max_age=15, 
            n_init=2, 
            nms_max_overlap=1.0, 
            max_cosine_distance=0.2,
            max_iou_distance=0.7
        )
        
        # 3. 状态存储
        self.ball_tracks = {}
        self.ball_histories = {} # {track_id: deque([(x,y), ...])}
        self.ball_velocities = {} # {track_id: (vx, vy)}
        
        self.hit_count = 0
        self.last_hit_frame = 0
        self.min_hit_frames = 10  # 击球冷却时间（帧数）
        
        # 4. 视觉与参数
        self.trail_length = 15
        self.colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        
        # 5. 高级击球判断参数
        self.hit_distance_threshold = 80 # 像素距离阈值（根据视频分辨率调整）
        self.velocity_change_threshold = 0.5 # 速度方向变化的余弦相似度阈值

        # 缓存球拍位置
        self.current_rackets = []

    def get_vector_angle(self, v1, v2):
        """计算两个向量之间的夹角余弦值"""
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        norm_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        norm_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0
        return dot_product / (norm_v1 * norm_v2)

    def check_box_overlap(self, box1, box2):
        """检查两个包围盒是否有重叠或非常接近"""
        # box: [x1, y1, x2, y2]
        # 扩大一点球拍的范围进行检测（容错）
        margin = 10
        r_x1, r_y1, r_x2, r_y2 = box2[0]-margin, box2[1]-margin, box2[2]+margin, box2[3]+margin
        b_x1, b_y1, b_x2, b_y2 = box1
        
        # 判断不重叠的情况
        if b_x2 < r_x1 or b_x1 > r_x2 or b_y2 < r_y1 or b_y1 > r_y2:
            return False
        return True

    def detect_and_track(self, frame):
        # --- 1. 检测 ---
        # 提高conf阈值减少误检
        ball_results = self.ball_model(frame, classes=[0], conf=0.25, verbose=False)
        racket_results = self.racket_model(frame, conf=0.25, verbose=False)
        
        # 处理网球检测
        detections = []
        for result in ball_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                conf = float(box.conf[0])
                # 过滤掉过大的框（可能是误检，如人的头部或衣服）
                if w < frame.shape[1] * 0.1 and h < frame.shape[0] * 0.1:
                    detections.append(([x1, y1, w, h], conf, 'ball'))
        
        # 处理球拍检测
        self.current_rackets = []
        for result in racket_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                self.current_rackets.append([x1, y1, x2, y2])

        # --- 2. 跟踪 (DeepSort) ---
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        self.ball_tracks = {}
        
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb() # [left, top, right, bottom]
            bbox = [int(x) for x in ltrb]
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            
            self.ball_tracks[track_id] = bbox
            
            # 更新历史轨迹
            if track_id not in self.ball_histories:
                self.ball_histories[track_id] = deque(maxlen=self.trail_length)
            self.ball_histories[track_id].append(center)
            
        return frame

    def analyze_shots(self, frame_count):
        hit_detected = False
        hit_info = None # 用于存储击球位置以便可视化
        
        # 对每一个正在跟踪的球进行分析
        for track_id, history in self.ball_histories.items():
            if len(history) < 3: # 需要至少3帧数据来计算变化
                continue
                
            # 1. 获取当前位置和包围盒
            curr_pos = history[-1]
            if track_id not in self.ball_tracks:
                continue
            ball_box = self.ball_tracks[track_id]
            
            # 2. 计算速度向量
            # v_curr: 当前帧与前一帧的向量
            v_curr = (history[-1][0] - history[-2][0], history[-1][1] - history[-2][1])
            # v_prev: 前一帧与再前一帧的向量
            v_prev = (history[-2][0] - history[-3][0], history[-2][1] - history[-3][1])
            
            speed = math.sqrt(v_curr[0]**2 + v_curr[1]**2)
            
            # 如果球基本静止，忽略
            if speed < 2: 
                continue

            # 3. 检查与所有球拍的相互作用
            for racket_box in self.current_rackets:
                # --- 核心逻辑 A: 空间重叠 ---
                # 检查球的中心是否足够靠近球拍中心，或者包围盒是否重叠
                racket_center = ((racket_box[0] + racket_box[2]) // 2, (racket_box[1] + racket_box[3]) // 2)
                dist = math.sqrt((curr_pos[0] - racket_center[0])**2 + (curr_pos[1] - racket_center[1])**2)
                
                is_overlapping = self.check_box_overlap(ball_box, racket_box)
                is_close = dist < self.hit_distance_threshold
                
                if is_overlapping or is_close:
                    # --- 核心逻辑 B: 轨迹突变 (Hit Event) ---
                    
                    # 计算两个速度向量的夹角余弦
                    # 1.0 表示同向，-1.0 表示反向，0 表示垂直
                    cosine_sim = self.get_vector_angle(v_prev, v_curr)
                    
                    # 击球判定条件：
                    # 1. 距离足够近/重叠
                    # 2. 且 (方向发生急剧改变 OR 速度突然大幅增加)
                    # 3. 且 冷却时间已过
                    
                    # cosine_sim < 0.5 意味着角度变化超过60度（击球通常会显著改变方向）
                    direction_change = cosine_sim < 0.5 
                    
                    if direction_change and (frame_count - self.last_hit_frame > self.min_hit_frames):
                        self.hit_count += 1
                        self.last_hit_frame = frame_count
                        hit_detected = True
                        hit_info = curr_pos
                        # 检测到击球后跳出循环，避免重复
                        break
            
            if hit_detected:
                break
                
        return hit_detected, hit_info

    def visualize(self, frame, hit_detected, hit_info):
        # 绘制球拍
        for box in self.current_rackets:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 2)
            cv2.putText(frame, "Racket", (box[0], box[1]-5), 0, 0.5, (255, 255, 0), 2)

        # 绘制网球及轨迹
        for track_id, bbox in self.ball_tracks.items():
            color = self.colors[int(track_id) % len(self.colors)]
            
            # 画框
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"ID:{track_id}", (bbox[0], bbox[1]-5), 0, 0.5, color, 2)
            
            # 画轨迹线
            pts = self.ball_histories.get(track_id, [])
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], color, 2)

        # 绘制UI信息
        cv2.rectangle(frame, (10, 10), (250, 60), (0, 0, 0), -1)
        cv2.putText(frame, f"Shots Count: {self.hit_count}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 击球特效
        if hit_detected and hit_info:
            cv2.circle(frame, hit_info, 20, (0, 0, 255), -1) # 实心红圈
            cv2.circle(frame, hit_info, 30, (0, 255, 255), 3) # 黄色光环
            cv2.putText(frame, "HIT!", (hit_info[0]-20, hit_info[1]-30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return frame

def main():
    # 实例化
    detector = TennisShotDetector()
    
    video_path = 'D:/pytorch/tennis_robot_new/test.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # 调整输出尺寸（可选，为了显示方便）
    target_width = 1280
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # 性能优化：如果视频太大，可以先resize，但网球很小，尽量保持原分辨率或不要缩太小
        # frame = cv2.resize(frame, (1280, 720)) 
        
        # 1. 检测与跟踪
        detector.detect_and_track(frame)
        
        # 2. 逻辑分析
        hit_detected, hit_info = detector.analyze_shots(frame_count)
        
        # 3. 可视化
        result_frame = detector.visualize(frame, hit_detected, hit_info)
        
        # 显示
        cv2.imshow('Advanced Tennis Analytics', result_frame)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()