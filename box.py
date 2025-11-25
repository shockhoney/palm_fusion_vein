import os
import cv2
import glob
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================== 配置区域 ==================
# 1. 模型路径 (请修改为你实际的路径)
MODEL_PATH = r'E:\\document\\company\\Yolo-FastestV2-main\\mediapipe\\hand_landmarker.task'

# 2. 数据集路径
INPUT_IMAGES_DIR = '1'  # 你的原图文件夹路径
OUTPUT_LABELS_DIR = 'labels' # 输出txt标签的文件夹路径

# 是否保存可视化图片用于检查 (True/False)
SAVE_VISUALIZATION = True
OUTPUT_VIS_DIR = 'vis_check' 
# =============================================

def to_yolo_format(x_min, y_min, x_max, y_max, img_w, img_h):
    """将坐标转换为 YOLO 归一化格式: cx cy w h"""
    # 限制边界在图片内
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(img_w, x_max), min(img_h, y_max)

    box_w = x_max - x_min
    box_h = y_max - y_min
    cx = x_min + box_w / 2.0
    cy = y_min + box_h / 2.0

    # 归一化
    return cx / img_w, cy / img_h, box_w / img_w, box_h / img_h

def get_square_box_from_points(p1, p2, img_w, img_h):
    """计算两点间的小方框 (用于 5-9 和 13-17)"""
    x1, y1 = p1.x * img_w, p1.y * img_h
    x2, y2 = p2.x * img_w, p2.y * img_h
    
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    # 以两点间距作为边长生成正方形
    half_size = dist / 2
    return cx - half_size, cy - half_size, cx + half_size, cy + half_size

def get_strict_roi(points, img_w, img_h):
    """计算严格的最小外接矩形 (用于 palm_roi)"""
    xs = [p.x * img_w for p in points]
    ys = [p.y * img_h for p in points]
    return min(xs), min(ys), max(xs), max(ys)

def main():
    # 初始化 MediaPipe
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # 创建输出目录
    os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)
    if SAVE_VISUALIZATION:
        os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    # 获取所有图片
    image_files = glob.glob(os.path.join(INPUT_IMAGES_DIR, '*.[jJ][pP]*[gG]')) # 匹配 jpg, jpeg, png
    print(f"找到 {len(image_files)} 张图片，开始处理...")

    for img_path in image_files:
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # 加载图片
        mp_image = mp.Image.create_from_file(img_path)
        img_h, img_w = mp_image.height, mp_image.width
        
        # 检测
        detection_result = detector.detect(mp_image)
        
        # 准备写入标签
        labels = [] # 存储格式: (class_id, cx, cy, w, h)
        
        # 如果需要可视化，读取cv2图像
        vis_img = cv2.imread(img_path) if SAVE_VISUALIZATION else None

        for hand_landmarks in detection_result.hand_landmarks:
            # 获取关键点
            p0, p2, p5 = hand_landmarks[0], hand_landmarks[2], hand_landmarks[5]
            p9, p13, p17 = hand_landmarks[9], hand_landmarks[13], hand_landmarks[17]

            # --- 1. point_0 (Class 0): 5号和9号之间 ---
            box0 = get_square_box_from_points(p5, p9, img_w, img_h)
            labels.append((0, *to_yolo_format(*box0, img_w, img_h)))

            # --- 2. identity_point_0 (Class 1): 13号和17号之间 ---
            box1 = get_square_box_from_points(p13, p17, img_w, img_h)
            labels.append((1, *to_yolo_format(*box1, img_w, img_h)))

            # --- 3. palm_roi (Class 2): 0, 2, 5, 17 严格矩形 ---
            box2 = get_strict_roi([p0, p2, p5, p17], img_w, img_h)
            labels.append((2, *to_yolo_format(*box2, img_w, img_h)))

            # --- 可视化绘制 (可选) ---
            if vis_img is not None:
                # 绿色 Class 0
                cv2.rectangle(vis_img, (int(box0[0]), int(box0[1])), (int(box0[2]), int(box0[3])), (0, 255, 0), 2)
                # 黄色 Class 1
                cv2.rectangle(vis_img, (int(box1[0]), int(box1[1])), (int(box1[2]), int(box1[3])), (0, 255, 255), 2)
                # 蓝色 Class 2
                cv2.rectangle(vis_img, (int(box2[0]), int(box2[1])), (int(box2[2]), int(box2[3])), (255, 0, 0), 2)

        # 保存 txt 文件
        txt_path = os.path.join(OUTPUT_LABELS_DIR, filename + '.txt')
        with open(txt_path, 'w') as f:
            for lbl in labels:
                # 格式: class cx cy w h
                f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")

        # 保存可视化图片
        if vis_img is not None:
            cv2.imwrite(os.path.join(OUTPUT_VIS_DIR, filename + '.jpg'), vis_img)

    print("处理完毕！")

if __name__ == "__main__":
    main()