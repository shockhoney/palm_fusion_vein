from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import argparse
import time
import sys
from pathlib import Path
from tqdm import tqdm

def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    if not detection_result.hand_landmarks:
        return annotated_image
    
    for idx, hand_landmarks in enumerate(detection_result.hand_landmarks):
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image, hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )
        
        # 添加左右手标签
        h, w, _ = annotated_image.shape
        x_coords = [lm.x for lm in hand_landmarks]
        y_coords = [lm.y for lm in hand_landmarks]
        text_pos = (int(min(x_coords) * w), int(min(y_coords) * h) - 10)
        cv2.putText(annotated_image, detection_result.handedness[idx][0].category_name,
                    text_pos, cv2.FONT_HERSHEY_DUPLEX, 1, (88, 205, 54), 1, cv2.LINE_AA)
    
    return annotated_image

def extract_palm_roi(image, hand_landmarks, scale=0.6, padding=5):
    h, w = image.shape[:2]
    palm_indices = [0, 2, 5, 9, 13, 17] 
    x_coords = [int(hand_landmarks[i].x * w) for i in palm_indices]
    y_coords = [int(hand_landmarks[i].y * h) for i in palm_indices]
    
    x_center = (min(x_coords) + max(x_coords)) / 2
    y_center = (min(y_coords) + max(y_coords)) / 2
    roi_w = (max(x_coords) - min(x_coords)) * scale
    roi_h = (max(y_coords) - min(y_coords)) * scale
    
    min_size = 32
    roi_w = max(roi_w, min_size)
    roi_h = max(roi_h, min_size)
    
    x_min = max(0, int(x_center - roi_w / 2) - padding)
    y_min = max(0, int(y_center - roi_h / 2) - padding)
    x_max = min(w, int(x_center + roi_w / 2) + padding)
    y_max = min(h, int(y_center + roi_h / 2) + padding)
    
    roi = image[y_min:y_max, x_min:x_max]
    
    if roi.size == 0 or roi.shape[0] < 16 or roi.shape[1] < 16:
        return None, None
    
    return roi, (x_min, y_min, x_max - x_min, y_max - y_min)

def process_single_image(image_path, detector, output_dir, scale=0.6, padding=5, 
                         save_landmarks=True, save_roi=True):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, 0, 0
        
        if len(img.shape) == 2: 
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        detection_result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb))
        base_name = Path(image_path).stem
        
        num_hands = len(detection_result.hand_landmarks) if detection_result.hand_landmarks else 0
        
        saved_files = 0
        
        if save_landmarks:
            landmarks_dir = os.path.join(output_dir, "landmarks")
            os.makedirs(landmarks_dir, exist_ok=True)
            annotated = draw_landmarks_on_image(img_rgb, detection_result)
            landmark_path = os.path.join(landmarks_dir, f"{base_name}_landmarks.jpg")
            if cv2.imwrite(landmark_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)):
                saved_files += 1
        
        if save_roi and num_hands > 0:
            roi_dir = os.path.join(output_dir, "roi")
            os.makedirs(roi_dir, exist_ok=True)
            for idx, hand_lm in enumerate(detection_result.hand_landmarks):
                roi, bbox = extract_palm_roi(img_rgb, hand_lm, scale, padding)
                if roi is None or roi.size == 0:
                    continue
                suffix = "_roi.jpg" if num_hands == 1 else f"_roi_{idx}.jpg"
                roi_path = os.path.join(roi_dir, base_name + suffix)
                if cv2.imwrite(roi_path, cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)):
                    saved_files += 1
        
        return True, num_hands, saved_files
    except Exception as e:
        return False, 0, 0

def batch_process(input_path, output_dir, model_path, scale=0.6, padding=5, 
                  save_landmarks=True, save_roi=True, num_hands=2):

    detector = None
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['MEDIAPIPE_DISABLE_GPU'] = '0' 
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'        
              
        try:
            base_options = python.BaseOptions(
                model_asset_path=model_path,
                delegate=python.BaseOptions.Delegate.GPU
            )
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=num_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            detector = vision.HandLandmarker.create_from_options(options)
        except Exception as gpu_error:
            print(f"GPU初始化失败: {gpu_error}")
            
            base_options = python.BaseOptions(
                model_asset_path=model_path,
                delegate=python.BaseOptions.Delegate.CPU
            )
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=num_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            detector = vision.HandLandmarker.create_from_options(options)
        
        input_path_obj = Path(input_path)
        if input_path_obj.is_file():
            image_files = [str(input_path_obj)]
        elif input_path_obj.is_dir():
            exts = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
            image_files = [str(f) for ext in exts for f in input_path_obj.rglob(f'*{ext}')]
        else:
            return None
        
        if not image_files:
            return None
        
        unique_image_files = list(set(image_files))
        
        os.makedirs(output_dir, exist_ok=True)
        
        success_count = total_hands = total_saved = 0

        pbar = tqdm(
            unique_image_files, 
            desc='Processing batch images', 
            ncols=100,
            file=sys.stdout,  
            mininterval=0.1,   
            maxinterval=1.0,   
            disable=False,      
            leave=True         
        )
        
        for img_path in pbar:
            success, num, saved = process_single_image(img_path, detector, output_dir, 
                                                       scale, padding, save_landmarks, save_roi)
            if success:
                success_count += 1
                total_hands += num
                total_saved += saved       
        return success_count, total_hands, total_saved
    finally:
        if detector is not None:
            del detector
    
def main():
    parser = argparse.ArgumentParser(description='roi_extraction:')
    parser.add_argument('-i', '--input', required=True, help='输入图像/文件夹路径')
    parser.add_argument('-o', '--output', default='roi_output', help='输出目录（默认: roi_output）')
    args = parser.parse_args()
    
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models', 'hand_landmarker.task'
    )  
    result = batch_process(args.input, args.output, model_path, 
                 scale=0.6, padding=5, save_landmarks=True, 
                 save_roi=True, num_hands=2)
    if result:
        print(f"\n最终统计: 成功={result[0]}, 手部={result[1]}, 文件={result[2]}")

if __name__ == "__main__":
    main()