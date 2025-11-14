from pathlib import Path
import cv2
import numpy as np

def load_image(image_path):
    try:
        image_data = np.fromfile(str(image_path), dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        if image is None:
            print(f"无法加载图像: {image_path}")
            return None
        return image
    except Exception as e:
        print(f"加载图像时出错: {e}")
        return None

def save_image(image, output_path):
    try:
        success, encoded_img = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if success:
            encoded_img.tofile(str(output_path))
            return True
        else:
            print(f"图像编码失败: {output_path}")
            return False
    except Exception as e:
        print(f"保存图像时出错: {e}")
        return False

def ir_preprocess_and_binary(gray):
    # 去噪与增强
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    #  使用更严格的Otsu阈值
    _, binary_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 基于统计的阈值
    mean_val = np.mean(blurred)
    std_val = np.std(blurred)
    # 使用更严格的阈值，减少背景粘连
    stat_thresh = mean_val - 0.3 * std_val
    _, binary_stat = cv2.threshold(blurred, stat_thresh, 255, cv2.THRESH_BINARY)
    # 自适应阈值
    binary_adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    binary = cv2.bitwise_and(binary_otsu, binary_stat)
    binary = cv2.bitwise_and(binary, binary_adaptive)
    # 先进行开运算去除小噪声
    ksize_small = max(3, min(gray.shape[0], gray.shape[1]) // 100)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize_small, ksize_small))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=2)
    # 再进行闭运算连接手掌区域
    ksize = max(5, min(gray.shape[0], gray.shape[1]) // 80)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return binary

def detect_ir_palm_center_distance_transform(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = ir_preprocess_and_binary(gray)
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None
        # 选择最大轮廓（手掌）
        palm_contour = max(contours, key=cv2.contourArea)
        # 创建手掌掩码
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [palm_contour], 255)
        # 距离变换
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        # 找到最大距离点（手掌中心）
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(dist_transform)
        center_x, center_y = max_loc
        # 计算内切圆半径
        max_radius = int(max_val)
        # 验证中心点是否在手掌内部
        if mask[center_y, center_x] == 0:
            palm_points = np.where(mask > 0)
            if len(palm_points[0]) > 0:
                distances = np.sqrt((palm_points[1] - center_x)**2 + (palm_points[0] - center_y)**2)
                min_idx = np.argmin(distances)
                center_y, center_x = palm_points[0][min_idx], palm_points[1][min_idx]
                max_radius = int(dist_transform[center_y, center_x])
        
        # 创建结果图像
        result_image = image.copy()
        cv2.drawContours(result_image, [palm_contour], -1, (0, 255, 0), 2)
        cv2.circle(result_image, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.circle(result_image, (center_x, center_y), max_radius, (255, 0, 0), 2)
        cv2.putText(result_image, f"Center: ({center_x}, {center_y})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_image, f"Radius: {max_radius}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return (center_x, center_y), max_radius, result_image
        
    except Exception as e:
        print(f"距离变换检测失败: {e}")
        return None, None, None

def extract_roi_with_inner_circle_ir(image_path, output_dir="roi_extraction_results_ir"):
    try:
        image = load_image(image_path)
        if image is None:
            return False
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        # 检测手掌中心和内切圆
        center, radius, result_image = detect_ir_palm_center_distance_transform(image)
        if center is None:
            print("未检测到手掌中心")
            return False
        center_x, center_y = center
        # 根据手掌大小动态调整扩展比例
        image_area = image.shape[0] * image.shape[1]
        palm_area = np.pi * radius * radius
        palm_ratio = palm_area / image_area
        
        if palm_ratio > 0.15:  # 手掌较大
            expand_ratio = 0.7
        elif palm_ratio > 0.08:  # 手掌中等
            expand_ratio = 0.8
        else:  # 手掌较小
            expand_ratio = 0.9
            
        expanded_radius = int(radius * expand_ratio)
        
        # 计算ROI边界，确保不超出图像范围
        x_min = max(0, center_x - expanded_radius)
        y_min = max(0, center_y - expanded_radius)
        x_max = min(image.shape[1], center_x + expanded_radius)
        y_max = min(image.shape[0], center_y + expanded_radius)
        
        # 提取ROI
        roi = image[y_min:y_max, x_min:x_max]
        
        # 保存ROI
        base_name = Path(image_path).stem
        roi_path = output_path / f"{base_name}_roi_circle.jpg"
        
        if save_image(roi, roi_path):
            print(f"ROI已保存: {roi_path}")
            print(f"手掌中心: ({center_x}, {center_y}), 内切圆半径: {radius}")
            
            # 保存带内切圆的可视化图像
            # result_path = output_path / f"{base_name}_with_circle.jpg"
            # save_image(result_image, result_path)
            # print(f"可视化图像已保存: {result_path}")
            
            return True
        else:
            print("ROI保存失败")
            return False
            
    except Exception as e:
        print(f"ROI提取失败: {e}")
        return False

def batch_extract_roi_circle_ir(input_folder, output_dir="roi_extraction_results_ir"):
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"输入文件夹不存在: {input_folder}")
        return
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    # 处理所有图像
    processed_count = 0
    success_count = 0
    
    for image_file in input_path.iterdir():
        if image_file.suffix.lower() in image_extensions:
            print(f"处理图像: {image_file.name}")
            if extract_roi_with_inner_circle_ir(image_file, output_dir):
                success_count += 1
            processed_count += 1
    
    print(f"处理完成: {success_count}/{processed_count} 张图像成功提取ROI")

def batch_process_folder(input_folder, output_dir="batch_processing_results", 
                        detect_palm=True, extract_roi=True, debug=False):
    """
    综合批次处理文件夹功能
    可以同时进行手掌检测和ROI提取
    
    Args:
        input_folder: 输入文件夹路径
        output_dir: 输出文件夹路径
        detect_palm: 是否进行手掌检测
        extract_roi: 是否提取ROI
        debug: 是否保存调试图像
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"输入文件夹不存在: {input_folder}")
        return
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 创建子目录
    if detect_palm:
        palm_output_dir = output_path / "palm_detection"
        palm_output_dir.mkdir(exist_ok=True)
    
    if extract_roi:
        roi_output_dir = output_path / "roi_extraction"
        roi_output_dir.mkdir(exist_ok=True)
    
    if debug:
        debug_output_dir = output_path / "debug_images"
        debug_output_dir.mkdir(exist_ok=True)
    
    # 统计信息
    processed_count = 0
    palm_success_count = 0
    roi_success_count = 0
    
    print(f"开始批次处理文件夹: {input_folder}")
    print(f"输出目录: {output_dir}")
    print(f"手掌检测: {'是' if detect_palm else '否'}")
    print(f"ROI提取: {'是' if extract_roi else '否'}")
    print(f"调试模式: {'是' if debug else '否'}")
    print("-" * 50)
    
    # 获取所有图像文件
    image_files = [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions]
    total_files = len(image_files)
    
    if total_files == 0:
        print("未找到支持的图像文件")
        return
    
    print(f"找到 {total_files} 个图像文件")
    
    # 处理每个图像文件
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{total_files}] 处理: {image_file.name}")
        
        # 手掌检测
        if detect_palm:
            print("  进行手掌检测...")
            if detect_palm_ir(image_file, str(palm_output_dir), debug):
                palm_success_count += 1
                print("  ✓ 手掌检测成功")
            else:
                print("  ✗ 手掌检测失败")
        
        # ROI提取
        if extract_roi:
            print("  提取ROI...")
            if extract_roi_with_inner_circle_ir(image_file, str(roi_output_dir)):
                roi_success_count += 1
                print("  ✓ ROI提取成功")
            else:
                print("  ✗ ROI提取失败")
        
        # 调试图像保存
        if debug:
            print("  保存调试图像...")
            try:
                image = load_image(image_file)
                if image is not None:
                    debug_file_path = debug_output_dir / f"{image_file.stem}_debug"
                    save_ir_debug_images(image, output_path=debug_file_path)
                    print("  ✓ 调试图像已保存")
            except Exception as e:
                print(f"  ✗ 调试图像保存失败: {e}")
        
        processed_count += 1
    
    # 输出统计结果
    print("\n" + "=" * 50)
    print("批次处理完成!")
    print(f"总处理文件数: {processed_count}")
    if detect_palm:
        print(f"手掌检测成功: {palm_success_count}/{processed_count}")
    if extract_roi:
        print(f"ROI提取成功: {roi_success_count}/{processed_count}")
    print(f"所有结果保存在: {output_dir}")
    print("=" * 50)

def batch_process_subfolders(parent_folder, output_base_dir="subfolder_batch_results", 
                           detect_palm=True, extract_roi=True, debug=False):
    """
    批次处理父文件夹中的所有子文件夹
    每个子文件夹对应一个输出文件夹
    
    Args:
        parent_folder: 包含子文件夹的父文件夹路径
        output_base_dir: 输出基础目录
        detect_palm: 是否进行手掌检测
        extract_roi: 是否提取ROI
        debug: 是否保存调试图像
    """
    parent_path = Path(parent_folder)
    if not parent_path.exists():
        print(f"父文件夹不存在: {parent_folder}")
        return
    
    # 获取所有子文件夹
    subfolders = [f for f in parent_path.iterdir() if f.is_dir()]
    
    if not subfolders:
        print("未找到子文件夹")
        return
    
    print(f"开始批次处理子文件夹: {parent_folder}")
    print(f"输出基础目录: {output_base_dir}")
    print(f"手掌检测: {'是' if detect_palm else '否'}")
    print(f"ROI提取: {'是' if extract_roi else '否'}")
    print(f"调试模式: {'是' if debug else '否'}")
    print(f"找到 {len(subfolders)} 个子文件夹")
    print("-" * 60)
    
    # 创建输出基础目录
    output_base_path = Path(output_base_dir)
    output_base_path.mkdir(exist_ok=True)
    
    # 统计信息
    total_subfolders = len(subfolders)
    processed_subfolders = 0
    total_images = 0
    total_palm_success = 0
    total_roi_success = 0
    
    # 处理每个子文件夹
    for i, subfolder in enumerate(subfolders, 1):
        print(f"\n[{i}/{total_subfolders}] 处理子文件夹: {subfolder.name}")
        
        # 为每个子文件夹创建对应的输出目录
        subfolder_output_dir = output_base_path / subfolder.name
        subfolder_output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        if detect_palm:
            palm_output_dir = subfolder_output_dir / "palm_detection"
            palm_output_dir.mkdir(exist_ok=True)
        
        if extract_roi:
            roi_output_dir = subfolder_output_dir / "roi_extraction"
            roi_output_dir.mkdir(exist_ok=True)
        
        if debug:
            debug_output_dir = subfolder_output_dir / "debug_images"
            debug_output_dir.mkdir(exist_ok=True)
        
        # 获取子文件夹中的图像文件
        image_extensions = {'.jpg', '.jpeg', '.png'}
        image_files = [f for f in subfolder.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"  子文件夹 {subfolder.name} 中没有找到图像文件")
            continue
        
        print(f"  找到 {len(image_files)} 个图像文件")
        
        # 统计当前子文件夹的处理结果
        subfolder_palm_success = 0
        subfolder_roi_success = 0
        
        # 处理子文件夹中的每个图像
        for j, image_file in enumerate(image_files, 1):
            print(f"    [{j}/{len(image_files)}] 处理: {image_file.name}")
            
            # 手掌检测
            if detect_palm:
                if detect_palm_ir(image_file, str(palm_output_dir), debug):
                    subfolder_palm_success += 1
                    total_palm_success += 1
            
            # ROI提取
            if extract_roi:
                if extract_roi_with_inner_circle_ir(image_file, str(roi_output_dir)):
                    subfolder_roi_success += 1
                    total_roi_success += 1
            
            # 调试图像保存
            if debug:
                try:
                    image = load_image(image_file)
                    if image is not None:
                        debug_file_path = debug_output_dir / f"{image_file.stem}_debug"
                        save_ir_debug_images(image, output_path=debug_file_path)
                except Exception as e:
                    print(f"      调试图像保存失败: {e}")
            
            total_images += 1
        
        # 输出当前子文件夹的统计结果
        print(f"  子文件夹 {subfolder.name} 处理完成:")
        if detect_palm:
            print(f"    手掌检测成功: {subfolder_palm_success}/{len(image_files)}")
        if extract_roi:
            print(f"    ROI提取成功: {subfolder_roi_success}/{len(image_files)}")
        print(f"    输出目录: {subfolder_output_dir}")
        
        processed_subfolders += 1
    
    # 输出总体统计结果
    print("\n" + "=" * 60)
    print("子文件夹批次处理完成!")
    print(f"处理子文件夹数: {processed_subfolders}/{total_subfolders}")
    print(f"总处理图像数: {total_images}")
    if detect_palm:
        print(f"手掌检测成功: {total_palm_success}/{total_images}")
    if extract_roi:
        print(f"ROI提取成功: {total_roi_success}/{total_images}")
    print(f"所有结果保存在: {output_base_dir}")
    print("=" * 60)

def _center_radius_and_roi_from_mask(image, palm_contour, palm_mask, expand_ratio=1.2):
    dist_transform = cv2.distanceTransform(palm_mask, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
    center_x, center_y = max_loc
    radius = int(max_val)

    # 可视化
    vis = image.copy()
    cv2.drawContours(vis, [palm_contour], -1, (0, 255, 0), 2)
    cv2.circle(vis, (center_x, center_y), 5, (0, 0, 255), -1)
    cv2.circle(vis, (center_x, center_y), radius, (255, 0, 0), 2)

    # ROI矩形
    expanded_radius = int(radius * expand_ratio)
    x_min = max(0, center_x - expanded_radius)
    y_min = max(0, center_y - expanded_radius)
    x_max = min(image.shape[1], center_x + expanded_radius)
    y_max = min(image.shape[0], center_y + expanded_radius)
    roi = image[y_min:y_max, x_min:x_max]

    return (center_x, center_y), radius, roi, vis

def extract_roi_with_inner_circle_unified(image_path, mode="ir", output_dir="roi_extraction_results_unified", expand_ratio=1.2):
    """统一的ROI提取：mode in {"ir","rgb"}。"""
    try:
        image = load_image(image_path)
        if image is None:
            return False
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        if mode == "ir":
            palm_contour, palm_mask, _ = detect_ir_palm_region(image)
        # elif mode == "rgb":
        #     palm_contour, palm_mask = detect_rgb_palm_region(image)
        else:
            print(f"不支持的模式: {mode}")
            return False

        if palm_contour is None or palm_mask is None:
            print(f"{mode.upper()}: 未检测到手掌区域")
            return False

        center, radius, roi, vis = _center_radius_and_roi_from_mask(
            image, palm_contour, palm_mask, expand_ratio=expand_ratio
        )

        base_name = Path(image_path).stem
        roi_path = output_path / f"{base_name}_roi_circle.jpg"
        vis_path = output_path / f"{base_name}_with_circle.jpg"
        ok1 = save_image(roi, roi_path)
        ok2 = save_image(vis, vis_path)
        if ok1:
            print(f"{mode.upper()} ROI已保存: {roi_path}")
            if ok2:
                print(f"{mode.upper()} 可视化已保存: {vis_path}")
            return True
        print(f"{mode.upper()} ROI保存失败")
        return False
    except Exception as e:
        print(f"统一ROI提取失败: {e}")
        return False

def batch_extract_roi_circle_unified(input_folder, mode="ir", output_dir="roi_extraction_results_unified", expand_ratio=1.2):
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"输入文件夹不存在: {input_folder}")
        return
    image_extensions = {'.jpg', '.jpeg', '.png'}
    processed_count = 0
    success_count = 0
    for image_file in input_path.iterdir():
        if image_file.suffix.lower() in image_extensions:
            print(f"处理{mode.upper()}图像: {image_file.name}")
            if extract_roi_with_inner_circle_unified(image_file, mode=mode, output_dir=output_dir, expand_ratio=expand_ratio):
                success_count += 1
            processed_count += 1
    print(f"{mode.upper()}处理完成: {success_count}/{processed_count} 张图像成功提取ROI")
# ===================== RGB 手掌检测与ROI提取 =====================
# def detect_rgb_skin_mask(image):
#     # 预处理
#     blurred = cv2.GaussianBlur(image, (5, 5), 0)
#     # YCrCb空间肤色分割
#     ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
#     y, cr, cb = cv2.split(ycrcb)
#     mask_ycrcb = cv2.inRange(ycrcb, (0, 128, 70), (255, 180, 135))
#     # HSV空间肤色分割（适度宽松范围）
#     hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#     mask_hsv1 = cv2.inRange(hsv, (0, 30, 60), (20, 200, 255))
#     mask_hsv2 = cv2.inRange(hsv, (160, 30, 60), (180, 200, 255))
#     mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
#     # 融合两种肤色掩码
#     skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
#     # 形态学清理
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#     skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#     skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     return skin_mask

# def detect_rgb_palm_region(image):
#     mask = detect_rgb_skin_mask(image)
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None, None
#     h, w = image.shape[:2]
#     min_area = h * w * 0.01
#     max_area = h * w * 0.60
#     candidates = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
#     if not candidates:
#         return None, None
#     def score(c):
#         area = cv2.contourArea(c)
#         x, y, bw, bh = cv2.boundingRect(c)
#         rect_area = max(1, bw * bh)
#         fill_ratio = area / rect_area
#         hull = cv2.convexHull(c)
#         hull_area = max(1.0, cv2.contourArea(hull))
#         solidity = area / hull_area
#         per = max(1.0, cv2.arcLength(c, True))
#         circularity = 4 * np.pi * area / (per * per)
#         # 偏向面积大、实心度高、填充度高且不过分圆
#         return 0.4 * solidity + 0.3 * fill_ratio + 0.3 * (1 - abs(circularity - 0.5))
#     best = max(candidates, key=score)
#     palm_mask = np.zeros((h, w), dtype=np.uint8)
#     cv2.fillPoly(palm_mask, [best], 255)
#     return best, palm_mask

# def detect_palm_rgb(image_path, output_dir="palm_detection_results_rgb", debug=False):
#     image = load_image(image_path)
#     if image is None:
#         return False
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True)
#     palm_contour, palm_mask = detect_rgb_palm_region(image)
#     if palm_contour is None:
#         print("RGB: 未检测到手掌区域")
#         return False
#     # 评分与可视化
#     score = calculate_score(palm_contour, image.shape)
#     result_image = image.copy()
#     cv2.drawContours(result_image, [palm_contour], -1, (0, 0, 255), 1)
#     x, y, w, h = cv2.boundingRect(palm_contour)
#     cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
#     cv2.putText(result_image, f"Score: {score}/100", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     base_name = Path(image_path).stem
#     result_path = output_path / f"{base_name}_result.jpg"
#     if save_image(result_image, result_path):
#         print(f"RGB分数: {score}/100")
#         return True
#     print("RGB结果保存失败")
#     return False

# def batch_detect_palms_rgb(input_folder, output_dir="palm_detection_results_rgb"):
#     input_path = Path(input_folder)
#     if not input_path.exists():
#         print(f"输入文件夹不存在: {input_folder}")
#         return
#     image_extensions = {'.jpg', '.jpeg', '.png'}
#     processed_count = 0
#     for image_file in input_path.iterdir():
#         if image_file.suffix.lower() in image_extensions:
#             print(f"处理RGB图像: {image_file.name}")
#             if detect_palm_rgb(image_file, output_dir):
#                 processed_count += 1
#     print(f"RGB成功处理: {processed_count} 张图像")

# def extract_roi_with_inner_circle_rgb(image_path, output_dir="roi_extraction_results"):
#     """基于RGB肤色分割+距离变换的ROI提取"""
#     try:
#         image = load_image(image_path)
#         if image is None:
#             return False
#         output_path = Path(output_dir)
#         output_path.mkdir(exist_ok=True)
#         # 检测手掌掩码
#         palm_contour, palm_mask = detect_rgb_palm_region(image)
#         if palm_contour is None:
#             print("RGB: 未检测到手掌区域")
#             return False
#         # 距离变换找中心与半径
#         dist_transform = cv2.distanceTransform(palm_mask, cv2.DIST_L2, 5)
#         _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
#         center_x, center_y = max_loc
#         radius = int(max_val)
#         # 结果可视化
#         vis = image.copy()
#         cv2.drawContours(vis, [palm_contour], -1, (0, 255, 0), 2)
#         cv2.circle(vis, (center_x, center_y), 5, (0, 0, 255), -1)
#         cv2.circle(vis, (center_x, center_y), radius, (255, 0, 0), 2)
#         # ROI矩形（可按需放大比例）
#         expanded_radius = int(radius * 0.9)
#         x_min = max(0, center_x - expanded_radius)
#         y_min = max(0, center_y - expanded_radius)
#         x_max = min(image.shape[1], center_x + expanded_radius)
#         y_max = min(image.shape[0], center_y + expanded_radius)
#         roi = image[y_min:y_max, x_min:x_max]
#         base_name = Path(image_path).stem
#         roi_path = output_path / f"{base_name}_roi_circle.jpg"
#         vis_path = output_path / f"{base_name}_with_circle.jpg"
#         ok1 = save_image(roi, roi_path)
#         ok2 = save_image(vis, vis_path)
#         if ok1:
#             print(f"RGB ROI已保存: {roi_path}")
#             if ok2:
#                 print(f"RGB 可视化已保存: {vis_path}")
#             return True
#         print("RGB ROI保存失败")
#         return False
#     except Exception as e:
#         print(f"RGB ROI提取失败: {e}")
#         return False

# def batch_extract_roi_circle_rgb(input_folder, output_dir="roi_extraction_results_rgb"):
#     input_path = Path(input_folder)
#     if not input_path.exists():
#         print(f"输入文件夹不存在: {input_folder}")
#         return
#     image_extensions = {'.jpg', '.jpeg', '.png'}
#     processed_count = 0
#     success_count = 0
#     for image_file in input_path.iterdir():
#         if image_file.suffix.lower() in image_extensions:
#             print(f"处理RGB图像: {image_file.name}")
#             if extract_roi_with_inner_circle_rgb(image_file, output_dir):
#                 # 基于RGB肤色分割+距离变换的ROI提取
#                 success_count += 1
#             processed_count += 1
#     print(f"RGB处理完成: {success_count}/{processed_count} 张图像成功提取ROI")



# def detect_ir_palm_region(image, gray=None):
#     if gray is None:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # 统一的 IR 预处理与二值化
#     binary = ir_preprocess_and_binary(gray)
    
#     # 查找轮廓
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return None, None, None
    
#     # 轮廓筛选
#     min_area = gray.shape[0] * gray.shape[1] * 0.03   # 最小面积要求
#     max_area = gray.shape[0] * gray.shape[1] * 0.25   # 最大面积限制
    
#     candidates = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
#     if not candidates:
#         return None, None, None
#     # 二阶段综合评分挑选最佳轮廓
#     def contour_score(c):
#         area = cv2.contourArea(c)
#         x,y,bw,bh = cv2.boundingRect(c)
#         aspect_ratio = bw/bh if bh>0 else 0
#         rect_area = bw*bh if bw*bh>0 else 1
#         fill_ratio = area/rect_area
#         hull = cv2.convexHull(c)
#         hull_area = cv2.contourArea(hull) if hull is not None else 1
#         solidity = area/hull_area if hull_area>0 else 0
#         perimeter = cv2.arcLength(c, True)
#         circularity = 4*np.pi*area/(perimeter*perimeter) if perimeter>0 else 0
#         # 灰度均值特征
#         mask = np.zeros(gray.shape, np.uint8)
#         cv2.drawContours(mask, [c], -1, 255, -1)
#         mean_val = cv2.mean(gray, mask=mask)[0]/255.0
#         # 组合得分
#         shape_score = solidity *0.5+ circularity *0.3+ fill_ratio * 0.2
#         ratio_penalty = max(0,1-abs(aspect_ratio-1.0)*0.8)
#         return shape_score * ratio_penalty * mean_val
#     filtered = []
#     for c in candidates:
#         hull_idx = cv2.convexHull(c, returnPoints=False)
#         if hull_idx is not None and len(hull_idx) >= 3:
#             defects = cv2.convexityDefects(c, hull_idx)
#             if defects is not None and len(defects) < 2:
#                  continue  # 缺陷太少，不像手
#         filtered.append(c)

#     if not filtered:
#         return None, None, None

#     palm_contour = max(filtered, key=contour_score)
#     palm_mask = np.zeros(gray.shape, dtype=np.uint8)
#     cv2.fillPoly(palm_mask, [palm_contour], 255)


#     return palm_contour, palm_mask, binary

def detect_ir_palm_region(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = ir_preprocess_and_binary(gray)
        # 轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None, None
        h, w = gray.shape[:2]
        total_area = float(h * w)
        min_area = total_area * 0.01
        max_area = total_area * 0.70

        # 改进的轮廓选择 - 减少手臂粘连
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        palm_contour = None
        
        # 添加手掌区域定位逻辑
        for c in contours_sorted:
            a = cv2.contourArea(c)
            if min_area <= a <= max_area:
                # 检查轮廓位置 - 手掌通常在图像上半部分
                x, y, w, h = cv2.boundingRect(c)
                center_y = y + h // 2
                image_center_y = gray.shape[0] // 2
                
                # 手掌应该在图像上半部分（减少手臂粘连）
                if center_y < image_center_y * 1.2:  # 允许一定偏差
                    # 检查宽高比 - 手掌不应该太细长
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.5 <= aspect_ratio <= 2.0:  # 合理的宽高比
                        palm_contour = c
                        break
        
        # 如果没找到合适的手掌，选择最大的轮廓
        if palm_contour is None:
            palm_contour = contours_sorted[0]
            
        # 后处理：尝试分离手掌和手臂
        palm_contour = refine_palm_contour(palm_contour, gray)
        
        palm_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillPoly(palm_mask, [palm_contour], 255)
        return palm_contour, palm_mask, binary
    
    except Exception:
        return None, None, None

def refine_palm_contour(contour, gray):
    try:
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 计算轮廓中心
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w//2, y + h//2
        
        # 如果轮廓太高，尝试截取上半部分
        if h > w * 1.5:  # 高度明显大于宽度
            # 计算手掌区域（假设手掌在轮廓的上2/3部分）
            palm_height = int(h * 0.7)
            palm_y = y
            palm_h = palm_height
            
            # 创建手掌区域的掩码
            palm_mask = np.zeros(gray.shape, dtype=np.uint8)
            palm_roi = palm_mask[palm_y:palm_y+palm_h, x:x+w]
            
            # 在手掌区域重新查找轮廓
            palm_contour_roi = contour.copy()
            # 将轮廓点限制在手掌区域内
            palm_contour_roi = palm_contour_roi[palm_contour_roi[:, 0, 1] < palm_y + palm_h]
            palm_contour_roi = palm_contour_roi[palm_contour_roi[:, 0, 1] >= palm_y]
            
            if len(palm_contour_roi) > 10:  # 确保有足够的点
                return palm_contour_roi
        
        # 如果轮廓形状合理，直接返回
        return contour
        
    except Exception:
        return contour

def save_ir_debug_images(image, gray=None, output_path=None):
    # 如果gray参数为None，则从image中提取灰度图像
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 如果output_path为None，使用默认路径
    if output_path is None:
        output_path = Path("debug_images")
        output_path.mkdir(exist_ok=True)
    #  原始灰度图
    cv2.imwrite(str(output_path / "debug_1_original_gray.jpg"), gray)
    # CLAHE增强后
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    cv2.imwrite(str(output_path / "debug_2_clahe_enhanced.jpg"), enhanced)
    # 高斯模糊后
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    cv2.imwrite(str(output_path / "debug_3_blurred.jpg"), blurred)
    #  Otsu二值化
    _, binary1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(str(output_path / "debug_4_otsu_binary.jpg"), binary1)
    # 最终二值化结果
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binary = cv2.morphologyEx(binary1, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite(str(output_path / "debug_5_final_binary.jpg"), binary)

def calculate_score(palm_contour, image_shape):
    # 计算面积比例
    total_pixels = image_shape[0] * image_shape[1]
    area_ratio = cv2.contourArea(palm_contour) / total_pixels
    
    # 计算宽高比
    x, y, w, h = cv2.boundingRect(palm_contour)
    aspect_ratio = w / h if h > 0 else 0
    
    # 1. 面积评分
    if 0.08 <= area_ratio <= 0.25: 
        area_score = 100 - abs(area_ratio - 0.15) * 100
    else:
        area_score = max(0, 100 - abs(area_ratio - 0.15) * 300)
    
    # 2. 形状评分
    if 0.8 <= aspect_ratio <= 1.2:
        shape_score = 100 - abs(aspect_ratio - 1.0) * 25
    else:
        shape_score = max(0, 100 - abs(aspect_ratio - 1.0) * 100)
    
    # 3. 实心度评分
    hull = cv2.convexHull(palm_contour)
    hull_area = cv2.contourArea(hull)
    area = cv2.contourArea(palm_contour)
    solidity = area / hull_area if hull_area > 0 else 0
    solidity_score = min(100, solidity * 100)
    
    # 4. 圆形度评分（手掌不应该太圆）
    perimeter = cv2.arcLength(palm_contour, True)
    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
    circularity_score = max(0, 100 - circularity * 100)  
    
    # 加权综合评分
    total_score = (area_score * 0.3 + shape_score * 0.25 + 
                   solidity_score * 0.2 + circularity_score * 0.25 )
    
    return round(total_score, 2)

def detect_palm_ir(image_path, output_dir="palm_detection_results_ir", debug=False):
    # 加载图像
    image = load_image(image_path)
    if image is None:
        return False
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 检测手掌区域
    palm_contour, palm_mask, binary = detect_ir_palm_region(image)
    
    # 调试模式：保存中间处理步骤
    if debug:
        save_ir_debug_images(image, output_path=output_path)
    
    if palm_contour is None:
        print("未检测到手掌区域")
        return False
    
    # 计算分数
    score = calculate_score(palm_contour, image.shape)
    
    # 创建结果图像
    result_image = image.copy()
    
    # 绘制手掌轮廓
    cv2.drawContours(result_image, [palm_contour], -1, (0, 0, 255), 1)
    
    # 绘制边界框
    x, y, w, h = cv2.boundingRect(palm_contour)
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # 添加分数信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    score_text = f"Score: {score}/100"
    cv2.putText(result_image, score_text, (20, 40), font, 1, (0, 255, 0), 2)
    
    # 保存结果
    base_name = Path(image_path).stem
    result_path = output_path / f"{base_name}_result.jpg"
    
    if save_image(result_image, result_path):
        print(f"{score}/100")
        return True
    else:
        print("保存失败")
        return False

def batch_detect_palms_ir(input_folder, output_dir="palm_detection_results_ir"):
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"输入文件夹不存在: {input_folder}")
        return
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    # 处理所有图像
    processed_count = 0
    
    for image_file in input_path.iterdir():
        if image_file.suffix.lower() in image_extensions:
            print(f"处理图像: {image_file.name}")
            if detect_palm_ir(image_file, output_dir):
                processed_count += 1
    
    print(f"成功处理: {processed_count} 张图像")

def main():
    print("请选择运行模式：")
    print("1. 处理单张ir图像（手掌检测）")
    print("3. 批量处理ir图像（手掌检测）")
    print("4. 提取单张ir图像ROI")
    print("5. 批量提取ir图像ROI")
    print("6. 综合批次处理文件夹（手掌检测+ROI提取）")
    print("7. 批次处理多个子文件夹（每个子文件夹对应一个输出文件夹）")
    # print("8. 处理单张RGB图像（手掌检测）")
    # print("9. 批量处理RGB图像（手掌检测）")
    # print("10. 提取单张RGB图像ROI")
    # print("11. 批量提取RGB图像ROI")
    
    choice = input("请输入选择 (1-7): ").strip()
    
    if choice == "1":
        image_path = input("请输入ir图像路径（回车使用默认）: ").strip()
        if not image_path:
            image_path = "WIN_20250910_14_31_24_Pro.jpg"
        detect_palm_ir(image_path, debug=True)
    elif choice == "2":
        pass
    elif choice == "3":
        folder = input("请输入图像文件夹路径: ").strip()
        batch_detect_palms_ir(folder)
    elif choice == "4":
        image_path = input("请输入图像路径（回车使用默认）: ").strip()
        if not image_path:
            image_path = "WIN_20250910_14_31_24_Pro.jpg"
        extract_roi_with_inner_circle_ir(image_path)
    # 基于IR肤色分割+距离变换的ROI提取
    elif choice == "5":
        folder = input("请输入图像文件夹路径: ").strip()
        batch_extract_roi_circle_ir(folder)
    elif choice == "6":
        print("综合批次处理文件夹功能")
        print("可以同时进行手掌检测和ROI提取")
        print("-" * 40)
        
        folder = input("请输入图像文件夹路径: ").strip()
        if not folder:
            print("文件夹路径不能为空")
            return
        
        # 询问处理选项
        print("\n请选择处理选项:")
        detect_palm = input("是否进行手掌检测? (y/n, 默认y): ").strip().lower()
        detect_palm = detect_palm != 'n'
        
        extract_roi = input("是否提取ROI? (y/n, 默认y): ").strip().lower()
        extract_roi = extract_roi != 'n'
        
        debug = input("是否保存调试图像? (y/n, 默认n): ").strip().lower()
        debug = debug == 'y'
        
        # 询问输出目录
        output_dir = input("请输入输出目录（回车使用默认）: ").strip()
        if not output_dir:
            output_dir = "batch_processing_results"
        
        print(f"\n开始综合批次处理...")
        print(f"输入文件夹: {folder}")
        print(f"输出目录: {output_dir}")
        print(f"手掌检测: {'是' if detect_palm else '否'}")
        print(f"ROI提取: {'是' if extract_roi else '否'}")
        print(f"调试模式: {'是' if debug else '否'}")
        
        confirm = input("\n确认开始处理? (y/n): ").strip().lower()
        if confirm == 'y':
            batch_process_folder(folder, output_dir, detect_palm, extract_roi, debug)
        else:
            print("已取消处理")
    elif choice == "7":
        print("批次处理多个子文件夹功能")
        print("每个子文件夹对应一个输出文件夹")
        print("-" * 50)
        
        parent_folder = input("请输入包含子文件夹的父文件夹路径: ").strip()
        if not parent_folder:
            print("父文件夹路径不能为空")
            return
        
        # 询问处理选项
        print("\n请选择处理选项:")
        detect_palm = input("是否进行手掌检测? (y/n, 默认y): ").strip().lower()
        detect_palm = detect_palm != 'n'
        
        extract_roi = input("是否提取ROI? (y/n, 默认y): ").strip().lower()
        extract_roi = extract_roi != 'n'
        
        debug = input("是否保存调试图像? (y/n, 默认n): ").strip().lower()
        debug = debug == 'y'
        
        # 询问输出目录
        output_base_dir = input("请输入输出基础目录（回车使用默认）: ").strip()
        if not output_base_dir:
            output_base_dir = "subfolder_batch_results"
        
        print(f"\n开始批次处理子文件夹...")
        print(f"父文件夹: {parent_folder}")
        print(f"输出基础目录: {output_base_dir}")
        print(f"手掌检测: {'是' if detect_palm else '否'}")
        print(f"ROI提取: {'是' if extract_roi else '否'}")
        print(f"调试模式: {'是' if debug else '否'}")
        
        confirm = input("\n确认开始处理? (y/n): ").strip().lower()
        if confirm == 'y':
            batch_process_subfolders(parent_folder, output_base_dir, detect_palm, extract_roi, debug)
        else:
            print("已取消处理")
    # elif choice == "8":
    #     image_path = input("请输入RGB图像路径: ").strip()
    #     if not image_path:
    #         image_path = "jpg/rgb/rgb_88.jpg"
    #     detect_palm_rgb(image_path)
    # elif choice == "8":
    #     folder = input("请输入RGB图像文件夹路径: ").strip()
    #     if not folder:
    #         folder = "jpg/rgb"
    #     batch_detect_palms_rgb(folder)
    # elif choice == "9":
    #     image_path = input("请输入RGB图像路径: ").strip()
    #     if not image_path:
    #         image_path = "jpg/rgb/rgb_88.jpg"
    #     # 基于RGB肤色分割+距离变换的ROI提取
    #     extract_roi_with_inner_circle_rgb(image_path)
    # elif choice == "10":
    #     folder = input("请输入RGB图像文件夹路径: ").strip()
    #     if not folder:
    #         folder = "jpg/rgb"
    #     batch_extract_roi_circle_rgb(folder)
    else:
        print("无效选择")
if __name__ == "__main__":
    main()
