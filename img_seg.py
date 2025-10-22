import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
import json

# 设置模型路径和类型
sam_checkpoint = "models/sam_vit_h.pth"
model_type = "vit_h"

# 初始化模型和设备
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device)
predictor = SamPredictor(sam)

# 输入输出路径
input_folder = "data/bouncingballs/train"
output_folder = "output_mask_colored"
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹
image_list = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])

# HSV 范围定义（注意红色环绕 180 度）
red_lower1 = np.array([0, 100, 100])
red_upper1 = np.array([10, 255, 255])
red_lower2 = np.array([160, 100, 100])
red_upper2 = np.array([180, 255, 255])
green_lower = np.array([40, 50, 50])
green_upper = np.array([80, 255, 255])
blue_lower = np.array([100, 50, 50])
blue_upper = np.array([140, 255, 255])

# 白色和灰色区域范围（低饱和度，高亮度）
neutral_lower = np.array([0, 0, 120])
neutral_upper = np.array([180, 60, 255])

# 对每张图像进行处理
for i, fname in enumerate(tqdm(image_list)):
    image_path = os.path.join(input_folder, fname)
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    # 设置图片到 SAM
    predictor.set_image(image_rgb)

    # 寻找每种颜色的 mask 中心坐标
    def find_largest_center(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        best_center = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            if area > largest_area:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    best_center = [cX, cY]
                    largest_area = area
        return best_center

    red_mask1 = cv2.inRange(image_hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(image_hsv, red_lower2, red_upper2)
    raw_red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    shadow_mask = cv2.inRange(image_hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
    red_mask = cv2.bitwise_and(raw_red_mask, cv2.bitwise_not(shadow_mask))

    green_mask = cv2.inRange(image_hsv, green_lower, green_upper)
    blue_mask = cv2.inRange(image_hsv, blue_lower, blue_upper)

    # 强制颜色 mask 不重叠
    red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(green_mask))
    red_mask = cv2.bitwise_and(red_mask, cv2.bitwise_not(blue_mask))

    green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(red_mask))
    green_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(blue_mask))

    blue_mask = cv2.bitwise_and(blue_mask, cv2.bitwise_not(red_mask))
    blue_mask = cv2.bitwise_and(blue_mask, cv2.bitwise_not(green_mask))

    # neutral_mask = cv2.inRange(image_hsv, neutral_lower, neutral_upper)

    color_centers = {
        'red': find_largest_center(red_mask),
        'green': find_largest_center(green_mask),
        'blue': find_largest_center(blue_mask),
    }

    label_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    label_map = {
        'red': 1,
        'green': 2,
        'blue': 3
    }

    # 应用 SAM mask 的时候避免覆盖已有 label
    for color, center in color_centers.items():
        if center is None:
            continue
        masks, _, _ = predictor.predict(
            point_coords=np.array([center]),
            point_labels=np.array([1]),
            multimask_output=False,
        )
        mask = masks[0]
        label_mask[(mask == 1) & (label_mask == 0)] = label_map[color]  # 只填空白区域


    # 添加 neutral 区域（白色和灰色）作为 0 类背景
    # label_mask[neutral_mask > 0] = 0

    color_map = {
        0: (0, 0, 0),
        1: (255, 0, 0),
        2: (0, 255, 0),
        3: (0, 0, 255),
    }

    vis_mask = np.zeros((*label_mask.shape, 3), dtype=np.uint8)
    for label, color in color_map.items():
        vis_mask[label_mask == label] = color

    # Resize masks to 400x400 using nearest neighbor interpolation to preserve label values
    target_size = (400, 400)
    label_mask_resized = cv2.resize(label_mask, target_size, interpolation=cv2.INTER_NEAREST)
    vis_mask_resized = cv2.resize(vis_mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    out_path = os.path.join(output_folder, fname.replace(".png", "_label.png"))
    vis_out_path = os.path.join(output_folder, fname.replace(".png", "_vis.png"))

    # Get unique labels (excluding background 0)
    unique_labels = np.unique(label_mask_resized)
    unique_labels = unique_labels[unique_labels > 0]  # Remove background (0)
    
    # Create JSON with the new format
    json_label = {
        "image_name": fname,
        "label": label_mask_resized.astype(int).tolist()
    }

    json_path = os.path.join(output_folder, fname.replace(".png", "_label.json"))
    with open(json_path, "w") as f:
        json.dump(json_label, f, indent=2)

    # Save resized images
    cv2.imwrite(out_path, label_mask_resized)
    cv2.imwrite(vis_out_path, vis_mask_resized)
