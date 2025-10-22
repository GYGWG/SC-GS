import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor
import json

def segment_clean_rendered_images(input_folder="clean_rendered", output_folder="sam_segmented"):
    """
    Segment clean rendered images using SAM.
    
    Args:
        input_folder: Path to clean rendered images
        output_folder: Path to save segmentation results
        
    Returns:
        dict: Dictionary mapping camera indices to segmentation results
    """
    # Load SAM model
    sam_checkpoint = "models/sam_vit_h.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Get all PNG files from clean_rendered
    image_list = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])
    print(f"Found {len(image_list)} clean rendered images to segment")
    
    segmentation_results = {}

    # Process each clean rendered image
    for i, fname in enumerate(tqdm(image_list, desc="Segmenting clean rendered images")):
        image_path = os.path.join(input_folder, fname)
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        
        # Set image for SAM
        predictor.set_image(image_rgb)
        
        # HSV ranges for red, green, blue (assuming these are the ball colors)
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        green_lower = np.array([40, 50, 50])
        green_upper = np.array([80, 255, 255])
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([140, 255, 255])
        
        def find_largest_center(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_area = 0
            best_center = None
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 100: # Filter small noise
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
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        green_mask = cv2.inRange(image_hsv, green_lower, green_upper)
        blue_mask = cv2.inRange(image_hsv, blue_lower, blue_upper)
        
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
        
        for color, center in color_centers.items():
            if center is None:
                continue
            masks, _, _ = predictor.predict(
                point_coords=np.array([center]),
                point_labels=np.array([1]),
                multimask_output=False,
            )
            mask = masks[0]
            label_mask[(mask == 1) & (label_mask == 0)] = label_map[color]
        
        color_map = {
            0: (0, 0, 0),
            1: (0, 0, 255),
            2: (0, 255, 0),
            3: (255, 0, 0),
        }
        
        vis_mask = np.zeros((*label_mask.shape, 3), dtype=np.uint8)
        for label, color in color_map.items():
            vis_mask[label_mask == label] = color
        
        # Output paths
        base_fname = os.path.splitext(fname)[0]
        label_out_path = os.path.join(output_folder, f"{base_fname}_label.png")
        vis_out_path = os.path.join(output_folder, f"{base_fname}_vis.png")
        json_out_path = os.path.join(output_folder, f"{base_fname}_label.json")
        
        # Generate JSON metadata
        unique_labels = np.unique(label_mask)
        unique_labels = unique_labels[unique_labels > 0]
        
        json_data = {
            "image_name": fname,
            "label": label_mask.astype(int).tolist()
        }
        
        with open(json_out_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        cv2.imwrite(label_out_path, label_mask)
        cv2.imwrite(vis_out_path, vis_mask)
        
        # Store results for return
        cam_idx = int(base_fname[3:])  # Extract number from "cam000"
        segmentation_results[cam_idx] = {
            'label_mask': label_mask,
            'vis_mask': vis_mask,
            'json_data': json_data
        }

    print(f"\nSegmentation complete! Results saved to: {output_folder}")
    print(f"Processed {len(image_list)} images")
    
    return segmentation_results

# For backward compatibility, run the function if script is executed directly
if __name__ == "__main__":
    segment_clean_rendered_images()