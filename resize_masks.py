#!/usr/bin/env python3
"""
Resize all PNG files in output_mask_colored from 800x800 to 400x400
and generate corresponding *_label.json files.
"""

import os
import json
import numpy as np
from PIL import Image
import glob

def resize_masks_and_generate_labels():
    input_dir = "output_mask_colored"
    output_dir = "output_mask_colored_resized"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PNG files in the input directory
    png_files = glob.glob(os.path.join(input_dir, "*.png"))
    
    print(f"Found {len(png_files)} PNG files to process...")
    
    for png_file in png_files:
        # Get the base filename without extension
        base_name = os.path.splitext(os.path.basename(png_file))[0]
        
        print(f"Processing {base_name}...")
        
        # Load the image
        img = Image.open(png_file)
        
        # Check if it's 800x800
        if img.size == (800, 800):
            # Resize to 400x400
            resized_img = img.resize((400, 400), Image.NEAREST)  # Use NEAREST to preserve label values
            
            # Save resized image
            output_png = os.path.join(output_dir, f"{base_name}.png")
            resized_img.save(output_png)
            print(f"  Resized and saved: {output_png}")
            
            # Generate corresponding label.json
            # Convert image to numpy array to get unique labels
            img_array = np.array(resized_img)
            
            # Get unique labels (excluding 0 which is background)
            unique_labels = np.unique(img_array)
            unique_labels = unique_labels[unique_labels > 0]  # Remove background (0)
            
            # Create label data
            label_data = {
                "image_name": f"{base_name}.png",
                "image_size": [400, 400],
                "num_objects": len(unique_labels),
                "object_ids": unique_labels.tolist(),
                "object_labels": [f"object_{label}" for label in unique_labels]
            }
            
            # Save label.json
            output_json = os.path.join(output_dir, f"{base_name}_label.json")
            with open(output_json, 'w') as f:
                json.dump(label_data, f, indent=2)
            print(f"  Generated label file: {output_json}")
            
        else:
            print(f"  Skipping {base_name} - size is {img.size}, not 800x800")
    
    print(f"\nProcessing complete! Resized images and labels saved to: {output_dir}")

if __name__ == "__main__":
    resize_masks_and_generate_labels()




