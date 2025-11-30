"""
Preprocessing Script:
Iterates through raw frames, applies SIFT alignment, subtracts background,
and saves the result for training.
"""

import os
import cv2
from utils import get_sift_frame_difference, sort_key

def process_sift_subtraction(input_dir, output_dir, obj_classes, target_size=(256, 256)):
    # Initialize SIFT
    sift = cv2.SIFT_create()

    for class_name in obj_classes:
        class_input_dir = os.path.join(input_dir, class_name)
        
        if not os.path.exists(class_input_dir):
            print(f'Skipping {class_name}, directory not found.')
            continue
        
        # Get list of video clip directories
        subdirs = [d for d in os.listdir(class_input_dir) if os.path.isdir(os.path.join(class_input_dir, d))]
        
        print(f'\nProcessing class: {class_name} ({len(subdirs)} clips)')
        
        for subdir in subdirs:
            input_folder = os.path.join(class_input_dir, subdir)
            output_folder = os.path.join(output_dir, class_name, subdir)
            
            os.makedirs(output_folder, exist_ok=True)
            
            # Load file names and sort
            file_list = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png'))]
            file_list.sort(key=sort_key)
            
            if len(file_list) < 2:
                continue
            
            # Loop through frames to calculate difference
            for i in range(len(file_list) - 1):
                img_file_1 = os.path.join(input_folder, file_list[i])
                img_file_2 = os.path.join(input_folder, file_list[i + 1])
                
                # Read as Grayscale for SIFT
                img1 = cv2.imread(img_file_1, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.imread(img_file_2, cv2.IMREAD_GRAYSCALE)
                
                if img1 is None or img2 is None:
                    continue
                
                img1 = cv2.resize(img1, target_size)
                img2 = cv2.resize(img2, target_size)
                
                # Apply SIFT Alignment & Subtraction
                output = get_sift_frame_difference(img1, img2, sift_detector=sift)
                
                # Save the processed image
                out_file = os.path.join(output_folder, f"Img-{i:04d}.jpg")
                cv2.imwrite(out_file, output)

def main():
    # Define paths
    RAW_DATA_DIR = 'data/raw'
    PROCESSED_DATA_DIR = 'data/processed'
    obj_classes = ['turn_left', 'turn_right']
    
    process_sift_subtraction(RAW_DATA_DIR, PROCESSED_DATA_DIR, obj_classes)

if __name__ == '__main__':
    main()