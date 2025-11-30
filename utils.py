"""
Utility functions for Turn Light Detection.
Includes SIFT alignment, natural sorting, and dataset creation with sliding window.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re

def sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def get_sift_frame_difference(img_prev_gray, img_curr_gray, sift_detector=None):
    """
    Aligns the previous frame to the current frame using SIFT and returns the difference image.
    
    Args:
        img_prev_gray: Previous frame (Grayscale).
        img_curr_gray: Current frame (Grayscale).
        sift_detector: Pre-initialized cv2.SIFT_create() object for performance.
    """
    if sift_detector is None:
        sift_detector = cv2.SIFT_create()
    
    # 1. Detect keypoints and descriptors
    kp1, des1 = sift_detector.detectAndCompute(img_prev_gray, None)
    kp2, des2 = sift_detector.detectAndCompute(img_curr_gray, None)
    
    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        return np.zeros_like(img_curr_gray)
    
    # 2. Match descriptors between frames
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    if len(matches) < 5:
        return np.zeros_like(img_curr_gray)
    
    # 3. Calculate Translation Shift (Alignment)
    # Use the top 5 best matches to calculate the average X and Y shift.
    x_diff = []
    y_diff = []
    
    for m in matches[:5]:
        p1 = kp1[m.queryIdx].pt
        p2 = kp2[m.trainIdx].pt
        x_diff.append(p2[0] - p1[0]) # X shift
        y_diff.append(p2[1] - p1[1]) # Y shift
    
    x_shift = np.mean(x_diff)
    y_shift = np.mean(y_diff)
    
    # 4. Align the previous frame
    # Construct the translation matrix: [[1, 0, x_shift], [0, 1, y_shift]]
    T = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
    img_prev_aligned = cv2.warpAffine(img_prev_gray, T, (img_prev_gray.shape[1], img_prev_gray.shape[0]))
    
    # 5. Subtract (Current - Aligned Previous)
    diff = cv2.subtract(img_curr_gray, img_prev_aligned)
    
    return diff

def create_dataset(dataset_dir, classes_list, sequence_length, image_height, image_width):
    """
    Loads the dataset using a Sliding Window approach to increase training data.
    """
    features = []
    labels = []
    video_files_paths = []
    
    # How many frames to move the window.
    STRIDE = 5
    
    for class_index, class_name in enumerate(classes_list):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            print(f'Warning: Directory not found {class_dir}')
            continue
            
        print(f'Extracting Data for Class: {class_name}')
        
        # Iterate over each video clip folder
        video_folders = os.listdir(class_dir)
        
        for video_folder_name in video_folders:
            video_folder_path = os.path.join(class_dir, video_folder_name)
            
            if not os.path.isdir(video_folder_path):
                continue

            # 1. Get all image files
            image_files = [f for f in os.listdir(video_folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            # 2. Sorting
            image_files.sort(key=sort_key)
            
            if len(image_files) < sequence_length:
                continue
            
            # Read all frames in this clip into memory
            all_frames = []
            valid_clip = True
            for img_name in image_files:
                img_path = os.path.join(video_folder_path, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    valid_clip = False
                    break
                img = cv2.resize(img, (image_height, image_width))
                all_frames.append(img / 255.0) # Normalize to 0-1
            
            if not valid_clip:
                continue

            # 3. Apply Sliding Window
            # Slice the long clip into multiple sequences of length 20
            num_frames = len(all_frames)
            
            for i in range(0, num_frames - sequence_length + 1, STRIDE):
                sequence = all_frames[i : i + sequence_length]
                
                if len(sequence) == sequence_length:
                    features.append(sequence)
                    labels.append(class_index)
                    video_files_paths.append(f"{video_folder_path}_clip_{i}")

    features = np.asarray(features)
    labels = np.array(labels)
    
    return features, labels, video_files_paths

def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    """Helper function to plot training history."""
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]
    epochs = range(len(metric_value_1))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label=metric_name_2)
    plt.title(str(plot_name))
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{plot_name.replace(" ", "_")}.png')
    plt.close()