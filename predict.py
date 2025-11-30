"""
Prediction Script:
Reads a video, applies SIFT subtraction on-the-fly, and predicts using the trained model.
"""

import os
import argparse
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
from utils import get_sift_frame_difference

def predict_on_video(video_path, model_path, output_path):
    # Constants
    IMAGE_SIZE = (256, 256)
    SEQUENCE_LENGTH = 20
    CLASSES = ["Left Turn", "Right Turn"]
    
    # Load Model
    print(f"Loading model: {model_path}")
    if not os.path.exists(model_path):
        print("Error: Model file not found.")
        return
    model = load_model(model_path)
    
    # Open Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Initialize SIFT
    sift = cv2.SIFT_create()
    
    # Queue to store the sequence of processed frames
    frames_queue = deque(maxlen=SEQUENCE_LENGTH)
    
    prev_gray = None
    frame_idx = 0
    print("Processing video...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Preprocess Current Frame
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray_resized = cv2.resize(curr_gray, IMAGE_SIZE)
        
        # 2. Compute SIFT Difference if we have a previous frame
        if prev_gray is not None:
            # Apply SIFT Alignment & Subtraction
            diff = get_sift_frame_difference(prev_gray, curr_gray_resized, sift_detector=sift)
            
            # Convert grayscale diff to BGR and Normalize
            diff_bgr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
            normalized = diff_bgr / 255.0
            frames_queue.append(normalized)
            
        prev_gray = curr_gray_resized
        
        # 3. Predict
        label_text = "Initializing..."
        color = (0, 255, 255) # Yellow
        
        # Only predict when we have filled the queue (20 frames)
        if len(frames_queue) == SEQUENCE_LENGTH:
            # Expand dims to match model input: (1, 20, 256, 256, 3)
            probs = model.predict(np.expand_dims(list(frames_queue), axis=0), verbose=0)[0]
            class_idx = np.argmax(probs)
            confidence = probs[class_idx]
            
            if confidence > 0.7: # Confidence threshold
                label_text = f"{CLASSES[class_idx]} ({confidence:.2f})"
                color = (0, 255, 0) if class_idx == 0 else (0, 0, 255) # Green for Left, Red for Right
            else:
                label_text = "No Signal"
                color = (200, 200, 200)
        
        # 4. Draw & Write
        cv2.putText(frame, label_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        writer.write(frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames")

    cap.release()
    writer.release()
    print(f"Output saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, required=True, help='Path to .h5 model file')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to output video')
    args = parser.parse_args()
    
    predict_on_video(args.video, args.model, args.output)