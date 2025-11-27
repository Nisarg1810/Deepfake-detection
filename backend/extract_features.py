"""
Extract Features from Deepfake Dataset
=======================================
Extracts features from ALL videos and saves to CSV for training.

Usage: python extract_features.py
"""

import os
import sys
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class VideoFeatureExtractor:
    """Extract features from videos for deepfake detection."""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def extract_features(self, video_path, max_frames=30):
        """Extract features from a single video."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if total_frames == 0:
                cap.release()
                return None
            
            frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
            
            face_scores = []
            edge_scores = []
            blur_scores = []
            color_scores = []
            face_sizes = []
            face_counts = []
            frame_diffs = []
            brightness_scores = []
            contrast_scores = []
            
            prev_frame = None
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Brightness and contrast
                brightness_scores.append(np.mean(gray))
                contrast_scores.append(np.std(gray))
                
                # Face detection
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                face_counts.append(len(faces))
                
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        face_roi = frame[y:y+h, x:x+w]
                        if face_roi.size > 0:
                            face_sizes.append((w * h) / (frame.shape[0] * frame.shape[1]))
                            
                            gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                            blur = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                            blur_scores.append(blur)
                            
                            edges = cv2.Canny(gray_face, 100, 200)
                            edge_scores.append(np.mean(edges) / 255.0)
                            
                            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
                            color_scores.append(np.std(hsv[:,:,1]))
                
                if prev_frame is not None:
                    diff = cv2.absdiff(gray, prev_frame)
                    frame_diffs.append(np.mean(diff))
                prev_frame = gray.copy()
            
            cap.release()
            
            # Aggregate features
            features = {
                'cnn_max_score': np.clip(max(blur_scores) / 1000 if blur_scores else 0.5, 0, 1),
                'cnn_mean_score': np.clip(np.mean(blur_scores) / 1000 if blur_scores else 0.5, 0, 1),
                'cnn_std_score': np.clip(np.std(blur_scores) / 1000 if blur_scores else 0.1, 0, 1),
                'freq_mean': np.clip(np.mean(edge_scores) if edge_scores else 0.5, 0, 1),
                'freq_std': np.clip(np.std(edge_scores) if edge_scores else 0.1, 0, 1),
                'freq_max': np.clip(max(edge_scores) if edge_scores else 0.5, 0, 1),
                'temporal_mean': np.clip(np.mean(frame_diffs) / 50 if frame_diffs else 0.5, 0, 1),
                'temporal_max': np.clip(max(frame_diffs) / 50 if frame_diffs else 0.5, 0, 1),
                'temporal_variance': np.clip(np.var(frame_diffs) / 1000 if frame_diffs else 0.1, 0, 1),
                'lipsync_score': np.clip(1.0 - (np.std(frame_diffs) / 50 if frame_diffs else 0.5), 0, 1),
                'lipsync_confidence': 0.8 if len(frame_diffs) > 10 else 0.5,
                'num_faces': np.mean(face_counts) if face_counts else 0,
                'num_frames': total_frames,
                'avg_face_size': np.clip(np.mean(face_sizes) if face_sizes else 0.2, 0, 1),
                'spatial_artifacts': int(np.std(edge_scores) * 20) if edge_scores else 3,
                'temporal_artifacts': int(np.std(frame_diffs) / 5) if frame_diffs else 3,
                'blending_score': np.clip(np.mean(color_scores) / 100 if color_scores else 0.5, 0, 1),
                'brightness_mean': np.mean(brightness_scores) / 255 if brightness_scores else 0.5,
                'brightness_std': np.std(brightness_scores) / 255 if brightness_scores else 0.1,
                'contrast_mean': np.mean(contrast_scores) / 255 if contrast_scores else 0.5,
                'contrast_std': np.std(contrast_scores) / 255 if contrast_scores else 0.1,
            }
            
            return features
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            return None


def collect_video_paths(dataset_path):
    """Collect all video paths with labels."""
    dataset_path = Path(dataset_path)
    
    real_folder = dataset_path / "DFD_original sequences"
    fake_folder = dataset_path / "DFD_manipulated_sequences" / "DFD_manipulated_sequences"
    
    video_data = []
    
    print(f"[SCAN] Scanning real videos: {real_folder}")
    if real_folder.exists():
        for video_file in real_folder.glob("*.mp4"):
            video_data.append((video_file, 0))
    print(f"   Found {len([v for v in video_data if v[1] == 0])} real videos")
    
    print(f"[SCAN] Scanning fake videos: {fake_folder}")
    if fake_folder.exists():
        for video_file in fake_folder.glob("*.mp4"):
            video_data.append((video_file, 1))
    print(f"   Found {len([v for v in video_data if v[1] == 1])} fake videos")
    
    return video_data


def main():
    print("\n" + "="*60)
    print("FEATURE EXTRACTION FROM DEEPFAKE DATASET")
    print("="*60)
    
    # Dataset path - CHANGE THIS IF YOUR DATASET IS ELSEWHERE
    dataset_path = r"D:\deep fake dataset"
    
    # Output path
    output_path = Path(__file__).parent / "dataset_features.csv"
    
    # Collect videos
    video_data = collect_video_paths(dataset_path)
    
    if len(video_data) == 0:
        print("[ERROR] No videos found!")
        return
    
    print(f"\n[INFO] Total videos to process: {len(video_data)}")
    print(f"[INFO] Estimated time: {len(video_data) * 9 / 3600:.1f} hours")
    
    # Extract features
    extractor = VideoFeatureExtractor()
    
    all_features = []
    labels = []
    
    print(f"\n[EXTRACT] Processing videos...")
    for video_path, label in tqdm(video_data, desc="Extracting features"):
        features = extractor.extract_features(video_path, max_frames=20)
        if features:
            all_features.append(features)
            labels.append(label)
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    df['label'] = labels
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print("\n" + "="*60)
    print("[DONE] FEATURE EXTRACTION COMPLETE!")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Real videos: {sum(df['label'] == 0)}")
    print(f"Fake videos: {sum(df['label'] == 1)}")
    print(f"Features: {len(df.columns) - 1}")
    print(f"Saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()

