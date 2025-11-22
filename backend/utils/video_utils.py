"""Video processing utilities (mock implementation)."""
import os
import numpy as np
from PIL import Image
from pathlib import Path


def extract_frames(video_path, output_dir, fps=1):
    """
    Mock frame extraction. Creates dummy frames for testing.
    In a real implementation, this would use cv2 to extract frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create 10 dummy frames
    num_frames = 10
    for i in range(num_frames):
        # Create a random image
        img_array = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        img.save(frame_path)
    
    return num_frames
