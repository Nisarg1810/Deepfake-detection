"""Mouth region extraction utilities (mock implementation)."""
import os
import numpy as np
from PIL import Image
from pathlib import Path


def extract_mouth_frames(faces_dir, output_dir):
    """
    Mock mouth extraction. Creates dummy mouth crops for testing.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    face_files = sorted(Path(faces_dir).glob("face_*.jpg"))
    mouth_count = 0
    
    for face_file in face_files:
        # Create a dummy mouth crop
        img_array = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        mouth_path = os.path.join(output_dir, f"mouth_{mouth_count:04d}.jpg")
        img.save(mouth_path)
        mouth_count += 1
    
    return mouth_count
