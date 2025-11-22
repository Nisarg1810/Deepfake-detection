"""Face detection utilities (mock implementation)."""
import os
import numpy as np
from PIL import Image
from pathlib import Path


def extract_faces_from_frames(frames_dir, output_dir):
    """
    Mock face extraction. Creates dummy face crops for testing.
    In a real implementation, this would use a face detector.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
    face_count = 0
    
    for frame_file in frame_files:
        # Create a dummy face crop (smaller image)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        face_path = os.path.join(output_dir, f"face_{face_count:04d}.jpg")
        img.save(face_path)
        face_count += 1
    
    return face_count
