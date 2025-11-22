"""Model cache for loading detection models (mock implementation)."""
import numpy as np


class CNNDetector:
    """Mock CNN detector."""
    
    def predict(self, image_path):
        """Predict fake score for a single image."""
        # Return random score between 0.3 and 0.9
        return np.random.uniform(0.3, 0.9)
    
    def predict_batch(self, image_paths):
        """Predict fake scores for multiple images."""
        return [self.predict(p) for p in image_paths]


class TemporalDetector:
    """Mock temporal detector."""
    
    def predict_for_face_track(self, frames_dir, clip_len=16, stride=8):
        """Predict temporal scores for a face track."""
        # Return mock scores
        clip_scores = [np.random.uniform(0.4, 0.8) for _ in range(3)]
        return {"clip_scores": clip_scores}


class LipSyncDetector:
    """Mock lip-sync detector."""
    
    def extract_audio(self, video_path, out_wav):
        """Mock audio extraction."""
        import os
        os.makedirs(os.path.dirname(out_wav), exist_ok=True)
        # Create empty file
        with open(out_wav, 'w') as f:
            f.write("")
    
    def compute_sync_score(self, mouth_frames_dir, audio_path):
        """Compute lip-sync score."""
        # Return random score (higher = better sync)
        return np.random.uniform(0.5, 0.9)


class FrequencyDetector:
    """Mock frequency detector."""
    
    def batch_compute(self, faces_dir, output_debug_dir):
        """Compute frequency scores for all faces."""
        from pathlib import Path
        import os
        
        os.makedirs(output_debug_dir, exist_ok=True)
        
        face_files = sorted(Path(faces_dir).glob("face_*.jpg"))
        scores = {}
        
        for face_file in face_files:
            scores[face_file.name] = np.random.uniform(0.3, 0.8)
        
        return scores


class ModelCache:
    """Cache for all detection models."""
    
    def __init__(self):
        self._cnn_detector = None
        self._temporal_detector = None
        self._lipsync_detector = None
        self._freq_detector = None
    
    def initialize(self):
        """Initialize all models."""
        self._cnn_detector = CNNDetector()
        self._temporal_detector = TemporalDetector()
        self._lipsync_detector = LipSyncDetector()
        self._freq_detector = FrequencyDetector()
    
    def get_cnn_detector(self):
        """Get CNN detector."""
        if self._cnn_detector is None:
            self._cnn_detector = CNNDetector()
        return self._cnn_detector
    
    def get_temporal_detector(self):
        """Get temporal detector."""
        if self._temporal_detector is None:
            self._temporal_detector = TemporalDetector()
        return self._temporal_detector
    
    def get_lipsync_detector(self):
        """Get lip-sync detector."""
        if self._lipsync_detector is None:
            self._lipsync_detector = LipSyncDetector()
        return self._lipsync_detector
    
    def get_frequency_detector(self):
        """Get frequency detector."""
        if self._freq_detector is None:
            self._freq_detector = FrequencyDetector()
        return self._freq_detector


# Global model cache instance
model_cache = ModelCache()
