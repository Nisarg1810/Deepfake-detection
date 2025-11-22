"""Abnormality analysis utilities (mock implementation)."""
import numpy as np


class AbnormalityAnalyzer:
    """Analyzes spatial and temporal abnormalities in faces."""
    
    def generate_abnormality_report(self, faces_dir, detections, temporal_mean, 
                                    temporal_max, lip_sync_score=None):
        """Generate abnormality report."""
        spatial_artifacts = []
        
        # Mock spatial artifacts
        if detections:
            high_score_faces = [d for d in detections if d["fake_score"] > 0.7]
            for face in high_score_faces[:3]:  # Top 3
                spatial_artifacts.append({
                    "face_file": face["face_file"],
                    "artifact_type": "blending_boundary",
                    "confidence": face["fake_score"]
                })
        
        temporal_artifacts = []
        if temporal_max > 0.6:
            temporal_artifacts.append({
                "type": "temporal_inconsistency",
                "score": temporal_max
            })
        
        return {
            "spatial_artifacts": spatial_artifacts,
            "temporal_artifacts": temporal_artifacts,
            "lip_sync_score": lip_sync_score
        }
