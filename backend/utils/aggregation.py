"""Score aggregation utilities (mock implementation)."""
import numpy as np


def aggregate_scores(detections, lip_sync_score=None, temporal_mean=0.5, temporal_max=0.5):
    """Aggregate all detection scores."""
    if not detections:
        return {
            "total_faces": 0,
            "max_score": 0.5,
            "mean_score": 0.5,
            "frequency_score": 0.5,
            "lip_sync_score": lip_sync_score,
            "temporal_mean": temporal_mean,
            "temporal_max": temporal_max
        }
    
    fake_scores = [d["fake_score"] for d in detections]
    freq_scores = [d.get("freq_score", 0.5) for d in detections]
    
    return {
        "total_faces": len(detections),
        "max_score": max(fake_scores),
        "mean_score": np.mean(fake_scores),
        "frequency_score": np.mean(freq_scores),
        "lip_sync_score": lip_sync_score,
        "temporal_mean": temporal_mean,
        "temporal_max": temporal_max
    }


def decide_verdict(aggregation_result):
    """Decide verdict based on aggregated scores."""
    max_score = aggregation_result["max_score"]
    mean_score = aggregation_result["mean_score"]
    
    reasons = []
    
    if max_score > 0.7:
        label = "LIKELY_MANIPULATED"
        reasons.append("High CNN detection score")
    elif mean_score > 0.6:
        label = "POSSIBLY_MANIPULATED"
        reasons.append("Elevated average CNN score")
    else:
        label = "LIKELY_AUTHENTIC"
        reasons.append("Low manipulation scores")
    
    confidence = abs(max_score - 0.5) * 2  # Scale to 0-1
    
    return {
        "label": label,
        "confidence": confidence,
        "reason": reasons
    }


def generate_result(job_id, video_path, frames, detections, lip_sync_score, 
                   temporal_mean, temporal_max, abnormality_report=None, 
                   technique_report=None):
    """Generate complete result dictionary."""
    aggregation_result = aggregate_scores(
        detections, lip_sync_score, temporal_mean, temporal_max
    )
    
    verdict = decide_verdict(aggregation_result)
    
    return {
        "job_id": job_id,
        "video_path": video_path,
        "frames": frames,
        "faces": len(detections),
        "aggregation": aggregation_result,
        "verdict": verdict,
        "abnormality_report": abnormality_report,
        "technique_report": technique_report
    }
