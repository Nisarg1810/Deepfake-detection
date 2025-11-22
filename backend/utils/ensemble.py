"""Ensemble fusion utilities (mock implementation)."""
import numpy as np


class EnsembleCombiner:
    """Combines multiple detector scores using weighted fusion."""
    
    def __init__(self):
        # Weights for different detectors
        self.weights = {
            "cnn": 0.4,
            "frequency": 0.2,
            "temporal": 0.2,
            "lipsync": 0.2
        }
    
    def combine(self, aggregation_result):
        """Combine all scores into final verdict."""
        scores = []
        
        # CNN score
        scores.append(aggregation_result["max_score"] * self.weights["cnn"])
        
        # Frequency score
        scores.append(aggregation_result["frequency_score"] * self.weights["frequency"])
        
        # Temporal score
        scores.append(aggregation_result["temporal_max"] * self.weights["temporal"])
        
        # Lip-sync score (inverted - lower sync = more suspicious)
        if aggregation_result.get("lip_sync_score") is not None:
            lipsync_score = 1.0 - aggregation_result["lip_sync_score"]
            scores.append(lipsync_score * self.weights["lipsync"])
        
        final_score = sum(scores)
        
        # Determine label
        if final_score > 0.7:
            label = "LIKELY_MANIPULATED"
        elif final_score > 0.5:
            label = "POSSIBLY_MANIPULATED"
        else:
            label = "LIKELY_AUTHENTIC"
        
        return {
            "final_score": final_score,
            "final_label": label
        }
