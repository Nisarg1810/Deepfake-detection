"""
Unit tests for the deepfake detection pipeline.
"""
import unittest
import numpy as np
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from metrics import calculate_metrics, analyze_output_quality, generate_test_data
from backend.utils.model_cache import model_cache, CNNDetector, TemporalDetector
from backend.utils import aggregation, ensemble


class TestMetrics(unittest.TestCase):
    """Test metrics calculation."""
    
    def test_perfect_accuracy(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        self.assertEqual(metrics['accuracy'], 1.0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)
    
    def test_output_quality_random(self):
        """Test output quality analysis for random performance."""
        quality = analyze_output_quality(0.5)
        self.assertEqual(quality, "Random")
    
    def test_output_quality_medium(self):
        """Test output quality analysis for medium performance."""
        quality = analyze_output_quality(0.7)
        self.assertEqual(quality, "Medium")
    
    def test_output_quality_accurate(self):
        """Test output quality analysis for accurate performance."""
        quality = analyze_output_quality(0.85)
        self.assertEqual(quality, "Accurate")
    
    def test_generate_test_data(self):
        """Test synthetic data generation."""
        y_true, y_pred, scores = generate_test_data(num_samples=50)
        
        self.assertEqual(len(y_true), 50)
        self.assertEqual(len(y_pred), 50)
        self.assertEqual(len(scores), 50)
        self.assertTrue(all(label in [0, 1] for label in y_true))
        self.assertTrue(all(label in [0, 1] for label in y_pred))


class TestDetectors(unittest.TestCase):
    """Test detector classes."""
    
    def test_cnn_detector(self):
        """Test CNN detector."""
        detector = CNNDetector()
        score = detector.predict("dummy_path.jpg")
        
        self.assertIsInstance(score, (float, np.floating))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_cnn_detector_batch(self):
        """Test CNN detector batch prediction."""
        detector = CNNDetector()
        paths = ["path1.jpg", "path2.jpg", "path3.jpg"]
        scores = detector.predict_batch(paths)
        
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0 <= s <= 1 for s in scores))
    
    def test_temporal_detector(self):
        """Test temporal detector."""
        detector = TemporalDetector()
        result = detector.predict_for_face_track("dummy_dir", clip_len=16, stride=8)
        
        self.assertIn("clip_scores", result)
        self.assertIsInstance(result["clip_scores"], list)


class TestAggregation(unittest.TestCase):
    """Test score aggregation."""
    
    def test_aggregate_scores_empty(self):
        """Test aggregation with empty detections."""
        result = aggregation.aggregate_scores([], None, 0.5, 0.5)
        
        self.assertEqual(result['total_faces'], 0)
        self.assertEqual(result['max_score'], 0.5)
    
    def test_aggregate_scores_with_data(self):
        """Test aggregation with sample data."""
        detections = [
            {"fake_score": 0.7, "freq_score": 0.6},
            {"fake_score": 0.8, "freq_score": 0.7},
            {"fake_score": 0.6, "freq_score": 0.5}
        ]
        
        result = aggregation.aggregate_scores(detections, 0.8, 0.6, 0.7)
        
        self.assertEqual(result['total_faces'], 3)
        self.assertEqual(result['max_score'], 0.8)
        self.assertAlmostEqual(result['mean_score'], 0.7, places=2)
    
    def test_decide_verdict_manipulated(self):
        """Test verdict decision for manipulated content."""
        agg_result = {
            "max_score": 0.85,
            "mean_score": 0.75,
            "frequency_score": 0.7
        }
        
        verdict = aggregation.decide_verdict(agg_result)
        
        self.assertIn("MANIPULATED", verdict['label'])
        self.assertGreater(verdict['confidence'], 0.5)
    
    def test_decide_verdict_authentic(self):
        """Test verdict decision for authentic content."""
        agg_result = {
            "max_score": 0.3,
            "mean_score": 0.25,
            "frequency_score": 0.3
        }
        
        verdict = aggregation.decide_verdict(agg_result)
        
        self.assertIn("AUTHENTIC", verdict['label'])


class TestEnsemble(unittest.TestCase):
    """Test ensemble fusion."""
    
    def test_ensemble_combiner(self):
        """Test ensemble combiner."""
        combiner = ensemble.EnsembleCombiner()
        
        agg_result = {
            "max_score": 0.7,
            "frequency_score": 0.6,
            "temporal_max": 0.65,
            "lip_sync_score": 0.8
        }
        
        result = combiner.combine(agg_result)
        
        self.assertIn("final_score", result)
        self.assertIn("final_label", result)
        self.assertGreaterEqual(result['final_score'], 0.0)
        self.assertLessEqual(result['final_score'], 1.0)


class TestModelCache(unittest.TestCase):
    """Test model cache."""
    
    def test_model_cache_initialization(self):
        """Test model cache initialization."""
        cache = model_cache
        cache.initialize()
        
        self.assertIsNotNone(cache.get_cnn_detector())
        self.assertIsNotNone(cache.get_temporal_detector())
        self.assertIsNotNone(cache.get_lipsync_detector())
        self.assertIsNotNone(cache.get_frequency_detector())


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestDetectors))
    suite.addTests(loader.loadTestsFromTestCase(TestAggregation))
    suite.addTests(loader.loadTestsFromTestCase(TestEnsemble))
    suite.addTests(loader.loadTestsFromTestCase(TestModelCache))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    print("\n" + "="*60)
    print("RUNNING UNIT TESTS")
    print("="*60 + "\n")
    
    result = run_tests()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60 + "\n")
    
    if result.wasSuccessful():
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        sys.exit(1)
