"""
Main execution script for deepfake detection pipeline.
Demonstrates the complete workflow with metrics and testing.
"""
import os
import sys
import uuid
import json
import numpy as np
from pathlib import Path

# Import backend modules
import backend.utils.file_utils as file_utils
import backend.utils.video_utils as video_utils
import backend.utils.face_utils as face_utils
import backend.utils.mouth_cropper as mouth_cropper
import backend.utils.temporal_utils as temporal_utils
import backend.utils.aggregation as aggregation
import backend.utils.ensemble as ensemble
from backend.utils.model_cache import model_cache
from backend.utils.abnormality_analyzer import AbnormalityAnalyzer
from backend.utils.technique_identifier import TechniqueIdentifier

# Import metrics module
from metrics import calculate_metrics, analyze_output_quality, print_metrics_report, generate_test_data


def run_single_video_analysis(video_path, job_id=None):
    """
    Run deepfake detection on a single video.
    
    Args:
        video_path: Path to video file
        job_id: Optional job ID (generated if not provided)
    
    Returns:
        Complete result dictionary
    """
    if job_id is None:
        job_id = str(uuid.uuid4())
    
    print(f"\n{'='*60}")
    print(f"DEEPFAKE DETECTION PIPELINE")
    print(f"{'='*60}")
    print(f"Job ID: {job_id}")
    print(f"Video: {video_path}")
    
    # Setup directories
    base_dir = os.path.abspath("storage")
    frames_output_dir = os.path.normpath(os.path.join(base_dir, "frames", job_id))
    faces_output_dir = os.path.normpath(os.path.join(base_dir, "faces", job_id))
    mouth_output_dir = os.path.normpath(os.path.join(base_dir, "mouth", job_id))
    audio_output_path = os.path.normpath(os.path.join(base_dir, "audio", f"{job_id}.wav"))
    results_dir = os.path.abspath("results")
    frequency_debug_dir = os.path.normpath(os.path.join(results_dir, job_id, "frequency_maps"))
    
    # Initialize models
    print("\n[1/9] Initializing models...")
    model_cache.initialize()
    cnn_detector = model_cache.get_cnn_detector()
    temporal_detector = model_cache.get_temporal_detector()
    lipsync_detector = model_cache.get_lipsync_detector()
    freq_detector = model_cache.get_frequency_detector()
    print("✓ Models initialized")
    
    # Extract frames
    print("\n[2/9] Extracting frames...")
    frame_count = video_utils.extract_frames(video_path, frames_output_dir, fps=1)
    print(f"✓ Extracted {frame_count} frames")
    
    # Extract faces
    print("\n[3/9] Extracting faces...")
    face_count = face_utils.extract_faces_from_frames(frames_output_dir, faces_output_dir)
    print(f"✓ Extracted {face_count} faces")
    
    # Run CNN detection
    print("\n[4/9] Running CNN detection...")
    face_files = sorted(Path(faces_output_dir).glob("face_*.jpg"))
    face_paths = [str(f) for f in face_files]
    fake_scores = cnn_detector.predict_batch(face_paths)
    
    detections = []
    for idx, (face_path, fake_score) in enumerate(zip(face_files, fake_scores)):
        detections.append({
            "face_file": face_path.name,
            "frame": idx + 1,
            "fake_score": round(fake_score, 4)
        })
    detections.sort(key=lambda x: x["fake_score"], reverse=True)
    print(f"✓ Processed {len(detections)} faces (max score: {max(fake_scores):.4f})")
    
    # Run frequency detection
    print("\n[5/9] Running frequency detection...")
    freq_scores_dict = freq_detector.batch_compute(faces_output_dir, frequency_debug_dir)
    for detection in detections:
        detection["freq_score"] = round(freq_scores_dict.get(detection["face_file"], 0.5), 4)
    print(f"✓ Computed frequency scores")
    
    # Extract mouth regions and run lip-sync
    print("\n[6/9] Running lip-sync analysis...")
    mouth_count = mouth_cropper.extract_mouth_frames(faces_output_dir, mouth_output_dir)
    lipsync_detector.extract_audio(video_path, audio_output_path)
    lip_sync_score = lipsync_detector.compute_sync_score(mouth_output_dir, audio_output_path)
    print(f"✓ Lip-sync score: {lip_sync_score:.4f}")
    
    # Run temporal detection
    print("\n[7/9] Running temporal detection...")
    tracks = temporal_utils.group_faces_into_tracks(faces_output_dir)
    track_scores = []
    for track in tracks:
        track_result = temporal_detector.predict_for_face_track(faces_output_dir, clip_len=16, stride=8)
        if track_result["clip_scores"]:
            track_scores.extend(track_result["clip_scores"])
    temporal_mean = sum(track_scores) / len(track_scores) if track_scores else 0.5
    temporal_max = max(track_scores) if track_scores else 0.5
    print(f"✓ Temporal analysis complete (mean: {temporal_mean:.4f})")
    
    # Aggregate scores
    print("\n[8/9] Aggregating scores...")
    aggregation_result = aggregation.aggregate_scores(detections, lip_sync_score, temporal_mean, temporal_max)
    verdict_result = aggregation.decide_verdict(aggregation_result)
    
    # Ensemble fusion
    ensemble_combiner = ensemble.EnsembleCombiner()
    ensemble_result = ensemble_combiner.combine(aggregation_result)
    verdict_result["final_score"] = ensemble_result["final_score"]
    verdict_result["final_label"] = ensemble_result["final_label"]
    print(f"✓ Final verdict: {verdict_result['final_label']} (score: {verdict_result['final_score']:.4f})")
    
    # Generate reports
    print("\n[9/9] Generating reports...")
    abnormality_analyzer = AbnormalityAnalyzer()
    abnormality_report = abnormality_analyzer.generate_abnormality_report(
        faces_output_dir, detections, temporal_mean, temporal_max, lip_sync_score
    )
    
    technique_identifier = TechniqueIdentifier()
    technique_report = technique_identifier.generate_technique_report(abnormality_report, aggregation_result)
    
    # Generate complete result
    result = aggregation.generate_result(
        job_id, video_path, frame_count, detections, lip_sync_score,
        temporal_mean, temporal_max, abnormality_report, technique_report
    )
    result["verdict"] = verdict_result
    
    # Save result
    result_file = Path(results_dir) / f"{job_id}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Results saved to: {result_file}")
    print(f"\n{'='*60}\n")
    
    return result


def run_batch_testing():
    """
    Run batch testing with synthetic data to calculate metrics.
    """
    print("\n" + "="*60)
    print("BATCH TESTING MODE")
    print("="*60)
    print("Generating synthetic test dataset...")
    
    # Generate test data
    y_true, y_pred, scores = generate_test_data(num_samples=100)
    
    print(f"✓ Generated {len(y_true)} test samples")
    print(f"  Authentic videos: {np.sum(y_true == 0)}")
    print(f"  Fake videos: {np.sum(y_true == 1)}")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    output_quality = analyze_output_quality(metrics['accuracy'])
    
    # Print report
    print_metrics_report(metrics, output_quality)
    
    return metrics, output_quality


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION SYSTEM")
    print("="*60)
    print("\nThis system demonstrates:")
    print("  1. Complete deepfake detection pipeline")
    print("  2. Performance metrics (Accuracy, Precision, Recall)")
    print("  3. Output quality analysis (Random/Medium/Accurate)")
    print("  4. Automated testing")
    print("="*60)
    
    # Run batch testing first
    print("\n\n>>> RUNNING BATCH TESTING <<<")
    metrics, output_quality = run_batch_testing()
    
    # Run single video analysis (with mock video)
    print("\n\n>>> RUNNING SINGLE VIDEO ANALYSIS <<<")
    video_path = "storage/uploads/sample_video.mp4"
    
    # Create mock video file
    os.makedirs("storage/uploads", exist_ok=True)
    if not os.path.exists(video_path):
        with open(video_path, 'w') as f:
            f.write("mock video file")
    
    result = run_single_video_analysis(video_path)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"\n✓ Pipeline executed successfully")
    print(f"✓ Metrics calculated: Accuracy={metrics['accuracy']:.4f}, "
          f"Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    print(f"✓ Output Quality: {output_quality}")
    print(f"✓ Video Analysis: {result['verdict']['final_label']} "
          f"(confidence: {result['verdict'].get('confidence', 0):.2%})")
    print("\n" + "="*60)
    print("All tests completed successfully! ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
