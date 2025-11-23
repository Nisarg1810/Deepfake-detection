"""
Simple script to test the deepfake detection pipeline with your own video.
Usage: python test_video.py path/to/your/video.mp4
"""
import sys
import os
from pathlib import Path
import uuid
import json

# Import the pipeline
from run_pipeline import run_single_video_analysis
from metrics import calculate_metrics, analyze_output_quality, print_metrics_report


def test_single_video(video_path):
    """
    Test a single video file.
    
    Args:
        video_path: Path to the video file
    """
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        print("\nPlease provide a valid video path.")
        return
    
    # Get file info
    file_size = os.path.getsize(video_path) / (1024 * 1024)  # MB
    print("\n" + "="*60)
    print("VIDEO UPLOAD TEST")
    print("="*60)
    print(f"Video file: {video_path}")
    print(f"File size: {file_size:.2f} MB")
    print("="*60 + "\n")
    
    # Run analysis
    print("Starting analysis...\n")
    result = run_single_video_analysis(video_path)
    
    # Display detailed results
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    print(f"\n📊 Detection Scores:")
    print(f"  CNN Max Score: {result['aggregation']['max_score']:.4f}")
    print(f"  CNN Mean Score: {result['aggregation']['mean_score']:.4f}")
    print(f"  Frequency Score: {result['aggregation']['frequency_score']:.4f}")
    print(f"  Temporal Score: {result['aggregation']['temporal_max']:.4f}")
    if result['aggregation'].get('lip_sync_score'):
        print(f"  Lip-Sync Score: {result['aggregation']['lip_sync_score']:.4f}")
    
    print(f"\n🎯 Final Verdict:")
    print(f"  Label: {result['verdict']['final_label']}")
    print(f"  Score: {result['verdict']['final_score']:.4f}")
    print(f"  Confidence: {result['verdict'].get('confidence', 0):.2%}")
    
    print(f"\n📁 Results saved to:")
    print(f"  results/{result['job_id']}.json")
    
    print("\n" + "="*60 + "\n")
    
    return result


def main():
    """Main function."""
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION - VIDEO UPLOAD TEST")
    print("="*60)
    
    # Check if video path provided
    if len(sys.argv) < 2:
        print("\n📹 Usage: python test_video.py <path_to_video>")
        print("\nExamples:")
        print("  python test_video.py my_video.mp4")
        print("  python test_video.py C:/Videos/sample.mp4")
        print("  python test_video.py storage/uploads/test.mp4")
        
        # Check if there are any videos in storage/uploads
        uploads_dir = Path("storage/uploads")
        if uploads_dir.exists():
            video_files = list(uploads_dir.glob("*.mp4"))
            if video_files:
                print(f"\n📂 Found {len(video_files)} video(s) in storage/uploads:")
                for i, vf in enumerate(video_files[:5], 1):
                    print(f"  {i}. {vf.name}")
                
                print("\n💡 To test one of these, run:")
                print(f"  python test_video.py storage/uploads/{video_files[0].name}")
        
        print("\n" + "="*60 + "\n")
        return
    
    # Get video path from command line
    video_path = sys.argv[1]
    
    # Run test
    test_single_video(video_path)


if __name__ == "__main__":
    main()
