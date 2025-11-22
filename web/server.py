"""
Flask backend server for the web interface.
Handles video upload and analysis.
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import uuid
from pathlib import Path
from werkzeug.utils import secure_filename

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_pipeline import run_single_video_analysis

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = 'storage/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/analyze', methods=['POST'])
def analyze_video():
    """Handle video upload and analysis."""
    try:
        # Check if file is present
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        file_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
        file.save(file_path)
        
        print(f"Analyzing video: {file_path}")
        
        # Run analysis
        result = run_single_video_analysis(file_path, job_id)
        
        # Clean up uploaded file (optional)
        # os.remove(file_path)
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION WEB SERVER")
    print("="*60)
    print("Server starting on http://localhost:5000")
    print("Open web/index.html in your browser to use the interface")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)
