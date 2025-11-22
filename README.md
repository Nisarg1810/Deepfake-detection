# Deepfake Detection System

A comprehensive AI-powered deepfake detection system with web interface, machine learning models, and complete analysis pipeline.

## 🎯 Features

- **Multi-Detector Analysis**: CNN, Temporal, Frequency, and Lip-Sync detection
- **XGBoost ML Model**: Production-ready classifier with feature importance
- **Web Interface**: Beautiful drag-and-drop video upload interface
- **Complete Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Real-time Analysis**: Progress tracking and detailed results
- **Model Persistence**: Save and load trained models

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Web Interface
```bash
# Start the backend server
cd web
python server.py

# Open web/index.html in your browser
```

### 3. Test ML Model
```bash
python ml_model_xgboost.py
```

## 📁 Project Structure

```
├── backend/              # Detection modules
│   └── utils/           # Utility functions
├── web/                 # Web interface
│   ├── index.html      # Main page
│   ├── style.css       # Styling
│   ├── script.js       # Frontend logic
│   └── server.py       # Flask backend
├── tests/              # Unit tests
├── models/             # Trained models
├── storage/            # Data storage
├── results/           # Analysis results
├── ml_model_xgboost.py    # XGBoost model
├── run_pipeline.py        # Main pipeline
├── test_video.py          # Video tester
└── metrics.py             # Metrics calculation
```

## 🎯 ML Model Performance

**Current Results (Mock Data):**
- Accuracy: 71.00%
- Precision: 69.44%
- Recall: 58.14%
- F1-Score: 63.29%
- AUC-ROC: 0.7623

**Top Features:**
1. CNN Max Score (13.7%)
2. Temporal Max (8.1%)
3. Blending Score (7.2%)
4. Frequency Mean (7.0%)
5. Lip-Sync Score (6.3%)

## 📊 Usage

### Web Interface
1. Start server: `python web/server.py`
2. Open `web/index.html` in browser
3. Upload video and click "Analyze"
4. View results with confidence scores

### Command Line
```bash
# Test a single video
python test_video.py path/to/video.mp4

# Run complete pipeline
python run_pipeline.py
```

### ML Model Training
```python
from ml_model_xgboost import DeepfakeDetectionModel

# Initialize model
model = DeepfakeDetectionModel(use_mock_data=False)

# Train with your dataset
X_train, X_test, y_train, y_test = model.prepare_data(
    dataset_path="dataset.csv"
)
model.train(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)

# Predict
result = model.predict_single(features_dict)
```

## 🔧 Technologies Used

- **Backend**: Python, Flask
- **ML**: XGBoost, scikit-learn
- **Frontend**: HTML, CSS, JavaScript
- **Video Processing**: OpenCV, PIL
- **Visualization**: Matplotlib, Seaborn

## 📈 Metrics

The system provides comprehensive metrics:
- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, Confusion Matrix
- Feature Importance

## 🎨 Web Interface Features

- Modern dark theme with animations
- Drag-and-drop video upload
- Real-time progress tracking
- Detailed analysis results
- 4 detection metrics display
- Responsive design

## 📝 Documentation

- [ML Model Guide](ML_MODEL_GUIDE.md) - Complete ML model documentation
- [Web Interface Guide](web/README.md) - Web interface usage

## 🧪 Testing

```bash
# Run unit tests
python -m unittest discover tests

# Run specific test
python tests/test_pipeline.py
```

## 🔄 Dataset Integration

To use with real dataset:

1. Prepare CSV with features and labels
2. Update `ml_model_xgboost.py`:
   ```python
   model = DeepfakeDetectionModel(use_mock_data=False)
   X_train, X_test, y_train, y_test = model.prepare_data(
       dataset_path="your_dataset.csv"
   )
   ```
3. Run training

## 📦 Requirements

- Python 3.8+
- Flask, XGBoost, scikit-learn
- OpenCV, Matplotlib, Seaborn
- NumPy, Pandas

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👥 Authors

- Rudram Jhaveri

## 🙏 Acknowledgments

- Built with modern web technologies
- Uses state-of-the-art ML algorithms
- Inspired by deepfake detection research
