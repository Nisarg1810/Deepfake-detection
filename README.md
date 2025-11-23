# Deepfake Detection System

A comprehensive AI-powered deepfake detection system with React frontend, Flask backend, XGBoost ML model, and complete analysis pipeline.

---

## 🎯 Features

- ✅ **React + Vite Frontend** - Modern UI with drag-and-drop upload
- ✅ **Flask REST API** - Backend server for video processing
- ✅ **Multi-Detector Analysis** - CNN, Temporal, Frequency, Lip-Sync
- ✅ **XGBoost ML Model** - Feature importance & metrics
- ✅ **Complete Metrics** - Accuracy, Precision, Recall, F1, AUC-ROC
- ✅ **Real-time Progress** - Live analysis tracking
- ✅ **Beautiful UI** - Dark theme with animations

---

## 🚀 Quick Start

### **Option 1: Use Startup Script (Easiest)**
```bash
START_WEB_INTERFACE.bat
```

### **Option 2: Manual Start**

**1. Install Python Dependencies**
```bash
pip install -r requirements.txt
```

**2. Install Frontend Dependencies**
```bash
cd Frontend
npm install
```

**3. Start Backend Server**
```bash
cd web
python server.py
```
Backend runs on: `http://localhost:5000`

**4. Start Frontend**
```bash
cd Frontend
npm run dev
```
Frontend runs on: `http://localhost:5173`

**5. Open Browser**
Go to `http://localhost:5173` and upload a video!

---

## 📁 Project Structure

```
├── Frontend/              # React + Vite app
│   ├── src/components/   # UI components
│   │   ├── VideoUpload.jsx  # Upload modal
│   │   └── Hero.jsx      # Landing page
│   └── package.json
│
├── backend/              # Detection modules
│   └── utils/           # Mock detectors
│
├── web/                 # Flask backend
│   ├── server.py       # API server
│   ├── storage/        # Data storage
│   └── results/        # Analysis results
│
├── models/              # ML models
│   └── deepfake_xgboost.json
│
├── tests/               # Unit tests
│
├── ml_model_xgboost.py # XGBoost model
├── run_pipeline.py     # Detection pipeline
├── test_video.py       # Video tester
└── metrics.py          # Metrics
```

---

## 📊 Usage

### **Web Interface**
1. Click "Upload Video" button
2. Select or drag-drop video file
3. Click "Analyze Video"
4. View results:
   - Verdict (Real/Fake)
   - Confidence score
   - 4 detector metrics
   - Frame/face counts

### **Command Line**
```bash
# Test single video
python test_video.py "video.mp4"

# Run pipeline
python run_pipeline.py

# Test ML model
python ml_model_xgboost.py
```

---

## 🎯 ML Model

### **Performance (Mock Data)**
- Accuracy: 71.00%
- Precision: 69.44%
- Recall: 58.14%
- F1-Score: 63.29%
- AUC-ROC: 0.7623

### **Top Features**
1. CNN Max Score (13.7%)
2. Temporal Max (8.1%)
3. Blending Score (7.2%)
4. Frequency Mean (7.0%)
5. Lip-Sync Score (6.3%)

### **Training with Real Dataset**
```python
from ml_model_xgboost import DeepfakeDetectionModel

# Initialize
model = DeepfakeDetectionModel(use_mock_data=False)

# Train
X_train, X_test, y_train, y_test = model.prepare_data(
    dataset_path="dataset.csv"
)
model.train(X_train, y_train)

# Evaluate
metrics = model.evaluate(X_test, y_test)
```

---

## 🔧 Technologies

**Frontend:**
- React 18
- Vite
- Tailwind CSS
- Framer Motion
- Lucide Icons

**Backend:**
- Python 3.8+
- Flask + CORS
- XGBoost
- scikit-learn
- OpenCV
- Matplotlib

---

## 🧪 Testing

```bash
# Run all tests
python -m unittest discover tests

# Test specific
python tests/test_pipeline.py
```

**14 unit tests** covering metrics, detectors, aggregation, and ensemble.

---

## 📡 API

### **Endpoint**
```
POST http://localhost:5000/analyze
Content-Type: multipart/form-data
Body: { video: <file> }
```

### **Response**
```json
{
  "verdict": {
    "final_label": "POSSIBLY_MANIPULATED",
    "final_score": 0.67
  },
  "aggregation": {
    "max_score": 0.86,
    "frequency_score": 0.51,
    "temporal_max": 0.69,
    "lip_sync_score": 0.57
  },
  "frames": 10,
  "faces": 10
}
```

---

## 🎨 Features Breakdown

### **Detection Methods**
1. **CNN Detection** - Face manipulation artifacts
2. **Frequency Analysis** - FFT-based patterns
3. **Temporal Analysis** - Frame consistency
4. **Lip-Sync Analysis** - Audio-video sync

### **Web Interface**
- Drag-and-drop upload
- Real-time progress
- Beautiful results display
- Responsive design
- Dark theme

### **ML Model**
- XGBoost classifier
- Feature importance
- Full metrics
- Model persistence
- Dataset ready

---

## 📦 Dependencies

**Python:**
```
flask
flask-cors
xgboost
scikit-learn
opencv-python
matplotlib
seaborn
numpy
pandas
pillow
```

**Node.js:**
```
react
vite
tailwindcss
framer-motion
lucide-react
```

---

## 🔄 Workflow

```
User uploads video
    ↓
React Frontend
    ↓
Flask API (/analyze)
    ↓
Detection Pipeline
    ↓
4 Detectors (CNN, Freq, Temporal, Lip-Sync)
    ↓
Ensemble Fusion
    ↓
Final Verdict
    ↓
JSON Response
    ↓
React displays results
```

---

## 💡 Tips

- **For testing**: Use the included Morgan Freeman video
- **For development**: Both servers auto-reload on changes
- **For production**: Build frontend with `npm run build`
- **For real detection**: Replace mock backend with trained models

---

## 🐛 Troubleshooting

**"Failed to fetch" error:**
- Make sure Flask server is running on port 5000
- Check CORS is enabled

**"Analysis failed":**
- Restart Flask server
- Check video file format (MP4, AVI, MOV)

**Frontend won't start:**
- Run `npm install` in Frontend folder
- Check Node.js version (14+)

---

## 📄 License

MIT License - Open source

## 👥 Authors

- Rudram Jhaveri
- Nisarg (Collaborator)

## 🙏 Acknowledgments

Built with modern web technologies and state-of-the-art ML algorithms for deepfake detection research.

---

**🎉 Ready to detect deepfakes! Start with `START_WEB_INTERFACE.bat`**
