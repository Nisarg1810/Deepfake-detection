# DeepSearch - Deepfake Detection System

Complete integration of React frontend with Flask backend for AI-powered deepfake detection.

## 🚀 Quick Start

### 1. Install Backend Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install Frontend Dependencies
```bash
cd Frontend
npm install
```

### 3. Start Backend Server
```bash
# In root directory
cd web
python server.py
```

Backend runs on: `http://localhost:5000`

### 4. Start Frontend Development Server
```bash
# In Frontend directory
cd Frontend
npm run dev
```

Frontend runs on: `http://localhost:5173`

## 📁 Project Structure

```
├── Frontend/              # React + Vite frontend
│   ├── src/
│   │   ├── components/
│   │   │   ├── Hero.jsx
│   │   │   ├── VideoUpload.jsx  # NEW: Upload modal
│   │   │   └── ...
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── web/
│   └── server.py          # Flask backend API
├── backend/               # Detection modules
├── ml_model_xgboost.py   # ML model
└── run_pipeline.py       # Detection pipeline
```

## 🔌 API Integration

### Backend Endpoint
```
POST http://localhost:5000/analyze
Content-Type: multipart/form-data

Body: { video: <file> }
```

### Response Format
```json
{
  "job_id": "uuid",
  "verdict": {
    "final_label": "POSSIBLY_MANIPULATED",
    "final_score": 0.67,
    "confidence": 0.72
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

## 🎨 Features

### Frontend (React + Vite + Tailwind)
- ✅ Modern landing page with animations
- ✅ Video upload modal with drag-and-drop
- ✅ Real-time analysis progress
- ✅ Beautiful results display
- ✅ Responsive design

### Backend (Flask + Python)
- ✅ Multi-detector analysis (CNN, Temporal, Frequency, Lip-Sync)
- ✅ XGBoost ML model
- ✅ REST API with CORS support
- ✅ Video processing pipeline

## 🛠️ Development

### Frontend Development
```bash
cd Frontend
npm run dev        # Start dev server
npm run build      # Build for production
npm run preview    # Preview production build
```

### Backend Development
```bash
python web/server.py              # Start Flask server
python ml_model_xgboost.py        # Test ML model
python test_video.py video.mp4    # Test single video
```

## 📦 Technologies

**Frontend:**
- React 18
- Vite
- Tailwind CSS
- Framer Motion
- Lucide Icons

**Backend:**
- Python 3.8+
- Flask
- XGBoost
- OpenCV
- scikit-learn

## 🔧 Configuration

### CORS Settings
The Flask backend is configured to accept requests from `http://localhost:5173` (Vite dev server).

### API URL
Update in `VideoUpload.jsx` if needed:
```javascript
const response = await fetch('http://localhost:5000/analyze', {
    method: 'POST',
    body: formData,
});
```

## 🚀 Production Deployment

### Build Frontend
```bash
cd Frontend
npm run build
```

### Serve with Flask
Update `web/server.py` to serve the built React app:
```python
from flask import send_from_directory

@app.route('/')
def serve_frontend():
    return send_from_directory('../Frontend/dist', 'index.html')
```

## 📝 Notes

- Old HTML/CSS/JS files removed from `web/` folder
- React frontend now handles all UI
- Flask backend provides REST API only
- Upload modal integrated into Hero component

## 🎯 Usage

1. Start both servers (backend + frontend)
2. Open `http://localhost:5173` in browser
3. Click "Upload Video" button
4. Select or drag-drop video file
5. Click "Analyze Video"
6. View results with confidence scores

Enjoy your integrated deepfake detection system! 🎉
