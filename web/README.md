# Web Interface Usage Guide

## 🚀 Quick Start

### 1. Install Flask Dependencies
```bash
pip install flask flask-cors
```

### 2. Start the Backend Server
```bash
cd web
python server.py
```

You should see:
```
DEEPFAKE DETECTION WEB SERVER
============================================================
Server starting on http://localhost:5000
Open web/index.html in your browser to use the interface
============================================================
```

### 3. Open the Web Interface
Simply open `web/index.html` in your browser (double-click the file or drag it into your browser).

## 📱 How to Use

1. **Upload Video**: Click the upload box or drag and drop your video
2. **Preview**: Review your video before analysis
3. **Analyze**: Click "Analyze Video" button
4. **View Results**: See if the video is Real or Fake with detailed metrics

## 🎨 Features

- ✅ Beautiful dark theme UI
- ✅ Drag and drop upload
- ✅ Real-time progress tracking
- ✅ Detailed metrics display
- ✅ Responsive design
- ✅ Clean separation of HTML, CSS, JS

## 📁 File Structure

```
web/
├── index.html    # Main HTML page
├── style.css     # All styling
├── script.js     # Frontend logic
└── server.py     # Flask backend
```

## 🔧 Troubleshooting

**"Error analyzing video"**
- Make sure the Flask server is running (`python web/server.py`)
- Check the server terminal for error messages

**CORS errors**
- The server has CORS enabled, but if you still have issues, try using a local server instead of opening the HTML file directly

**Server not starting**
- Install dependencies: `pip install flask flask-cors`
- Check if port 5000 is available

## 💡 Tips

- Supported formats: MP4, AVI, MOV, MKV
- Best results with videos containing clear faces
- Processing takes ~10-30 seconds per video
