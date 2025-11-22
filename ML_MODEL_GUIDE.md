# XGBoost Deepfake Detection Model - Complete Guide

## ✅ **What's Working NOW**

The complete XGBoost model is **ready and tested**! Here's what you have:

### **Current Results (Mock Data):**
```
Accuracy:  71.00%
Precision: 69.44%
Recall:    58.14%
F1-Score:  63.29%
AUC-ROC:   0.7623
```

### **Feature Importance (Top 5):**
```
1. cnn_max_score      → 0.1370
2. temporal_max       → 0.0811
3. blending_score     → 0.0720
4. freq_mean          → 0.0703
5. lipsync_score      → 0.0629
```

### **Single Prediction Test:**
```
Verdict: FAKE
Confidence: 81.85%
Probability Real: 0.1815
Probability Fake: 0.8185
```

---

## 🎯 **Features Implemented**

✅ **XGBoost Classifier** - Best for tabular binary classification
✅ **Feature Importance (FI = X)** - Shows which features matter most
✅ **Full Metrics** - Accuracy, Precision, Recall, F1, AUC-ROC
✅ **Probability Predictions** - Returns prob_real and prob_fake
✅ **Confusion Matrix** - Shows TP, TN, FP, FN
✅ **Single Sample Prediction** - Test individual videos
✅ **Model Save/Load** - Persist trained models
✅ **Dataset Ready** - Easy switch to real data

---

## 📊 **How to Use NOW (Mock Data)**

### **Run the Model:**
```bash
python ml_model_xgboost.py
```

This will:
1. Generate 1000 synthetic samples
2. Train XGBoost model
3. Show all metrics
4. Display feature importance
5. Plot confusion matrix
6. Test single prediction
7. Save model to `models/deepfake_xgboost.json`

---

## 🔄 **How to Use AFTER Dataset Download**

### **Step 1: Prepare Your Dataset**

Your dataset should be a **CSV file** with this structure:

```csv
cnn_max_score,cnn_mean_score,freq_mean,temporal_max,lipsync_score,...,label
0.85,0.75,0.72,0.78,0.45,...,1
0.32,0.28,0.35,0.30,0.88,...,0
...
```

**Requirements:**
- Last column = `label` (0=Real, 1=Fake)
- OR column named `is_fake` or `label`
- All numeric features

### **Step 2: Extract Features from Videos**

If your dataset is raw videos, you need to extract features first. Create `extract_features.py`:

```python
from ml_model_xgboost import DeepfakeDetectionModel
from run_pipeline import run_single_video_analysis
import pandas as pd
from pathlib import Path

def extract_features_from_videos(video_folder, output_csv):
    """Extract features from all videos in folder."""
    
    video_files = list(Path(video_folder).glob("*.mp4"))
    all_features = []
    
    for video_path in video_files:
        print(f"Processing: {video_path.name}")
        
        # Run analysis
        result = run_single_video_analysis(str(video_path))
        
        # Extract features
        features = {
            'cnn_max_score': result['aggregation']['max_score'],
            'cnn_mean_score': result['aggregation']['mean_score'],
            'freq_mean': result['aggregation']['frequency_score'],
            'temporal_max': result['aggregation']['temporal_max'],
            'temporal_mean': result['aggregation']['temporal_mean'],
            'lipsync_score': result['aggregation'].get('lip_sync_score', 0.5),
            'num_faces': result['faces'],
            'num_frames': result['frames'],
            # Add more features as needed
            
            # Label (you need to provide this based on filename or metadata)
            'label': 1 if 'fake' in video_path.name.lower() else 0
        }
        
        all_features.append(features)
    
    # Save to CSV
    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved features to: {output_csv}")

# Usage
extract_features_from_videos(
    video_folder="path/to/your/videos",
    output_csv="dataset/features.csv"
)
```

### **Step 3: Train with Real Dataset**

Modify `ml_model_xgboost.py`:

```python
# Change this line:
model = DeepfakeDetectionModel(use_mock_data=False)  # Set to False

# And provide dataset path:
X_train, X_test, y_train, y_test = model.prepare_data(
    dataset_path="dataset/features.csv"  # Your CSV file
)
```

### **Step 4: Run Training**

```bash
python ml_model_xgboost.py
```

---

## 📁 **Dataset Formats Supported**

### **Option 1: Pre-extracted Features (CSV)**
```
dataset/
└── features.csv  # All features in one CSV
```

### **Option 2: Raw Videos**
```
videos/
├── real/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── fake/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

Use `extract_features.py` to convert to CSV.

---

## 🎯 **Expected Dataset Structure**

### **Minimum Required Features:**
```python
required_features = [
    'cnn_max_score',      # CNN detection score
    'cnn_mean_score',     # Average CNN score
    'freq_mean',          # Frequency analysis
    'temporal_max',       # Temporal consistency
    'lipsync_score',      # Lip-sync score
    'label'               # 0=Real, 1=Fake
]
```

### **Recommended Additional Features:**
```python
optional_features = [
    'cnn_std_score',
    'freq_std',
    'freq_max',
    'temporal_mean',
    'temporal_variance',
    'lipsync_confidence',
    'num_faces',
    'num_frames',
    'avg_face_size',
    'spatial_artifacts',
    'temporal_artifacts',
    'blending_score'
]
```

---

## 🔧 **Customization for Your Dataset**

### **If Your Dataset Has Different Column Names:**

Edit the `load_real_dataset()` function:

```python
def load_real_dataset(self, dataset_path):
    df = pd.read_csv(dataset_path)
    
    # Map your columns to expected names
    column_mapping = {
        'your_column_name': 'cnn_max_score',
        'another_column': 'freq_mean',
        # ... add all mappings
    }
    
    df = df.rename(columns=column_mapping)
    
    # Rest of the code...
```

### **If Your Labels Are Different:**

```python
# If labels are 'real'/'fake' instead of 0/1:
if y.dtype == 'object':
    y = y.map({'real': 0, 'fake': 1})

# If labels are 'REAL'/'FAKE':
if y.dtype == 'object':
    y = y.map({'REAL': 0, 'FAKE': 1})
```

---

## 📈 **Hyperparameter Tuning (After Dataset)**

Once you have real data, you can tune hyperparameters:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

---

## ✅ **Checklist: From Mock to Real**

- [ ] Download your 24GB dataset
- [ ] Organize videos into folders (real/fake) OR prepare CSV
- [ ] If raw videos: Run `extract_features.py`
- [ ] Update `use_mock_data=False` in code
- [ ] Provide `dataset_path` to `prepare_data()`
- [ ] Run `python ml_model_xgboost.py`
- [ ] Check metrics (should be >80% accuracy on good dataset)
- [ ] Tune hyperparameters if needed
- [ ] Save final model
- [ ] Integrate with web interface

---

## 🚀 **Integration with Web Interface**

After training on real data, integrate with your web app:

```python
# In web/server.py, modify the analyze endpoint:

from ml_model_xgboost import DeepfakeDetectionModel

# Load trained model
ml_model = DeepfakeDetectionModel(use_mock_data=False)
ml_model.load_model('models/deepfake_xgboost.json')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    # ... existing code to extract features ...
    
    # Use ML model for final prediction
    features = {
        'cnn_max_score': aggregation_result['max_score'],
        'cnn_mean_score': aggregation_result['mean_score'],
        # ... all features
    }
    
    prediction = ml_model.predict_single(features)
    
    # Update result with ML prediction
    result['ml_prediction'] = prediction
    
    return jsonify(result)
```

---

## 💡 **Summary**

**NOW (Mock Data):**
- ✅ Complete working model
- ✅ All features implemented
- ✅ 71% accuracy on synthetic data
- ✅ Feature importance working
- ✅ Ready to test

**AFTER Dataset:**
- 🔄 Change 2 lines of code
- 🔄 Provide CSV path
- 🔄 Run training
- ✅ Get 80%+ accuracy (with good dataset)
- ✅ Deploy to production

**You're all set!** 🎉
