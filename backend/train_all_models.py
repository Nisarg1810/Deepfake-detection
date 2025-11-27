"""
Train All Models for Deepfake Detection
========================================
Trains multiple ML models and compares their performance.

Models trained:
1. XGBoost
2. Random Forest
3. Gradient Boosting
4. Support Vector Machine (SVM)
5. Logistic Regression
6. K-Nearest Neighbors (KNN)
7. Neural Network (MLP)

Usage: python train_all_models.py
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import json
from datetime import datetime

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)

# Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and compare multiple ML models."""
    
    def __init__(self, features_path):
        self.features_path = Path(features_path)
        self.models_dir = self.features_path.parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_data(self):
        """Load features from CSV."""
        print("[LOAD] Loading features...")
        
        if not self.features_path.exists():
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        
        df = pd.read_csv(self.features_path)
        
        X = df.drop('label', axis=1)
        y = df['label']
        
        self.feature_names = X.columns.tolist()
        
        print(f"[OK] Loaded {len(df)} samples with {len(self.feature_names)} features")
        print(f"     Real: {sum(y == 0)}, Fake: {sum(y == 1)}")
        
        return X, y
    
    def prepare_data(self, X, y, test_size=0.2):
        """Split and scale data."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save scaler
        joblib.dump(self.scaler, self.models_dir / "scaler.pkl")
        
        print(f"[OK] Data split: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a trained model."""
        y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = 0.0
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'auc_roc': auc
        }
        
        return metrics
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model."""
        print("\n[1/7] Training XGBoost...")
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test, 'XGBoost')
        
        # Save model
        model.save_model(str(self.models_dir / "xgboost_model.json"))
        
        self.results['XGBoost'] = metrics
        print(f"      Accuracy: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1_score']*100:.2f}%")
        
        return model
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model."""
        print("\n[2/7] Training Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test, 'Random Forest')
        
        joblib.dump(model, self.models_dir / "random_forest_model.pkl")
        
        self.results['Random Forest'] = metrics
        print(f"      Accuracy: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1_score']*100:.2f}%")
        
        return model
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Train Gradient Boosting model."""
        print("\n[3/7] Training Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test, 'Gradient Boosting')
        
        joblib.dump(model, self.models_dir / "gradient_boosting_model.pkl")
        
        self.results['Gradient Boosting'] = metrics
        print(f"      Accuracy: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1_score']*100:.2f}%")
        
        return model
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """Train SVM model."""
        print("\n[4/7] Training SVM...")
        
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test, 'SVM')
        
        joblib.dump(model, self.models_dir / "svm_model.pkl")
        
        self.results['SVM'] = metrics
        print(f"      Accuracy: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1_score']*100:.2f}%")
        
        return model
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression model."""
        print("\n[5/7] Training Logistic Regression...")
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test, 'Logistic Regression')
        
        joblib.dump(model, self.models_dir / "logistic_regression_model.pkl")
        
        self.results['Logistic Regression'] = metrics
        print(f"      Accuracy: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1_score']*100:.2f}%")
        
        return model
    
    def train_knn(self, X_train, y_train, X_test, y_test):
        """Train KNN model."""
        print("\n[6/7] Training K-Nearest Neighbors...")
        
        model = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test, 'KNN')
        
        joblib.dump(model, self.models_dir / "knn_model.pkl")
        
        self.results['KNN'] = metrics
        print(f"      Accuracy: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1_score']*100:.2f}%")
        
        return model
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train Neural Network (MLP) model."""
        print("\n[7/7] Training Neural Network (MLP)...")
        
        model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        model.fit(X_train, y_train)
        metrics = self.evaluate_model(model, X_test, y_test, 'Neural Network')
        
        joblib.dump(model, self.models_dir / "neural_network_model.pkl")
        
        self.results['Neural Network'] = metrics
        print(f"      Accuracy: {metrics['accuracy']*100:.2f}%, F1: {metrics['f1_score']*100:.2f}%")
        
        return model
    
    def print_comparison(self):
        """Print comparison of all models."""
        print("\n" + "="*80)
        print("MODEL COMPARISON RESULTS")
        print("="*80)
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC-ROC':<12}")
        print("-"*80)
        
        # Sort by F1 score
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        for model_name, metrics in sorted_results:
            print(f"{model_name:<25} {metrics['accuracy']*100:>10.2f}% {metrics['precision']*100:>10.2f}% "
                  f"{metrics['recall']*100:>10.2f}% {metrics['f1_score']*100:>10.2f}% {metrics['auc_roc']*100:>10.2f}%")
        
        print("="*80)
        
        # Best model
        best_model = sorted_results[0][0]
        best_f1 = sorted_results[0][1]['f1_score']
        print(f"\n[BEST] {best_model} with F1-Score: {best_f1*100:.2f}%")
        
        return best_model
    
    def save_results(self):
        """Save results to JSON."""
        results_path = self.models_dir / "training_results.json"
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'features_file': str(self.features_path),
            'feature_names': self.feature_names,
            'results': self.results
        }
        
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"[OK] Results saved to: {results_path}")
    
    def train_all(self):
        """Train all models."""
        # Load data
        X, y = self.load_data()
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        # Train each model
        self.train_xgboost(X_train, y_train, X_test, y_test)
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        self.train_svm(X_train, y_train, X_test, y_test)
        self.train_logistic_regression(X_train, y_train, X_test, y_test)
        self.train_knn(X_train, y_train, X_test, y_test)
        self.train_neural_network(X_train, y_train, X_test, y_test)
        
        # Compare and save
        best_model = self.print_comparison()
        self.save_results()
        
        print("\n" + "="*60)
        print("[DONE] ALL MODELS TRAINED AND SAVED!")
        print("="*60)
        print(f"Models saved in: {self.models_dir}")
        print("="*60)
        
        return best_model


def main():
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION - TRAIN ALL MODELS")
    print("="*60)
    
    # Features file path
    features_path = Path(__file__).parent / "dataset_features.csv"
    
    # Check if features exist
    if not features_path.exists():
        # Try extracted_features.csv (from previous run)
        alt_path = Path(__file__).parent / "extracted_features.csv"
        if alt_path.exists():
            features_path = alt_path
        else:
            print("[ERROR] No features file found!")
            print("Run extract_features.py first to extract features from videos.")
            return
    
    # Train all models
    trainer = ModelTrainer(features_path)
    trainer.train_all()


if __name__ == "__main__":
    main()

