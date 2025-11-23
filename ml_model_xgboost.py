"""
Complete XGBoost Deepfake Detection Model
==========================================

This model works in TWO MODES:
1. MOCK MODE (current): Uses synthetic data for testing
2. REAL MODE (after dataset): Uses your actual dataset

Features:
- XGBoost classifier
- Feature Importance (FI = X)
- Full metrics (Accuracy, Precision, Recall, F1)
- Probability predictions
- Confusion matrix
- Easy dataset integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import xgboost as xgb

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


class DeepfakeDetectionModel:
    """Complete XGBoost-based deepfake detection model."""
    
    def __init__(self, use_mock_data=True):
        """
        Initialize the model.
        
        Args:
            use_mock_data: If True, uses synthetic data. If False, loads real dataset.
        """
        self.use_mock_data = use_mock_data
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_importance = None
        
    def generate_mock_features(self, n_samples=1000):
        """
        Generate synthetic features for testing.
        This simulates the features you'd extract from real videos.
        """
        np.random.seed(42)
        
        # Create features similar to what we'd extract from videos
        features = {
            # CNN-based features
            'cnn_max_score': np.random.uniform(0.2, 0.9, n_samples),
            'cnn_mean_score': np.random.uniform(0.2, 0.8, n_samples),
            'cnn_std_score': np.random.uniform(0.05, 0.3, n_samples),
            
            # Frequency analysis features
            'freq_mean': np.random.uniform(0.3, 0.85, n_samples),
            'freq_std': np.random.uniform(0.1, 0.4, n_samples),
            'freq_max': np.random.uniform(0.4, 0.95, n_samples),
            
            # Temporal features
            'temporal_mean': np.random.uniform(0.25, 0.75, n_samples),
            'temporal_max': np.random.uniform(0.3, 0.9, n_samples),
            'temporal_variance': np.random.uniform(0.05, 0.35, n_samples),
            
            # Lip-sync features
            'lipsync_score': np.random.uniform(0.4, 0.95, n_samples),
            'lipsync_confidence': np.random.uniform(0.5, 1.0, n_samples),
            
            # Video metadata features
            'num_faces': np.random.randint(1, 5, n_samples),
            'num_frames': np.random.randint(50, 500, n_samples),
            'avg_face_size': np.random.uniform(0.1, 0.5, n_samples),
            
            # Abnormality features
            'spatial_artifacts': np.random.randint(0, 10, n_samples),
            'temporal_artifacts': np.random.randint(0, 8, n_samples),
            'blending_score': np.random.uniform(0.2, 0.9, n_samples),
        }
        
        df = pd.DataFrame(features)
        
        # Generate labels (0=Real, 1=Fake)
        # Make it realistic: higher scores = more likely fake
        fake_probability = (
            df['cnn_max_score'] * 0.3 +
            df['freq_mean'] * 0.2 +
            df['temporal_max'] * 0.2 +
            (1 - df['lipsync_score']) * 0.15 +
            df['blending_score'] * 0.15
        )
        
        # Add some noise
        fake_probability += np.random.normal(0, 0.1, n_samples)
        fake_probability = np.clip(fake_probability, 0, 1)
        
        # Convert to binary labels
        labels = (fake_probability > 0.55).astype(int)
        
        return df, labels
    
    def load_real_dataset(self, dataset_path):
        """
        Load real dataset from CSV or extracted features.
        
        Args:
            dataset_path: Path to dataset CSV file
        
        Returns:
            X (features), y (labels)
        """
        print(f"Loading dataset from: {dataset_path}")
        
        # Load CSV
        df = pd.read_csv(dataset_path)
        
        # Assuming last column is the label
        # Adjust this based on your actual dataset structure
        if 'label' in df.columns:
            y = df['label']
            X = df.drop('label', axis=1)
        elif 'is_fake' in df.columns:
            y = df['is_fake']
            X = df.drop('is_fake', axis=1)
        else:
            # Assume last column is label
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]
        
        # Convert labels to 0/1 if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        print(f"✓ Loaded {len(X)} samples with {X.shape[1]} features")
        
        return X, y
    
    def prepare_data(self, dataset_path=None):
        """
        Prepare data for training.
        
        Args:
            dataset_path: Path to real dataset (if use_mock_data=False)
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        if self.use_mock_data:
            print("📊 Generating mock data for testing...")
            X, y = self.generate_mock_features(n_samples=1000)
        else:
            if dataset_path is None:
                raise ValueError("dataset_path required when use_mock_data=False")
            X, y = self.load_real_dataset(dataset_path)
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for feature names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names)
        
        print(f"✓ Data prepared:")
        print(f"  Training samples: {len(X_train)} (Real: {sum(y_train==0)}, Fake: {sum(y_train==1)})")
        print(f"  Test samples: {len(X_test)} (Real: {sum(y_test==0)}, Fake: {sum(y_test==1)})")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model with optimal hyperparameters.
        """
        print("\n🚀 Training XGBoost model...")
        
        # Best hyperparameters for binary classification
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42,
            'eval_metric': 'logloss',
            'early_stopping_rounds': 20,
            'verbosity': 0
        }
        
        # Create model
        self.model = xgb.XGBClassifier(**params)
        
        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        print("✓ Model trained successfully!")
        
        # Extract feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return self.model
    
    def predict(self, X):
        """
        Predict labels (0=Real, 1=Fake).
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probabilities.
        Returns: [prob_real, prob_fake]
        """
        return self.model.predict_proba(X)
    
    def predict_single(self, features_dict):
        """
        Predict for a single sample.
        
        Args:
            features_dict: Dictionary of features
        
        Returns:
            label, probability, verdict
        """
        # Convert to DataFrame
        df = pd.DataFrame([features_dict])
        
        # Ensure all features are present
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0
        
        df = df[self.feature_names]  # Reorder columns
        
        # Scale
        df_scaled = self.scaler.transform(df)
        
        # Predict
        label = self.model.predict(df_scaled)[0]
        proba = self.model.predict_proba(df_scaled)[0]
        
        verdict = "FAKE" if label == 1 else "REAL"
        confidence = proba[label] * 100
        
        return {
            'label': int(label),
            'verdict': verdict,
            'confidence': confidence,
            'prob_real': proba[0],
            'prob_fake': proba[1]
        }
    
    def evaluate(self, X_test, y_test):
        """
        Comprehensive evaluation with all metrics.
        """
        print("\n📊 EVALUATION RESULTS")
        print("="*60)
        
        # Predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
        print(f"AUC-ROC:   {auc:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives (Real→Real):   {cm[0,0]}")
        print(f"  False Positives (Real→Fake):  {cm[0,1]}")
        print(f"  False Negatives (Fake→Real):  {cm[1,0]}")
        print(f"  True Positives (Fake→Fake):   {cm[1,1]}")
        
        print("="*60)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc,
            'confusion_matrix': cm
        }
    
    def plot_feature_importance(self, top_n=15):
        """
        Plot feature importance.
        """
        plt.figure(figsize=(10, 6))
        
        top_features = self.feature_importance.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Feature Importance (FI = X)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("\n📈 FEATURE IMPORTANCE (FI = X)")
        print("="*60)
        for idx, row in top_features.iterrows():
            print(f"{row['feature']:30s} → {row['importance']:.4f}")
        print("="*60)
    
    def plot_confusion_matrix(self, cm):
        """
        Plot confusion matrix heatmap with percentages.
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both count and percentage
        annot = np.empty_like(cm).astype(str)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                    xticklabels=['Real', 'Fake'],
                    yticklabels=['Real', 'Fake'],
                    cbar_kws={'label': 'Count'})
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Confusion matrix saved to: models/confusion_matrix.png")
    
    def plot_roc_curve(self, y_test, y_proba):
        """
        Plot ROC curve with AUC score.
        """
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc:.4f})')
        
        # Plot diagonal (random classifier)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curve - Receiver Operating Characteristic', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('models/roc_curve.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ ROC curve saved to: models/roc_curve.png")
    
    def plot_performance_comparison(self, metrics):
        """
        Compare model performance with ideal performance.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Metrics comparison
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
        model_scores = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score'],
            metrics['auc_roc']
        ]
        ideal_scores = [1.0, 1.0, 1.0, 1.0, 1.0]  # Perfect scores
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        axes[0].bar(x - width/2, model_scores, width, label='Your Model', 
                   color='steelblue', alpha=0.8)
        axes[0].bar(x + width/2, ideal_scores, width, label='Ideal Model', 
                   color='lightcoral', alpha=0.8)
        
        axes[0].set_ylabel('Score', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Performance vs Ideal', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metric_names, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1.1])
        
        # Add value labels on bars
        for i, (model, ideal) in enumerate(zip(model_scores, ideal_scores)):
            axes[0].text(i - width/2, model + 0.02, f'{model:.3f}', 
                        ha='center', va='bottom', fontsize=9)
        
        # Performance gap analysis
        gaps = [ideal - model for ideal, model in zip(ideal_scores, model_scores)]
        colors = ['green' if gap < 0.1 else 'orange' if gap < 0.3 else 'red' 
                 for gap in gaps]
        
        axes[1].barh(metric_names, gaps, color=colors, alpha=0.7)
        axes[1].set_xlabel('Gap from Ideal', fontsize=12, fontweight='bold')
        axes[1].set_title('Performance Gap Analysis', fontsize=14, fontweight='bold')
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, gap in enumerate(gaps):
            axes[1].text(gap + 0.01, i, f'{gap:.3f}', 
                        va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('models/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Performance comparison saved to: models/performance_comparison.png")
    
    def plot_metrics_heatmap(self, metrics):
        """
        Create a comprehensive metrics heatmap.
        """
        # Create metrics matrix
        metrics_data = {
            'Accuracy': [metrics['accuracy'], 1.0],
            'Precision': [metrics['precision'], 1.0],
            'Recall': [metrics['recall'], 1.0],
            'F1-Score': [metrics['f1_score'], 1.0],
            'AUC-ROC': [metrics['auc_roc'], 1.0]
        }
        
        df = pd.DataFrame(metrics_data, index=['Your Model', 'Ideal Model'])
        
        plt.figure(figsize=(12, 4))
        sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlGn', 
                   vmin=0, vmax=1, cbar_kws={'label': 'Score'},
                   linewidths=2, linecolor='white')
        plt.title('Comprehensive Metrics Heatmap', fontsize=14, fontweight='bold')
        plt.ylabel('Model Type', fontsize=12, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('models/metrics_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Metrics heatmap saved to: models/metrics_heatmap.png")
    
    def generate_comprehensive_report(self, metrics, y_test, y_pred, y_proba):
        """
        Generate comprehensive performance report.
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE REPORT")
        print("="*80)
        
        # Basic metrics
        print("\n📊 CLASSIFICATION METRICS:")
        print("-" * 80)
        print(f"{'Metric':<20} {'Your Model':<15} {'Ideal Model':<15} {'Gap':<15}")
        print("-" * 80)
        
        metrics_list = [
            ('Accuracy', metrics['accuracy']),
            ('Precision', metrics['precision']),
            ('Recall', metrics['recall']),
            ('F1-Score', metrics['f1_score']),
            ('AUC-ROC', metrics['auc_roc'])
        ]
        
        for name, score in metrics_list:
            gap = 1.0 - score
            status = "✓" if gap < 0.1 else "⚠" if gap < 0.3 else "✗"
            print(f"{status} {name:<18} {score:<15.4f} {1.0:<15.4f} {gap:<15.4f}")
        
        # Confusion matrix details
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        
        print("\n🎯 CONFUSION MATRIX BREAKDOWN:")
        print("-" * 80)
        print(f"True Negatives (Real → Real):    {tn:>5} ({tn/(tn+fp)*100:>5.1f}%)")
        print(f"False Positives (Real → Fake):   {fp:>5} ({fp/(tn+fp)*100:>5.1f}%)")
        print(f"False Negatives (Fake → Real):   {fn:>5} ({fn/(fn+tp)*100:>5.1f}%)")
        print(f"True Positives (Fake → Fake):    {tp:>5} ({tp/(fn+tp)*100:>5.1f}%)")
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print("\n📈 ADDITIONAL METRICS:")
        print("-" * 80)
        print(f"Specificity (True Negative Rate): {specificity:.4f}")
        print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
        print(f"False Positive Rate:              {fp/(fp+tn):.4f}")
        print(f"False Negative Rate:              {fn/(fn+tp):.4f}")
        
        print("\n💡 PERFORMANCE INSIGHTS:")
        print("-" * 80)
        
        if metrics['accuracy'] >= 0.9:
            print("✓ Excellent accuracy! Model performs very well.")
        elif metrics['accuracy'] >= 0.7:
            print("⚠ Good accuracy, but there's room for improvement.")
        else:
            print("✗ Accuracy needs improvement. Consider more training data.")
        
        if metrics['auc_roc'] >= 0.9:
            print("✓ Excellent AUC-ROC! Strong discriminative ability.")
        elif metrics['auc_roc'] >= 0.7:
            print("⚠ Good AUC-ROC, model can distinguish classes reasonably well.")
        else:
            print("✗ Low AUC-ROC. Model struggles to distinguish real from fake.")
        
        if abs(metrics['precision'] - metrics['recall']) < 0.1:
            print("✓ Balanced precision and recall.")
        else:
            print("⚠ Imbalanced precision and recall. Consider adjusting threshold.")
        
        print("="*80 + "\n")
    
    def save_model(self, path='models/deepfake_xgboost.json'):
        """
        Save trained model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        print(f"✓ Model saved to: {path}")
    
    def load_model(self, path='models/deepfake_xgboost.json'):
        """
        Load trained model.
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        print(f"✓ Model loaded from: {path}")


def main():
    """
    Main execution function.
    """
    print("\n" + "="*60)
    print("DEEPFAKE DETECTION - XGBOOST MODEL")
    print("="*60)
    
    # Initialize model (MOCK MODE for now)
    model = DeepfakeDetectionModel(use_mock_data=True)
    
    # Prepare data
    X_train, X_test, y_train, y_test = model.prepare_data()
    
    # Train model
    model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    
    # Get predictions for visualizations
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Generate comprehensive report
    model.generate_comprehensive_report(metrics, y_test, y_pred, y_proba)
    
    # Visualizations
    print("\n📊 Generating visualizations...")
    
    # 1. Feature Importance
    model.plot_feature_importance(top_n=15)
    
    # 2. Confusion Matrix Heatmap
    model.plot_confusion_matrix(metrics['confusion_matrix'])
    
    # 3. ROC Curve
    model.plot_roc_curve(y_test, y_proba)
    
    # 4. Performance Comparison
    model.plot_performance_comparison(metrics)
    
    # 5. Metrics Heatmap
    model.plot_metrics_heatmap(metrics)
    
    # Test single prediction
    print("\n🎯 SINGLE PREDICTION TEST")
    print("="*60)
    
    # Create a sample (suspicious features)
    sample_features = {
        'cnn_max_score': 0.85,
        'cnn_mean_score': 0.75,
        'cnn_std_score': 0.15,
        'freq_mean': 0.72,
        'freq_std': 0.25,
        'freq_max': 0.88,
        'temporal_mean': 0.65,
        'temporal_max': 0.78,
        'temporal_variance': 0.22,
        'lipsync_score': 0.45,
        'lipsync_confidence': 0.65,
        'num_faces': 1,
        'num_frames': 250,
        'avg_face_size': 0.35,
        'spatial_artifacts': 7,
        'temporal_artifacts': 5,
        'blending_score': 0.82
    }
    
    result = model.predict_single(sample_features)
    
    print(f"Verdict: {result['verdict']}")
    print(f"Confidence: {result['confidence']:.2f}%")
    print(f"Probability Real: {result['prob_real']:.4f}")
    print(f"Probability Fake: {result['prob_fake']:.4f}")
    print("="*60)
    
    # Save model
    model.save_model('models/deepfake_xgboost.json')
    
    print("\n✅ ALL DONE!")
    print("\n💡 TO USE WITH REAL DATASET:")
    print("   1. Set use_mock_data=False")
    print("   2. Provide dataset_path to prepare_data()")
    print("   3. Run training again")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
