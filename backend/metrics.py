"""
Metrics calculation module for deepfake detection.
Provides Accuracy, Precision, Recall, and output quality analysis.
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np


def calculate_metrics(y_true, y_pred):
    """
    Calculate performance metrics.
    
    Args:
        y_true: Ground truth labels (0=authentic, 1=fake)
        y_pred: Predicted labels (0=authentic, 1=fake)
    
    Returns:
        Dictionary with accuracy, precision, recall, and confusion matrix
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm.tolist(),
        "true_positives": int(cm[1, 1]) if cm.shape == (2, 2) else 0,
        "true_negatives": int(cm[0, 0]) if cm.shape == (2, 2) else 0,
        "false_positives": int(cm[0, 1]) if cm.shape == (2, 2) else 0,
        "false_negatives": int(cm[1, 0]) if cm.shape == (2, 2) else 0
    }


def analyze_output_quality(accuracy):
    """
    Analyze output quality based on accuracy.
    
    Args:
        accuracy: Accuracy score (0-1)
    
    Returns:
        Quality category: "Random", "Medium", or "Accurate"
    """
    if accuracy < 0.6:
        return "Random"
    elif accuracy < 0.8:
        return "Medium"
    else:
        return "Accurate"


def print_metrics_report(metrics, output_quality):
    """
    Print a formatted metrics report.
    
    Args:
        metrics: Dictionary of metrics from calculate_metrics()
        output_quality: Quality category from analyze_output_quality()
    """
    print("\n" + "="*60)
    print("PERFORMANCE METRICS REPORT")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"\nOutput Quality: {output_quality}")
    print("\nConfusion Matrix:")
    print(f"  True Negatives:  {metrics['true_negatives']}")
    print(f"  False Positives: {metrics['false_positives']}")
    print(f"  False Negatives: {metrics['false_negatives']}")
    print(f"  True Positives:  {metrics['true_positives']}")
    print("="*60 + "\n")


def generate_test_data(num_samples=50):
    """
    Generate synthetic test data for demonstration.
    
    Args:
        num_samples: Number of test samples to generate
    
    Returns:
        Tuple of (y_true, y_pred, scores)
    """
    # Generate ground truth (50% fake, 50% authentic)
    y_true = np.random.randint(0, 2, num_samples)
    
    # Generate predictions with ~75% accuracy
    y_pred = y_true.copy()
    # Flip 25% of predictions to simulate errors
    flip_indices = np.random.choice(num_samples, size=num_samples//4, replace=False)
    y_pred[flip_indices] = 1 - y_pred[flip_indices]
    
    # Generate corresponding scores
    scores = []
    for true_label, pred_label in zip(y_true, y_pred):
        if pred_label == 1:  # Predicted fake
            scores.append(np.random.uniform(0.6, 0.95))
        else:  # Predicted authentic
            scores.append(np.random.uniform(0.05, 0.4))
    
    return y_true, y_pred, np.array(scores)
