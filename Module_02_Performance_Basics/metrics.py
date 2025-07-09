"""
Performance Metrics Calculator for Biometrics
Module 2: Modeling Basics

This module provides functions to calculate various performance metrics
used in biometric systems evaluation.
"""

import numpy as np


def calculate_confusion_matrix_values(y_true, y_pred):
    """
    Calculate TP, TN, FP, FN from true and predicted labels.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
        
    Returns:
        tuple: (TP, TN, FP, FN)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    
    return TP, TN, FP, FN


def sensitivity_recall_tpr(TP, FN):
    """
    Calculate Sensitivity/Recall/True Positive Rate
    TPR = TP / P = TP / (TP + FN) = 1 – FNR
    """
    P = TP + FN
    if P == 0:
        return 0.0
    return TP / P


def specificity_selectivity_tnr(TN, FP):
    """
    Calculate Specificity/Selectivity/True Negative Rate
    TNR = TN / N = TN / (TN + FP) = 1 – FPR
    """
    N = TN + FP
    if N == 0:
        return 0.0
    return TN / N


def precision_ppv(TP, FP):
    """
    Calculate Precision/Positive Predictive Value
    PPV = TP / (TP + FP)
    """
    if (TP + FP) == 0:
        return 0.0
    return TP / (TP + FP)


def false_positive_rate(FP, TN):
    """
    Calculate False Positive Rate
    FPR = FP / N = FP / (FP + TN) = 1 - TNR
    """
    N = FP + TN
    if N == 0:
        return 0.0
    return FP / N


def false_negative_rate(FN, TP):
    """
    Calculate False Negative Rate
    FNR = FN / P = FN / (FN + TP) = 1 - TPR
    """
    P = FN + TP
    if P == 0:
        return 0.0
    return FN / P


def accuracy(TP, TN, FP, FN):
    """
    Calculate Accuracy
    Accuracy = (TP + TN) / (P + N) = (TP + TN) / (TP + TN + FP + FN)
    """
    total = TP + TN + FP + FN
    if total == 0:
        return 0.0
    return (TP + TN) / total


def f1_score(TP, FP, FN):
    """
    Calculate F1 Score
    F1 = 2 * TP / (2*TP + FP + FN)
    """
    denominator = 2 * TP + FP + FN
    if denominator == 0:
        return 0.0
    return (2 * TP) / denominator


def calculate_all_metrics(y_true, y_pred):
    """
    Calculate all performance metrics from true and predicted labels.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
        
    Returns:
        dict: Dictionary containing all calculated metrics
    """
    TP, TN, FP, FN = calculate_confusion_matrix_values(y_true, y_pred)
    
    metrics = {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'TPR': sensitivity_recall_tpr(TP, FN),
        'TNR': specificity_selectivity_tnr(TN, FP),
        'PPV': precision_ppv(TP, FP),
        'FPR': false_positive_rate(FP, TN),
        'FNR': false_negative_rate(FN, TP),
        'Accuracy': accuracy(TP, TN, FP, FN),
        'F1': f1_score(TP, FP, FN)
    }
    
    return metrics


def print_metrics(metrics):
    """
    Print all metrics in a formatted way.
    
    Args:
        metrics: Dictionary containing all metrics
    """
    print("=== Performance Metrics ===")
    print(f"True Positives (TP): {metrics['TP']}")
    print(f"True Negatives (TN): {metrics['TN']}")
    print(f"False Positives (FP): {metrics['FP']}")
    print(f"False Negatives (FN): {metrics['FN']}")
    print("\n=== Rates ===")
    print(f"Sensitivity/Recall/TPR: {metrics['TPR']:.4f}")
    print(f"Specificity/Selectivity/TNR: {metrics['TNR']:.4f}")
    print(f"Precision/PPV: {metrics['PPV']:.4f}")
    print(f"False Positive Rate (FPR): {metrics['FPR']:.4f}")
    print(f"False Negative Rate (FNR): {metrics['FNR']:.4f}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"F1 Score: {metrics['F1']:.4f}")
