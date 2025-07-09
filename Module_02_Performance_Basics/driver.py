"""
Driver Script for Biometrics Performance Metrics
Module 2: Modeling Basics

This script demonstrates the usage of the metrics and confusion matrix modules
by generating test data and computing various performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from metrics import calculate_all_metrics, print_metrics
from confusion_matrix import (visualize_confusion_matrix, print_confusion_matrix, 
                             compare_confusion_matrices)


def create_results_directory():
    """
    Create a timestamped directory for saving results.
    
    Returns:
        str: Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"biometrics_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def generate_test_data(n_samples=1000, scenario='balanced'):
    """
    Generate synthetic test data for different scenarios.
    
    Args:
        n_samples: Number of samples to generate
        scenario: Type of scenario ('balanced', 'imbalanced', 'high_precision', 
                 'high_recall', 'poor_performance')
        
    Returns:
        tuple: (y_true, y_pred) arrays
    """
    np.random.seed(42)  # For reproducibility
    
    if scenario == 'balanced':
        # Balanced dataset with good performance
        y_true = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
        y_pred = np.copy(y_true)
        # Add some noise
        flip_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        y_pred[flip_indices] = 1 - y_pred[flip_indices]
        
    elif scenario == 'imbalanced':
        # Imbalanced dataset (more negatives)
        y_true = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
        y_pred = np.copy(y_true)
        # Add some noise
        flip_indices = np.random.choice(n_samples, size=int(0.15 * n_samples), replace=False)
        y_pred[flip_indices] = 1 - y_pred[flip_indices]
        
    elif scenario == 'high_precision':
        # High precision, lower recall
        y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        y_pred = np.copy(y_true)
        # Make many false negatives but few false positives
        positive_indices = np.where(y_true == 1)[0]
        negative_indices = np.where(y_true == 0)[0]
        
        # Flip 40% of positives to negatives (high FN)
        flip_pos = np.random.choice(positive_indices, size=int(0.4 * len(positive_indices)), replace=False)
        y_pred[flip_pos] = 0
        
        # Flip only 5% of negatives to positives (low FP)
        flip_neg = np.random.choice(negative_indices, size=int(0.05 * len(negative_indices)), replace=False)
        y_pred[flip_neg] = 1
        
    elif scenario == 'high_recall':
        # High recall, lower precision
        y_true = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
        y_pred = np.copy(y_true)
        # Make many false positives but few false negatives
        positive_indices = np.where(y_true == 1)[0]
        negative_indices = np.where(y_true == 0)[0]
        
        # Flip only 10% of positives to negatives (low FN)
        flip_pos = np.random.choice(positive_indices, size=int(0.1 * len(positive_indices)), replace=False)
        y_pred[flip_pos] = 0
        
        # Flip 30% of negatives to positives (high FP)
        flip_neg = np.random.choice(negative_indices, size=int(0.3 * len(negative_indices)), replace=False)
        y_pred[flip_neg] = 1
        
    elif scenario == 'poor_performance':
        # Poor performance with many errors
        y_true = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
        y_pred = np.copy(y_true)
        # Add lots of noise
        flip_indices = np.random.choice(n_samples, size=int(0.4 * n_samples), replace=False)
        y_pred[flip_indices] = 1 - y_pred[flip_indices]
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    return y_true, y_pred


def demonstrate_single_scenario(scenario_name, y_true, y_pred, save_figures=True, results_dir=None):
    """
    Demonstrate metrics calculation and visualization for a single scenario.
    
    Args:
        scenario_name: Name of the scenario
        y_true: True labels
        y_pred: Predicted labels
        save_figures: Whether to save the confusion matrix figures
        results_dir: Directory to save results in
    """
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name.upper()}")
    print(f"{'='*60}")
    
    # Calculate and print metrics
    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics(metrics)
    
    print(f"\n{'-'*40}")
    print("CONFUSION MATRIX")
    print(f"{'-'*40}")
    print_confusion_matrix(y_true, y_pred)
    
    # Visualize confusion matrix
    save_path = None
    if save_figures:
        filename = f"confusion_matrix_{scenario_name}.png"
        save_path = os.path.join(results_dir, filename) if results_dir else filename
    
    fig = visualize_confusion_matrix(y_true, y_pred, 
                                   title=f"Confusion Matrix - {scenario_name.title()}",
                                   save_path=save_path)
    plt.show()
    
    return metrics


def demonstrate_comparison(save_figures=True, results_dir=None):
    """
    Demonstrate comparison of multiple scenarios.
    
    Args:
        save_figures: Whether to save the comparison figure
        results_dir: Directory to save results in
    """
    print(f"\n{'='*60}")
    print("COMPARISON OF MULTIPLE SCENARIOS")
    print(f"{'='*60}")
    
    scenarios = ['balanced', 'imbalanced', 'high_precision', 'high_recall', 'poor_performance']
    y_true_list = []
    y_pred_list = []
    titles = []
    
    for scenario in scenarios:
        y_true, y_pred = generate_test_data(n_samples=500, scenario=scenario)
        y_true_list.append(y_true)
        y_pred_list.append(y_pred)
        titles.append(scenario.replace('_', ' ').title())
    
    # Create comparison visualization
    save_path = None
    if save_figures:
        filename = "confusion_matrices_comparison.png"
        save_path = os.path.join(results_dir, filename) if results_dir else filename
    
    fig = compare_confusion_matrices(y_true_list, y_pred_list, titles, figsize=(20, 4),
                                   save_path=save_path)
    plt.show()
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Scenario':<15} {'Accuracy':<10} {'F1':<10} {'Precision':<10} {'Recall':<10} {'Specificity':<12}")
    print("-" * 100)
    
    for i, scenario in enumerate(scenarios):
        metrics = calculate_all_metrics(y_true_list[i], y_pred_list[i])
        print(f"{scenario.replace('_', ' ').title():<15} "
              f"{metrics['Accuracy']:<10.3f} "
              f"{metrics['F1']:<10.3f} "
              f"{metrics['PPV']:<10.3f} "
              f"{metrics['TPR']:<10.3f} "
              f"{metrics['TNR']:<12.3f}")


def test_edge_cases():
    """
    Test edge cases and boundary conditions.
    """
    print(f"\n{'='*60}")
    print("EDGE CASES AND BOUNDARY CONDITIONS")
    print(f"{'='*60}")
    
    # Test case 1: Perfect classifier
    print("\n1. Perfect Classifier:")
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics(metrics)
    
    # Test case 2: Worst possible classifier
    print("\n2. Worst Possible Classifier:")
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 1])
    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics(metrics)
    
    # Test case 3: All positive predictions
    print("\n3. All Positive Predictions:")
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics(metrics)
    
    # Test case 4: All negative predictions
    print("\n4. All Negative Predictions:")
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    metrics = calculate_all_metrics(y_true, y_pred)
    print_metrics(metrics)


def main(save_figures=True):
    """
    Main function to run all demonstrations.
    
    Args:
        save_figures: Whether to save confusion matrix figures
    """
    print("BIOMETRICS PERFORMANCE METRICS DEMONSTRATION")
    print("Module 2: Modeling Basics")
    print("=" * 60)
    
    # Create results directory if saving figures
    results_dir = None
    if save_figures:
        results_dir = create_results_directory()
        print(f"Note: Confusion matrix figures will be saved to: {results_dir}")
        print("=" * 60)
    
    # Test individual scenarios
    scenarios = ['balanced', 'imbalanced', 'high_precision', 'high_recall', 'poor_performance']
    
    for scenario in scenarios:
        y_true, y_pred = generate_test_data(n_samples=1000, scenario=scenario)
        demonstrate_single_scenario(scenario, y_true, y_pred, save_figures=save_figures, 
                                  results_dir=results_dir)
    
    # Demonstrate comparison
    demonstrate_comparison(save_figures=save_figures, results_dir=results_dir)
    
    # Test edge cases
    test_edge_cases()
    
    print(f"\n{'='*60}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*60}")
    
    if save_figures and results_dir:
        print(f"\nAll confusion matrix figures have been saved to: {results_dir}")
        print("Files saved:")
        print("- confusion_matrix_balanced.png")
        print("- confusion_matrix_imbalanced.png")
        print("- confusion_matrix_high_precision.png")
        print("- confusion_matrix_high_recall.png")
        print("- confusion_matrix_poor_performance.png")
        print("- confusion_matrices_comparison.png")
    
    print("\nKey Takeaways:")
    print("1. Green cells represent correct classifications (TPR, TNR)")
    print("2. Red cells represent incorrect classifications (FPR, FNR)")
    print("3. Darker colors indicate higher rates")
    print("4. Text color changes based on rate: white (<50%), black (≥50%)")
    print("5. Different scenarios show trade-offs between precision and recall")


if __name__ == "__main__":
    main()
