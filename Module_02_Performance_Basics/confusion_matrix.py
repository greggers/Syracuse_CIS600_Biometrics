"""
Confusion Matrix Generator and Visualizer for Biometrics
Module 2: Modeling Basics

This module provides functions to generate and visualize confusion matrices
with color coding based on performance rates.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from metrics import calculate_confusion_matrix_values, calculate_all_metrics


def create_confusion_matrix(y_true, y_pred):
    """
    Create a 2x2 confusion matrix from true and predicted labels.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
        
    Returns:
        numpy.ndarray: 2x2 confusion matrix
    """
    TP, TN, FP, FN = calculate_confusion_matrix_values(y_true, y_pred)
    
    # Standard confusion matrix layout:
    # [[TN, FP],
    #  [FN, TP]]
    confusion_matrix = np.array([[TN, FP],
                                [FN, TP]])
    
    return confusion_matrix


def get_color_and_text_color(value, is_correct):
    """
    Determine the background color and text color based on value and correctness.
    
    Args:
        value: The rate value (0.0 to 1.0)
        is_correct: Boolean indicating if this is a correct classification
        
    Returns:
        tuple: (background_color, text_color)
    """
    if value == 0.0:
        return 'black', 'white'
    
    if is_correct:
        # Correct classifications: green intensity based on value
        green_intensity = value
        bg_color = (0, green_intensity, 0)
    else:
        # Incorrect classifications: red intensity based on value
        red_intensity = value
        bg_color = (red_intensity, 0, 0)
    
    # Text color: white if value < 0.5, black if value >= 0.5
    text_color = 'white' if value < 0.5 else 'black'
    
    return bg_color, text_color


def visualize_confusion_matrix(y_true, y_pred, title="Confusion Matrix", figsize=(10, 8), 
                             save_path=None, dpi=300):
    """
    Visualize confusion matrix with color coding based on performance rates.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
        title: Title for the plot
        figsize: Figure size tuple
        save_path: Path to save the figure (optional)
        dpi: DPI for saved figure
        
    Returns:
        matplotlib.figure.Figure: The confusion matrix figure
    """
    # Calculate metrics
    metrics = calculate_all_metrics(y_true, y_pred)
    confusion_matrix = create_confusion_matrix(y_true, y_pred)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define the rates for each cell
    rates = np.array([[metrics['TNR'], metrics['FPR']],
                     [metrics['FNR'], metrics['TPR']]])
    
    # Define which cells represent correct classifications
    is_correct = np.array([[True, False],
                          [False, True]])
    
    # Create the visualization
    for i in range(2):
        for j in range(2):
            value = rates[i, j]
            count = confusion_matrix[i, j]
            correct = is_correct[i, j]
            
            # Get colors
            bg_color, text_color = get_color_and_text_color(value, correct)
            
            # Create rectangle for cell
            rect = patches.Rectangle((j, 1-i), 1, 1, 
                                   linewidth=2, edgecolor='black', 
                                   facecolor=bg_color)
            ax.add_patch(rect)
            
            # Add text with count and rate
            ax.text(j + 0.5, 1-i + 0.6, f'{count}', 
                   ha='center', va='center', fontsize=16, 
                   fontweight='bold', color=text_color)
            ax.text(j + 0.5, 1-i + 0.3, f'{value:.3f}', 
                   ha='center', va='center', fontsize=12, 
                   color=text_color)
    
    # Set up the plot
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_aspect('equal')
    
    # Labels
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(['Predicted: 0', 'Predicted: 1'])
    ax.set_yticklabels(['Actual: 1', 'Actual: 0'])
    
    # Title and labels
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    # Add cell labels
    cell_labels = [['TN (TNR)', 'FP (FPR)'],
                   ['FN (FNR)', 'TP (TPR)']]
    
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, 1-i + 0.05, cell_labels[i][j], 
                   ha='center', va='center', fontsize=10, 
                   style='italic', color='gray')
    
    # Add metrics summary
    metrics_text = (f"Accuracy: {metrics['Accuracy']:.3f}\n"
                   f"F1 Score: {metrics['F1']:.3f}\n"
                   f"Precision: {metrics['PPV']:.3f}")
    
    ax.text(2.1, 1.5, metrics_text, fontsize=10, 
           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                                             facecolor="lightgray", alpha=0.7))
    
    # Add legend
    legend_elements = [
        patches.Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.7, 
                         label='Correct Classification'),
        patches.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.7, 
                         label='Incorrect Classification'),
        patches.Rectangle((0, 0), 1, 1, facecolor='black', 
                         label='Zero Rate')
    ]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(2.1, 1.0))
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    return fig


def print_confusion_matrix(y_true, y_pred):
    """
    Print confusion matrix in a formatted way.
    
    Args:
        y_true: Array of true binary labels (0 or 1)
        y_pred: Array of predicted binary labels (0 or 1)
    """
    confusion_matrix = create_confusion_matrix(y_true, y_pred)
    metrics = calculate_all_metrics(y_true, y_pred)
    
    print("=== Confusion Matrix ===")
    print("                 Predicted")
    print("                 0      1")
    print(f"Actual    0    {confusion_matrix[0,0]:4d}   {confusion_matrix[0,1]:4d}")
    print(f"          1    {confusion_matrix[1,0]:4d}   {confusion_matrix[1,1]:4d}")
    print()
    print("=== Rates Matrix ===")
    print("                 Predicted")
    print("                 0        1")
    print(f"Actual    0    {metrics['TNR']:.3f}   {metrics['FPR']:.3f}")
    print(f"          1    {metrics['FNR']:.3f}   {metrics['TPR']:.3f}")
    print()
    print("Legend:")
    print("TN (True Negative) - TNR (True Negative Rate)")
    print("FP (False Positive) - FPR (False Positive Rate)")
    print("FN (False Negative) - FNR (False Negative Rate)")
    print("TP (True Positive) - TPR (True Positive Rate)")


def compare_confusion_matrices(y_true_list, y_pred_list, titles, figsize=(15, 5), 
                             save_path=None, dpi=300):
    """
    Compare multiple confusion matrices side by side.
    
    Args:
        y_true_list: List of true label arrays
        y_pred_list: List of predicted label arrays
        titles: List of titles for each matrix
        figsize: Figure size tuple
        save_path: Path to save the figure (optional)
        dpi: DPI for saved figure
        
    Returns:
        matplotlib.figure.Figure: The comparison figure
    """
    n_matrices = len(y_true_list)
    fig, axes = plt.subplots(1, n_matrices, figsize=figsize)
    
    if n_matrices == 1:
        axes = [axes]
    
    for idx, (y_true, y_pred, title) in enumerate(zip(y_true_list, y_pred_list, titles)):
        ax = axes[idx]
        
        # Calculate metrics
        metrics = calculate_all_metrics(y_true, y_pred)
        confusion_matrix = create_confusion_matrix(y_true, y_pred)
        
        # Define the rates for each cell
        rates = np.array([[metrics['TNR'], metrics['FPR']],
                         [metrics['FNR'], metrics['TPR']]])
        
        # Define which cells represent correct classifications
        is_correct = np.array([[True, False],
                              [False, True]])
        
        # Create the visualization
        for i in range(2):
            for j in range(2):
                value = rates[i, j]
                count = confusion_matrix[i, j]
                correct = is_correct[i, j]
                
                # Get colors
                bg_color, text_color = get_color_and_text_color(value, correct)
                
                # Create rectangle for cell
                rect = patches.Rectangle((j, 1-i), 1, 1, 
                                       linewidth=2, edgecolor='black', 
                                       facecolor=bg_color)
                ax.add_patch(rect)
                
                # Add text with count and rate
                ax.text(j + 0.5, 1-i + 0.6, f'{count}', 
                       ha='center', va='center', fontsize=14, 
                       fontweight='bold', color=text_color)
                ax.text(j + 0.5, 1-i + 0.3, f'{value:.3f}', 
                       ha='center', va='center', fontsize=10, 
                       color=text_color)
        
        # Set up the plot
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_aspect('equal')
        
        # Labels
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(['Pred: 0', 'Pred: 1'])
        ax.set_yticklabels(['True: 1', 'True: 0'])
        
        # Title
        ax.set_title(f'{title}\nAcc: {metrics["Accuracy"]:.3f}', 
                    fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure if path is provided
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Comparison figure saved to: {save_path}")
    
    return fig
