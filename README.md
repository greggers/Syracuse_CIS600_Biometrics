# Syracuse_CIS663_Biometrics
Repository for Biometrics Graduate course at Syracuse University (CIS 663) taught by Dr. Greg Wagner. Covers biometrics techniques and performance measurement tools.

## Course Overview
This repository contains practical implementations and demonstrations of biometric system evaluation techniques, focusing on performance metrics and analysis methods used in biometric authentication systems.

## Modules

### Module 2: Modeling Basics - Performance Metrics
**Location:** `Module_02_Modeling_Basics/`

This module provides comprehensive tools for calculating and visualizing biometric system performance metrics.

#### Features:
- **Comprehensive Metrics Calculation**: Implementation of all standard biometric performance metrics
- **Visual Confusion Matrices**: Color-coded confusion matrices with intuitive visualization
- **Multiple Scenarios**: Demonstration of different biometric system scenarios
- **Automated Reporting**: Timestamped results with saved figures

#### Files:
- `metrics.py` - Core performance metrics calculations
- `confusion_matrix.py` - Confusion matrix generation and visualization
- `driver.py` - Demonstration script with test scenarios

#### Metrics Implemented:
- **Sensitivity/Recall/True Positive Rate (TPR)**: `TPR = TP / (TP + FN)`
- **Specificity/Selectivity/True Negative Rate (TNR)**: `TNR = TN / (TN + FP)`
- **Precision/Positive Predictive Value (PPV)**: `PPV = TP / (TP + FP)`
- **False Positive Rate (FPR)**: `FPR = FP / (FP + TN)`
- **False Negative Rate (FNR)**: `FNR = FN / (FN + TP)`
- **Accuracy**: `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
- **F1 Score**: `F1 = 2 * TP / (2*TP + FP + FN)`

#### Visualization Features:
- **Color-coded confusion matrices**: Green for correct classifications, red for incorrect
- **Rate-based coloring**: Brightness corresponds to rate values (0-100%)
- **Adaptive text color**: White text for rates <50%, black text for rates ≥50%
- **Scenario comparisons**: Side-by-side visualization of multiple scenarios
- **Automated saving**: High-resolution PNG exports with timestamps

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Required Dependencies
Install the required packages using pip:

```bash
pip install numpy matplotlib
```

Or install all requirements at once:

```bash
pip install -r requirements.txt
```

### Requirements File
Create a `requirements.txt` file with:
```
numpy>=1.19.0
matplotlib>=3.3.0
```

## Usage

### Running Module 2
Navigate to the Module 2 directory and run the demonstration:

```bash
cd Module_02_Modeling_Basics
python driver.py
```

#### Expected Output:
1. **Console Output**: Detailed metrics for each scenario
2. **Interactive Plots**: Confusion matrices displayed in separate windows
3. **Saved Files**: Timestamped directory containing high-resolution PNG files:
   - `confusion_matrix_balanced.png`
   - `confusion_matrix_imbalanced.png`
   - `confusion_matrix_high_precision.png`
   - `confusion_matrix_high_recall.png`
   - `confusion_matrix_poor_performance.png`
   - `confusion_matrices_comparison.png`

#### Customization:
```python
# Run without saving figures
python -c "from driver import main; main(save_figures=False)"

# Import and use individual functions
from metrics import calculate_all_metrics
from confusion_matrix import visualize_confusion_matrix

# Calculate metrics for your own data
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
metrics = calculate_all_metrics(y_true, y_pred)
```

## Test Scenarios

The module includes five predefined scenarios to demonstrate different biometric system behaviors:

1. **Balanced**: Equal distribution with good overall performance
2. **Imbalanced**: Skewed data distribution (common in biometrics)
3. **High Precision**: Few false positives, more false negatives
4. **High Recall**: Few false negatives, more false positives
5. **Poor Performance**: High error rates across all metrics

## Educational Use

This module is designed for educational purposes to help students understand:
- Trade-offs between precision and recall in biometric systems
- Visual interpretation of confusion matrices
- Impact of data distribution on performance metrics
- Practical implications of different error types in biometric authentication

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Dr. Greg Wagner  
Syracuse University  
Department of Computer Science
