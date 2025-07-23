# Syracuse_CIS663_Biometrics
Repository for Biometrics Graduate course at Syracuse University (CIS 663) taught by Dr. Greg Wagner.

## Course Overview
This repository contains practical implementations and demonstrations of biometric system evaluation techniques, focusing on performance metrics and analysis methods used in biometric authentication systems.

## Modules

### Module 1: Introduction to Biometrics
**Location:** `Module_01_Introduction/`

This module provides a comprehensive introduction to the field of biometrics, including its history, types, and applications.
#### Features:

*   **Biometric Definitions**: Explanation of key terms and concepts in biometrics.
*   **Types of Biometrics**: Discussion of various biometric modalities (e.g., face, fingerprint, iris).
*   **Applications of Biometrics**: Overview of biometric systems used in real-world scenarios (e.g., access control, identity verification).

### Module 2: Modeling Basics - Performance Metrics
**Location:** `Module_02_Modeling_Basics/`

This module provides comprehensive tools for calculating and visualizing biometric system performance metrics.
#### Features:

*   **Comprehensive Metrics Calculation**: Implementation of all standard biometric performance metrics.
*   **Visual Confusion Matrices**: Color-coded confusion matrices with intuitive visualization.
*   **Multiple Scenarios**: Demonstration of different biometric system scenarios.
*   **Automated Reporting**: Timestamped results with saved figures.

### Module 3: Fingerprint Minutiae Detection and Analysis
**Location:** `Module_03_FingerPrinting/`

This module demonstrates fingerprint minutiae detection, analysis, and matching techniques.

#### Features:

*   **Fingerprint Preprocessing**: Histogram equalization, Gabor filtering for enhancement.
*   **Orientation Field Calculation**: Essential for singularity detection.
*   **Poincaré Index Method**: Mathematical approach to detect and classify singularities.
*   **Pattern Classification**: Distinguishing between arch, loop, and whorl patterns.
*   **Cross-Correlation Matching**: Template matching with rotation invariance.

### Module 4: Face Detection and Recognition
**Location:** `Module_04_Face_Detection/`

This module demonstrates face detection using the Viola-Jones algorithm and recognition techniques based on minutiae features.

#### Features:
*   **Face Detection**: Implementation of Haar Cascade Classifiers for face detection.

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

