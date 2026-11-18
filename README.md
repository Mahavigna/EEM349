# Classification of Smart-Grid Instability for Industry 4.0: Comparative Analysis Using SVM and k-NN with the Impact of PCA

## Task: Classification

The task is binary classification. Using 12 numerical grid-operation features, the models predict whether an operating condition is **stable** or **unstable**.

The unstable class is treated as the positive class because detecting potentially disruptive power conditions is more important than merely confirming normal operation.

## Dataset Justification

Although the dataset represents smart-grid operating conditions rather than faults in an individual manufacturing machine, it is relevant to Industry 4.0 because intelligent manufacturing depends on reliable electrical infrastructure.

Grid instability can affect interconnected production machinery, robotic systems, industrial controllers, sensors, and communication devices.

Consequently, classifying unstable grid conditions can be treated as infrastructure-level fault-risk detection for smart manufacturing environments.

## Algorithm Justification

### k-Nearest Neighbours

k-Nearest Neighbours was selected as a simple instance-based classifier. It predicts grid stability by comparing a new operating condition with similar historical observations. This provides an interpretable baseline for determining whether stable and unstable operating states form distinguishable neighbourhoods in the feature space.

### Support Vector Machine

Support Vector Machine was selected because grid stability may depend on nonlinear relationships among participant reaction times, power values, and control coefficients. Radial basis function and polynomial kernels are investigated to determine which nonlinear decision boundary is more suitable. SVM applies the maximum-margin principle, making it suitable for distinguishing complex stable and unstable operating regions.

## Objectives

1. Classify industrially relevant grid conditions as stable or unstable using SVM and k-NN.
2. Compare both classifiers using accuracy, precision, recall, F1-score, confusion matrices, training time, and prediction time.
3. Investigate the effects of $C$, gamma, and kernel selection on SVM performance.
4. Investigate the effects of neighbour count, distance metric, and weighting method on k-NN performance.
5. Evaluate both algorithms with and without Principal Component Analysis.
6. Determine which model is more suitable for detecting power-instability risks associated with Industry 4.0 operations.

## Problem Statement

**Theme:** Manufacturing & Industry 4.0

Modern Industry 4.0 facilities depend on stable electrical power to operate automated production lines, industrial robots, sensors, programmable logic controllers, and Industrial Internet of Things devices.

An unstable power-grid condition can interrupt production, create abnormal equipment operation, and increase the risk of industrial downtime.

Therefore, this project applies supervised machine learning to classify grid operating conditions as stable or unstable, providing an early indication of power-system instability that could support fault detection and operational monitoring in smart manufacturing facilities.
