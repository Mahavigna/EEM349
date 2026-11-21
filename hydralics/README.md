# Condition Monitoring of Hydraulic Systems for Industry 4.0: Comparative Analysis Using SVM and k-NN with the Impact of PCA

## Task: Classification

The task is binary classification. Using 34 numerical sensor features, the models predict whether an operating condition is **stable** or **unstable**.

The unstable class is treated as the positive class because detecting potentially disruptive hydraulic failures is more important than merely confirming normal operation.

## Dataset Justification

The dataset represents real-world hydraulic test rig sensor readings from an Industry 4.0 condition monitoring setup, measuring physical variables across 2,205 operating cycles.

Hydraulic system instability can affect interconnected production machinery, heavy presses, automated actuators, industrial controllers, and robotic manipulators.

Consequently, classifying unstable hydraulic conditions can be treated as component- and system-level fault-risk detection for smart manufacturing environments.

## Algorithm Justification

### k-Nearest Neighbours

k-Nearest Neighbours was selected as a simple instance-based classifier. It predicts hydraulic stability by comparing a new operating condition with similar historical observations. This provides an interpretable baseline for determining whether stable and unstable operating states form distinguishable neighbourhoods in the feature space.

### Support Vector Machine

Support Vector Machine was selected because hydraulic stability may depend on nonlinear relationships among sensor pressures, fluid temperatures, flow rates, and vibration levels. Radial basis function and polynomial kernels are investigated to determine which nonlinear decision boundary is more suitable. SVM applies the maximum-margin principle, making it suitable for distinguishing complex stable and unstable operating regions.

## Objectives

1. Classify industrially relevant hydraulic operating cycles as stable or unstable using SVM and k-NN.
2. Compare both classifiers using accuracy, precision, recall, F1-score, confusion matrices, training time, and prediction time.
3. Investigate the effects of $C$, gamma, and kernel selection on SVM performance.
4. Investigate the effects of neighbour count, distance metric, and weighting method on k-NN performance.
5. Evaluate both algorithms with and without Principal Component Analysis.
6. Determine which model is more suitable for detecting hydraulic instability risks associated with Industry 4.0 operations.

## Problem Statement

**Theme:** Manufacturing & Industry 4.0

Modern Industry 4.0 facilities depend on stable hydraulic systems to operate automated production lines, industrial presses, heavy machinery, programmable logic controllers, and robotic actuators.

An unstable hydraulic condition can interrupt production, create abnormal equipment operation, and increase the risk of industrial downtime.

Therefore, this project applies supervised machine learning to classify hydraulic operating conditions as stable or unstable, providing an early indication of equipment instability that could support fault detection and operational monitoring in smart manufacturing facilities.