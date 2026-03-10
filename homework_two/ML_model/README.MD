# Congestion Control Model: `cwnd_delta` Calculation Guide

This document explains the mathematical and logical process used to calculate the change in the Congestion Window ($\Delta cwnd$) using the extracted Machine Learning weights and normalization parameters.

## Mathematical Formula
The model uses a **Linear Regression** approach based on a sliding window of the last 5 time-steps. To ensure stability across different network scales, all inputs are standardized before being passed to the model.

### Step 1: Feature Normalization (Z-Score)
For every raw metric $x$ (Goodput, RTT, Loss) in the 5-step history, we calculate the normalized value $z$:

$$z = \frac{x_{raw} - \mu}{\sigma}$$

* **$\mu$ (Mean):** The average value of that metric for the specific destination.
* **$\sigma$ (StdDev):** The standard deviation of that metric for the specific destination.

### Step 2: Dot Product Calculation
The `cwnd_delta` is the sum of the bias and the product of each normalized feature and its corresponding weight:

$$\Delta cwnd = \text{Bias} + \sum_{i=0}^{14} (Weight_i \cdot z_i)$$
