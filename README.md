#  **IntegratedAESEModel** :  Anomaly Detection Method based on Imbalanced learning and Autoencoder 

## Introduction

AESE Model is an anomaly detection model based on Autoencoder and Self-Paced ensemble model. It is designed to detect anomalies in imbalanced traffic network dataset. It combined the unsupervised learning with imbalanced learning to improve the detection accuracy.

## Usage

The project includes `unsw_main.py` , `nsl_main.py`, `kdd_main.py` ， run the three codes can get the different result in different datasets. We use the three different network datasets, UNSW-NB15, NSL-KDD, KDDCup99. 

`compare_exp.py` is the control group code. You can see clearly the promotion in our method. `Utils.py` is the tool code with method functions. 

## Model Structure

The model first standardizes the input, extracts data labeled as normal traffic from the training set, and then trains the Autoencoder (AE). It computes the confidence interval in the training set to obtain the threshold for anomaly detection. Subsequently, the trained AE is used for anomaly detection on the test set. Based on the detection results, data labeled as normal by the AE, referred to as pseudo-normal traffic, is extracted for further detection. The Self-Paced Ensemble (SPE) is then used for downsampling and classification, resulting in the final classification result. Finally, the outputs are integrated to form the final result.

![](./img\revised.png)