# remote_sensing_object_detection

# Introduction
A Highly Generalizable Large Vision Models for Remote Sensing Object Detection in 2023 GFSAIT Competition.

# Method
this study proposes a novel remote sensing object detection framework integrating ground-based prior construction and on-orbit inconsistency measurement. The method employs the encoder of the Segment Anything Model (SAM) for feature extraction and compresses data using Gaussian Mixture Models (GMM) with Bayesian Information Criterion (BIC), significantly reducing storage and transmission requirements. Object detection is performed via a dictionary look-up mechanism, efficiently identifying deviations using Manhattan distance within SAM’s representation space.
Flowchart of Ground-based Priors Construction:
![image](https://github.com/user-attachments/assets/e8f207fa-b1a7-4c91-8254-d298e25e4ca8)

Flowchart of Object Anomaly Detection:
![image](https://github.com/user-attachments/assets/5b276f6b-d223-41f4-983a-6c55dc62fd38)



