# CSCI 520 Homework 3: Inverse-Kinematics

External libraries "Eigen" and ADOL-C were used to perform any linear algebraic equations. 

This program implements forward and inverse kinematics to perform an interactive model that can be dragged and folded. This assignment is used to implement forward and inverse kinematics using two methods to calculate inverse kinematics:
1. Pseudo-Inverse
2. Damped Least Squares (a.k.a Tikhonov Regularization)

This program actually only implemented the damped least squares method but you can check out the pseudo-inverse method by changing the value of 'alpha' in IK.cpp to 0. 

Two different skinning methods have also been implemented:
1. Linear Blend Skinning
2. Dual Quaternions 
