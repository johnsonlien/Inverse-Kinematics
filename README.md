#Inverse Kinematics in C++
## Course Description
This was the third assignment for CSCI 520 - Computer Animations course. 

External libraries [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) and [ADOL-C](https://github.com/coin-or/ADOL-C) were used to perform any linear algebraic equations. 

This program implements forward and inverse kinematics to perform an interactive model that can be dragged and folded. This assignment is used to implement forward and inverse kinematics using two methods to calculate inverse kinematics:
1. Pseudo-Inverse
2. Damped Least Squares (a.k.a Tikhonov Regularization)

This program actually only implemented the damped least squares method but you can check out the pseudo-inverse method by changing the value of 'alpha' in IK.cpp to 0. 

Two different skinning methods have also been implemented:
1. Linear Blend Skinning
2. Dual Quaternions 
