#include "IK.h"
#include "FK.h"
#include "minivectorTemplate.h"
#include <Eigen/Dense>
#include <adolc/adolc.h>
#include <cassert>
#if defined(_WIN32) || defined(WIN32)
  #ifndef _USE_MATH_DEFINES
    #define _USE_MATH_DEFINES
  #endif
#endif
#include <math.h>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

namespace
{

// Converts degrees to radians.
template<typename real>
inline real deg2rad(real deg) { return deg * M_PI / 180.0; }

template<typename real>
Mat3<real> Euler2Rotation(const real angle[3], RotateOrder order)
{
  Mat3<real> RX = Mat3<real>::getElementRotationMatrix(0, deg2rad(angle[0]));
  Mat3<real> RY = Mat3<real>::getElementRotationMatrix(1, deg2rad(angle[1]));
  Mat3<real> RZ = Mat3<real>::getElementRotationMatrix(2, deg2rad(angle[2]));

  switch(order)
  {
    case RotateOrder::XYZ:
      return RZ * RY * RX;
    case RotateOrder::YZX:
      return RX * RZ * RY;
    case RotateOrder::ZXY:
      return RY * RX * RZ;
    case RotateOrder::XZY:
      return RY * RZ * RX;
    case RotateOrder::YXZ:
      return RZ * RX * RY;
    case RotateOrder::ZYX:
      return RX * RY * RZ;
  }
  assert(0);
}

// Performs forward kinematics, using the provided "fk" class.
// This is the function whose Jacobian matrix will be computed using adolc.
// numIKJoints and IKJointIDs specify which joints serve as handles for IK:
//   IKJointIDs is an array of integers of length "numIKJoints"
// Input: numIKJoints, IKJointIDs, fk, eulerAngles (of all joints)
// Output: handlePositions (world-coordinate positions of all the IK joints; length is 3 * numIKJoints)
template<typename real>
void forwardKinematicsFunction(
    int numIKJoints, const int * IKJointIDs, const FK & fk,
    const std::vector<real> & eulerAngles, std::vector<real> & handlePositions)
{
  // Students should implement this.
  // The implementation of this function is very similar to function computeLocalAndGlobalTransforms in the FK class.
  // The recommended approach is to first implement FK::computeLocalAndGlobalTransforms.
  // Then, implement the same algorithm into this function. To do so,
  // you can use fk.getJointUpdateOrder(), fk.getJointRestTranslation(), and fk.getJointRotateOrder() functions.
  // Also useful is the multiplyAffineTransform4ds function in minivectorTemplate.h .
  // It would be in principle possible to unify this "forwardKinematicsFunction" and FK::computeLocalAndGlobalTransforms(),
  // so that code is only written once. We considered this; but it is actually not easily doable.
  // If you find a good approach, feel free to document it in the README file, for extra credit.

	// keeping the R matrix and translation vector separate because handles are joint translations
	vector<Mat3<real>> localRmatrices, globalRmatrices;
	vector<Vec3<real>> localtvectors, globaltvectors;
	
	//compute local transformations
	for (int i = 0; i < fk.getNumJoints(); i++) {
		//convert euler angles to rotation
		// R matrix
		Vec3<real> angles = { eulerAngles[i * 3], eulerAngles[i * 3 + 1], eulerAngles[i * 3 + 2] };
		Mat3<real> tempR = Euler2Rotation(angles.data(), fk.getJointRotateOrder(i));
		
		// R0 matrix
		Vec3<real> restAngles = { fk.getJointOrient(i)[0], fk.getJointOrient(i)[1], fk.getJointOrient(i)[2] };
		Mat3<real> tempR0 = Euler2Rotation(restAngles.data(), fk.getJointRotateOrder(i));

		// translation vector
		Vec3<real> temptvector = { fk.getJointRestTranslation(i)[0], fk.getJointRestTranslation(i)[1], fk.getJointRestTranslation(i)[2] };

		// store into our vectors
		localRmatrices.push_back(tempR0 * tempR);
		localtvectors.push_back(temptvector);
	}

	//compute global transformations
	for (int i = 0; i < fk.getNumJoints(); i++) {
		int child = fk.getJointUpdateOrder(i);
		int parent = fk.getJointParent(child);

		if (parent == -1) { // root
			globalRmatrices.push_back(localRmatrices[child]);
			globaltvectors.push_back(localtvectors[child]);
		}
		else {
			// multiplyAffineTransform4ds function takes outputs into Mat3<real> Rout and Vec3<real> tout

			//multiplyAffineTransform4ds(const Mat3<real> & R1, const Vec3<real> & t1, const Mat3<real> & R2, const Vec3<real> & t2, 
			// Mat3<real> & Rout, Vec3<real> & tout) <-- saves into here
			Mat3<real> Rout;
			Vec3<real> tout;
			multiplyAffineTransform4ds(globalRmatrices[parent], globaltvectors[parent], localRmatrices[child], localtvectors[child], Rout, tout);
			globalRmatrices.push_back(Rout);
			globaltvectors.push_back(tout);
		}
	}

	// Per assignment description the handles are the global translations
	for (int i = 0; i < numIKJoints; i++) {
		for (int j = 0; j < 3; j++) {
			int IKJointID = IKJointIDs[i];
			handlePositions[3 * i + j] = globaltvectors[IKJointID][j];
		}
	}
} // end forwardKinematicsFunction

} // end anonymous namespaces

IK::IK(int numIKJoints, const int * IKJointIDs, FK * inputFK, int adolc_tagID)
{
  this->numIKJoints = numIKJoints;
  this->IKJointIDs = IKJointIDs;
  this->fk = inputFK;
  this->adolc_tagID = adolc_tagID;

  FKInputDim = fk->getNumJoints() * 3;
  FKOutputDim = numIKJoints * 3;

  train_adolc();
}

void IK::train_adolc()
{
  // Students should implement this.
  // Here, you should setup adol_c:
  //   Define adol_c inputs and outputs. 
  //   Use the "forwardKinematicsFunction" as the function that will be computed by adol_c.
  //   This will later make it possible for you to compute the gradient of this function in IK::doIK
  //   (in other words, compute the "Jacobian matrix" J).
  // See ADOLCExample.cpp .
	trace_on(adolc_tagID);
	vector<adouble> input(FKInputDim);		// input for the function
	vector<double> output(FKOutputDim);		// where to output
	vector<adouble> handles(FKOutputDim);	// middle variables

	for (int i = 0; i < FKInputDim; i++) {
		input[i] <<= 0.0;		// Initialize to 0 using ADOL-C <<= operator
	}
	
	// The function to calculate Jacobian matrix with
	forwardKinematicsFunction(numIKJoints, IKJointIDs, *fk, input, handles);
	
	// Output 
	for (int i = 0; i < FKOutputDim; i++) {
		handles[i] >>= output[i];
	}

	trace_off();
}

void IK::doIK(const Vec3d * targetHandlePositions, Vec3d * jointEulerAngles)
{

  // FKInputDim = n, FKOutputDim = m
  int numJoints = fk->getNumJoints(); // Note that is NOT the same as numIKJoints!
  double * handles = new double[FKOutputDim];
  double * J = new double[FKOutputDim * FKInputDim];		// m x n
  double ** Jrows = new double*[FKOutputDim];				// pointers to each of J's starting rows
  double alpha = 0.01;										// try 0.01, 0.001
  Eigen::VectorXd deltab(FKOutputDim);						// m x 1
  Eigen::VectorXd deltatheta(FKInputDim);					// n x 1
  Eigen::MatrixXd Jacobian(FKOutputDim, FKInputDim);		// m x n	(Used to store J as eigen matrix for matrix operations)
  Eigen::MatrixXd Identity = Eigen::MatrixXd::Identity(FKInputDim, FKInputDim);	// n x n

  // Use adolc to evalute the forwardKinematicsFunction and its gradient (Jacobian). It was trained in train_adolc().
  // Specifically, use ::function, and ::jacobian .
  // See ADOLCExample.cpp .
  //
  // Use it implement the Tikhonov IK method (or the pseudoinverse method for extra credit).
  // Note that at entry, "jointEulerAngles" contains the input Euler angles. 
  // Upon exit, jointEulerAngles should contain the new Euler angles.
  
  // Initialize to 0.0
  for (int i = 0; i < FKOutputDim; i++) {
	  handles[i] = 0.0;
  }

  ::function(adolc_tagID, FKOutputDim, FKInputDim, jointEulerAngles->data(), handles);
  
  // Calculate the Jacobian
  for (int i = 0; i < FKOutputDim; i++) {
	  Jrows[i] = &(J[i*FKInputDim]);
  }
  ::jacobian(adolc_tagID, FKOutputDim, FKInputDim, jointEulerAngles->data(), Jrows);
  
  // Solve the IK equation (J^T J + alpha I)deltatheta = J^T deltab
  for (int i = 0; i < numIKJoints; i++) {
	  for (int j = 0; j < 3; j++) {
		  deltab(3 * i + j) = targetHandlePositions[i][j] - handles[3 * i + j];
	  }
  }
  
  // We can convert J -> Eigen::MatrixXd to use matrix operations
  for (int i = 0; i < FKOutputDim; i++) {
	  for (int j = 0; j < FKInputDim; j++) {
		  Jacobian(i, j) = J[FKInputDim * i + j];
	  }
  }
  
  // deltatheta = (J^T * J + alpha * I)^(-1) * (J^T * deltab)
  Eigen::MatrixXd JacobianT = Jacobian.transpose();		// Store the transpose and optimize computation by not having to call transpose() function twice
  deltatheta = ((JacobianT * Jacobian) + (alpha * Identity)).ldlt().solve(JacobianT * deltab);
  
  for (int i = 0; i < numJoints; i++) {
	  for (int j = 0; j < 3; j++) {
		  jointEulerAngles[i][j] += deltatheta(3 * i + j);
	  }
  }
  
}

