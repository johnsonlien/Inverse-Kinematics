#include "skinning.h"
#include "vec3d.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>		// for quaternions

using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

Skinning::Skinning(int numMeshVertices, const double * restMeshVertexPositions,
    const std::string & meshSkinningWeightsFilename)
{
  this->numMeshVertices = numMeshVertices;
  this->restMeshVertexPositions = restMeshVertexPositions;

  cout << "Loading skinning weights..." << endl;
  ifstream fin(meshSkinningWeightsFilename.c_str());
  assert(fin);
  int numWeightMatrixRows = 0, numWeightMatrixCols = 0;
  fin >> numWeightMatrixRows >> numWeightMatrixCols;
  assert(fin.fail() == false);
  assert(numWeightMatrixRows == numMeshVertices);
  int numJoints = numWeightMatrixCols;

  vector<vector<int>> weightMatrixColumnIndices(numWeightMatrixRows);
  vector<vector<double>> weightMatrixEntries(numWeightMatrixRows);
  fin >> ws;
  while(fin.eof() == false)
  {
    int rowID = 0, colID = 0;
    double w = 0.0;
    fin >> rowID >> colID >> w;
    weightMatrixColumnIndices[rowID].push_back(colID);
    weightMatrixEntries[rowID].push_back(w);
    assert(fin.fail() == false);
    fin >> ws;
  }
  fin.close();

  // Build skinning joints and weights.
  numJointsInfluencingEachVertex = 0;
  for (int i = 0; i < numMeshVertices; i++)
    numJointsInfluencingEachVertex = std::max(numJointsInfluencingEachVertex, (int)weightMatrixEntries[i].size());
  assert(numJointsInfluencingEachVertex >= 2);

  // Copy skinning weights from SparseMatrix into meshSkinningJoints and meshSkinningWeights.
  meshSkinningJoints.assign(numJointsInfluencingEachVertex * numMeshVertices, 0);
  meshSkinningWeights.assign(numJointsInfluencingEachVertex * numMeshVertices, 0.0);
  for (int vtxID = 0; vtxID < numMeshVertices; vtxID++)
  {
    vector<pair<double, int>> sortBuffer(numJointsInfluencingEachVertex);
    for (size_t j = 0; j < weightMatrixEntries[vtxID].size(); j++)
    {
      int frameID = weightMatrixColumnIndices[vtxID][j];
      double weight = weightMatrixEntries[vtxID][j];
      sortBuffer[j] = make_pair(weight, frameID);
    }
    sortBuffer.resize(weightMatrixEntries[vtxID].size());
    assert(sortBuffer.size() > 0);
    sort(sortBuffer.rbegin(), sortBuffer.rend()); // sort in descending order using reverse_iterators
    for(size_t i = 0; i < sortBuffer.size(); i++)
    {
      meshSkinningJoints[vtxID * numJointsInfluencingEachVertex + i] = sortBuffer[i].second;
      meshSkinningWeights[vtxID * numJointsInfluencingEachVertex + i] = sortBuffer[i].first;
    }

    // Note: When the number of joints used on this vertex is smaller than numJointsInfluencingEachVertex,
    // the remaining empty entries are initialized to zero due to vector::assign(XX, 0.0) .
  }
}

void Skinning::applySkinning(const RigidTransform4d * jointSkinTransforms, double * newMeshVertexPositions) const
{
	string mode;
#if 0		// Change this to 0 for DualQuaternionSkinning or 1 for LinearBlendSkinning
	mode = "DualQuaternionSkinning";
#else 
	mode = "LinearBlendSkinning";
#endif

	if (mode == "LinearBlendSkinning") {
		for (int i = 0; i < numMeshVertices; i++)	// Iterate through each skinning vertex
		{
			//Initialize Mesh positions to 0
			newMeshVertexPositions[3 * i + 0] = 0.0;
			newMeshVertexPositions[3 * i + 1] = 0.0;
			newMeshVertexPositions[3 * i + 2] = 0.0;
			for (int j = 0; j < numJointsInfluencingEachVertex; j++)	// Iterate through each joint
			{
				int index = i * numJointsInfluencingEachVertex + j;
				int jointIndex = meshSkinningJoints[index];

				Vec4d p_ibar = Vec4d(restMeshVertexPositions[3 * i], restMeshVertexPositions[3 * i + 1], restMeshVertexPositions[3 * i + 2], 1.0);
				Vec4d product = meshSkinningWeights[index] * jointSkinTransforms[jointIndex] * p_ibar;		// p_i = sum(w * M * p_ibar)
				// Sum everything up
				for (int k = 0; k < 3; k++) {
					newMeshVertexPositions[3 * i + k] += product[k];
				}
			}
		}
	}
	else if (mode == "DualQuaternionSkinning") {
		// dual quaternion q = sum(weight_i * q_i)
		// q_i = q0 + epsilon * q1 is dual quaternion for joint i
		// q0 is rotation matrix
		// q1 is translation = 0.5*t*q0
		// t = 2 * q1/q0
		for (int i = 0; i < numMeshVertices; i++) {

			// To store the summed values each quaterion for each joint
			Eigen::Quaterniond sumq0(0.0, 0.0, 0.0, 0.0);
			Eigen::Quaterniond sumq1(0.0, 0.0, 0.0, 0.0);

			// Get dual quaternions for each joint
			for (int j = 0; j < numJointsInfluencingEachVertex; j++) {
				// Dual quaternions for each individual joint
				Eigen::Quaterniond q0;
				Eigen::Quaterniond q1;

				int index = i * numJointsInfluencingEachVertex + j;
				int jointIndex = meshSkinningJoints[index];

				// Get the Rotation Matrix of the joint and save into q0
				Eigen::Matrix3d R;
				for (int row = 0; row < 3; row++) {
					for (int col = 0; col < 3; col++) {
						R(row, col) = jointSkinTransforms[jointIndex][row][col];
					}
				}
				q0 = Eigen::Quaterniond(R);
				if (q0.w() < 0) q0.w() *= -1.0;		// We want positive w value
				
				// Get the translation of the joint as a quaternion
				Vec3d temp = jointSkinTransforms[jointIndex].getTranslation();	// Using Vec3d because Eigen::Vector3d doesn't work
				Eigen::Quaterniond t(0.0, temp[0], temp[1], temp[2]);
				
				// Calculate q1 = (0.5 * t * q0)
				q1 = t * q0;
				q1.coeffs() *= 0.5;

				// Sum = weight of joint * dual quaternion of joint
				sumq0.coeffs() += meshSkinningWeights[index] * q0.coeffs();
				sumq1.coeffs() += meshSkinningWeights[index] * q1.coeffs();
			}

			// final vertex positions = Rx + t, where x is rest vertex position
			Eigen::Matrix3d R(sumq0.toRotationMatrix());
			Eigen::Quaterniond q_t = sumq1 * sumq0.inverse();
			q_t.coeffs() *= 2.0;
			Eigen::Vector3d t(q_t.x(), q_t.y(), q_t.z());
			Eigen::Vector3d x(restMeshVertexPositions[3*i], restMeshVertexPositions[3 * i + 1], restMeshVertexPositions[3 * i + 2]);
			Eigen::Vector3d result = R * x + t;

			// Save the results
			for (int j = 0; j < 3; j++) {
				newMeshVertexPositions[3 * i + j] = result[j];
			}

		}
	}
}

