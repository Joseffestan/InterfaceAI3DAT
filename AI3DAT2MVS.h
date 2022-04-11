#pragma once
#ifndef _AI3DAT2MVS_H_
#define _AI3DAT2MVS_H_

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include "kernel.cuh"


struct Ai3dCamera {
	float K[9];	// the intrinsic camera parameters (3x3)
	float R[9];	// rotation (3x3) and
	float C[3];	// translation (3,1), the extrinsic camera parameters
	float P[12];// the composed projection matrix (3x4)
};

struct Recti {
	int x;
	int y;
	int width;
	int height;
};

struct Ai3dImage {
	bool mIsValid;
	std::string  mImagePath;
	std::string mUndistortedImagePath;
	std::string mCuttedImagePath;
	int mOriCols;
	int mOriRows;
	Ai3dCamera mAi3dCamera;
	Ai3dCamera mOriAi3dCamera;

	unsigned int mCols;
	unsigned int mRows;
	unsigned int mBands;
	float		 mAvgDepth; // average depth of the points seen by this camera
	float		 mScale;
	Recti		 mRect;
	Recti        mOriRect;
	bool		 mFlag;
	int			 mGlobalId;
	//NeighborArr  mNeighbors;
	std::vector<float> mDistortedParameterVector;
	std::vector<float> mUndistortedParameterVector;
};

struct Ai3dPoint {
	float x;
	float y;
	float z;
};

class AI3DAT2MVS
{
private:
	struct std::vector<Ai3dImage> m_ai3dImages;
	struct std::vector<Ai3dPoint> m_ai3dPoints;
	std::vector<std::vector<unsigned short>> m_ai3dPointViews;



public:
	AI3DAT2MVS();
	~AI3DAT2MVS();

	void ComposeP(Ai3dCamera& ai3dcamera);
	void ComposeK(Ai3dCamera& ai3dcamera, float focal_lenth_x, float focal_lenth_y, float width, float height);
	void SetDistortParameter(Ai3dImage& AiImage, std::vector<float>& distorted_parameter_vector);
	void UpdateCamera(Ai3dImage& AiImage);

	void ReadAi3datImages(const std::string FilePath);
	void ReadAi3datPointCloud(const std::string FilePath);
	std::vector<Ai3dImage> GetAi3dImages();
	std::vector<Ai3dPoint> GetAi3dPoints();
	std::vector<std::vector<unsigned short>> GetAi3dPointViews();

	void ExportMVSModel(const std::string FilePath);
};


#endif // _AI3DAT2MVS_H_