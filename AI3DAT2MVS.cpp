#include "AI3DAT2MVS.h"



using namespace std;

AI3DAT2MVS::AI3DAT2MVS()
{
}

AI3DAT2MVS::~AI3DAT2MVS()
{
}

// compose P from K, R and C
void AI3DAT2MVS::ComposeP(Ai3dCamera& ai3dcamera)
{
	Eigen::Matrix<float, 3, 4, Eigen::RowMajor> _P(Eigen::Matrix<float, 3, 4, Eigen::RowMajor>::Zero(3, 4));
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> _K = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(ai3dcamera.K, 3, 3);
	Eigen::Matrix<float, 3, 3, Eigen::RowMajor> _R = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(ai3dcamera.R, 3, 3);
	_P(0, 0) = 1;
	_P(1, 1) = 1;
	_P(2, 2) = 1;
	_P(0, 3) = -ai3dcamera.C[0];
	_P(1, 3) = -ai3dcamera.C[1];
	_P(2, 3) = -ai3dcamera.C[2];
	_P = _K * _R * _P;
	memcpy(ai3dcamera.P, _P.data(), sizeof(float) * 12);
}

void AI3DAT2MVS::ComposeK(Ai3dCamera& ai3dcamera, float focal_lenth_x, float focal_lenth_y, float width, float height)
{
	ai3dcamera.K[0] = focal_lenth_x;
	ai3dcamera.K[1] = float(0);
	ai3dcamera.K[2] = float(0.5) * (width - 1);

	ai3dcamera.K[3] = float(0);
	ai3dcamera.K[4] = focal_lenth_y;
	ai3dcamera.K[5] = float(0.5) * (height - 1);

	ai3dcamera.K[6] = float(0);
	ai3dcamera.K[7] = float(0);
	ai3dcamera.K[8] = float(1);
}

void AI3DAT2MVS::SetDistortParameter(Ai3dImage& AiImage, vector<float>& distorted_parameter_vector)
{
	AiImage.mDistortedParameterVector = distorted_parameter_vector;
	AiImage.mUndistortedParameterVector.resize(6);
	AI3D::UndistortCamera(distorted_parameter_vector, AiImage.mUndistortedParameterVector);
	AiImage.mOriCols = AiImage.mUndistortedParameterVector[0];
	AiImage.mOriRows = AiImage.mUndistortedParameterVector[1];
	AiImage.mCols = AiImage.mOriCols;
	AiImage.mRows = AiImage.mOriRows;
	/*AiImage.mRect.x = 0;
	AiImage.mRect.y = 0;
	AiImage.mRect.width = AiImage.mCols;
	AiImage.mRect.height = AiImage.mRows;*/
	ComposeK(AiImage.mAi3dCamera, AiImage.mUndistortedParameterVector[2], AiImage.mUndistortedParameterVector[2], float(AiImage.mCols), float(AiImage.mRows));
}

void AI3DAT2MVS::UpdateCamera(Ai3dImage& AiImage)
{
	AiImage.mAi3dCamera.K[0] = AiImage.mOriAi3dCamera.K[0] * AiImage.mScale;
	AiImage.mAi3dCamera.K[4] = AiImage.mOriAi3dCamera.K[4] * AiImage.mScale;
	AiImage.mCols = int(AiImage.mRect.width * AiImage.mScale);
	AiImage.mRows = int(AiImage.mRect.height * AiImage.mScale);
	AiImage.mAi3dCamera.K[2] = (AiImage.mOriAi3dCamera.K[2] - AiImage.mRect.x) * (AiImage.mCols - 1) / (AiImage.mRect.width - 1);
	AiImage.mAi3dCamera.K[5] = (AiImage.mOriAi3dCamera.K[5] - AiImage.mRect.y) * (AiImage.mRows - 1) / (AiImage.mRect.height - 1);
	ComposeP(AiImage.mAi3dCamera);
}

//read .ai3dat input data
void AI3DAT2MVS::ReadAi3datImages(const string FilePath)
{
	ifstream Ai3datStream(FilePath);
	if (!Ai3datStream)
	{
		cout << "failed to open " << FilePath << " \n";
		return;
	}
	int ImageNum;
	Ai3datStream >> ImageNum;
	if (ImageNum < 1)
	{
		cout << "no image.\n";
		return;
	}
	m_ai3dImages.resize(ImageNum);
	for (int i = 0; i < ImageNum; ++i)
	{
		Ai3dImage& aimage = m_ai3dImages[i];
		aimage.mFlag = 1;
		aimage.mScale = 1.0;
		Ai3datStream >> aimage.mIsValid;
		if (!aimage.mIsValid)
		{
			aimage.mIsValid = 0;
			continue;
		}
		Ai3datStream >> aimage.mImagePath;
		int flag = aimage.mImagePath.find("images");
		aimage.mUndistortedImagePath = aimage.mImagePath;
		aimage.mUndistortedImagePath.replace(aimage.mUndistortedImagePath.begin() + flag, aimage.mUndistortedImagePath.begin() + flag + 6, "UndistortedImages");
		aimage.mCuttedImagePath = aimage.mImagePath;
		aimage.mCuttedImagePath.replace(aimage.mCuttedImagePath.begin() + flag, aimage.mCuttedImagePath.begin() + flag + 6, "CuttedImages");

		Ai3datStream >> aimage.mOriCols >> aimage.mOriRows;
		Ai3dCamera camera;
		Ai3datStream >> camera.C[0] >> camera.C[1] >> camera.C[2];
		Ai3datStream >> camera.R[0] >> camera.R[1] >> camera.R[2];
		Ai3datStream >> camera.R[3] >> camera.R[4] >> camera.R[5];
		Ai3datStream >> camera.R[6] >> camera.R[7] >> camera.R[8];
		Ai3datStream >> camera.K[0] >> camera.K[1] >> camera.K[2];
		Ai3datStream >> camera.K[3] >> camera.K[4] >> camera.K[5];
		Ai3datStream >> camera.K[6] >> camera.K[7] >> camera.K[8];
		ComposeP(camera);
		aimage.mAi3dCamera = camera;
		aimage.mOriAi3dCamera = camera;
		Recti rect;
		Ai3datStream >> rect.x >> rect.y >> rect.width >> rect.height;
		vector<float> distort_parameter(10);
		vector<float> undistort_parameter(5);
		Ai3datStream >> distort_parameter[0] >> distort_parameter[1] >> distort_parameter[2] >> distort_parameter[3] >>
			distort_parameter[4] >> distort_parameter[5] >> distort_parameter[6] >> distort_parameter[7] >>
			distort_parameter[8] >> distort_parameter[9];
		Ai3datStream >> undistort_parameter[0] >> undistort_parameter[1] >> undistort_parameter[2] >>
			undistort_parameter[3] >> undistort_parameter[4];
		SetDistortParameter(aimage, distort_parameter);
		aimage.mRect = rect;
		UpdateCamera(aimage);

	}

	Ai3datStream.close();
}

//read .vpc input pointcloud
void AI3DAT2MVS::ReadAi3datPointCloud(const std::string FilePath)
{
	string temp(FilePath);
	temp = temp.substr(temp.length() - 3, 3);
	if (temp.compare("vpc") != 0)
	{
		cout << "not a .vpc file.\n";
		return;
	}
	ifstream ifs(FilePath, ios::binary);
	if (!ifs.is_open())
	{
		cout << "failed to open .vpc file.\n";
		return;
	}
	unsigned int n_points(0);
	ifs.read((char*)(&n_points), sizeof(unsigned int));
	if (n_points < 1)
	{
		ifs.close();
		cout << "pointcloud empty.\n";
		return;
	}
	m_ai3dPoints.reserve(n_points);
	m_ai3dPointViews.resize(n_points);

	float x, y, z;
	unsigned int n_views(0);
	unsigned short view_id(0);
	for (int i = 0; i < n_points; ++i)
	{
		ifs.read((char*)(&x), sizeof(float));
		ifs.read((char*)(&y), sizeof(float));
		ifs.read((char*)(&z), sizeof(float));
		Ai3dPoint pt;
		pt.x = x; pt.y = y; pt.z = z;
		m_ai3dPoints.push_back(pt);

		std::vector<unsigned short>& views = m_ai3dPointViews[i];
		ifs.read((char*)(&n_views), sizeof(unsigned int));
		views.reserve(n_views);
		for (int j = 0; j < n_views; j++)
		{
			ifs.read((char*)(&view_id), sizeof(unsigned short));
			views.push_back(view_id);
		}
	}
	ifs.close();
	return;
}

vector<Ai3dImage> AI3DAT2MVS::GetAi3dImages()
{
	return m_ai3dImages;
}

std::vector<Ai3dPoint> AI3DAT2MVS::GetAi3dPoints()
{
	return m_ai3dPoints;
}

std::vector<std::vector<unsigned short>> AI3DAT2MVS::GetAi3dPointViews()
{
	return m_ai3dPointViews;
}

//write .mvs input data
void AI3DAT2MVS::ExportMVSModel(const string FilePath)
{

}
