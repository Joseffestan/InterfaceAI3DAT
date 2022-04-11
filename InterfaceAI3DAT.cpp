/*
 * InterfaceAI3DAT.cpp
 *
 * Copyright (c) 2014-2018 SEACAVE
 *
 * Author(s):
 *
 *      cDc <cdc.seacave@gmail.com>
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * Additional Terms:
 *
 *      You are required to preserve legal notices and author attributions in
 *      that material or in the Appropriate Legal Notices displayed by works
 *      containing it.
 */

#include "../../libs/MVS/Common.h"
#include "../../libs/MVS/Scene.h"
#define _USE_OPENCV
#include "../../libs/MVS/Interface.h"
#include <boost/program_options.hpp>
#include "endian.h"
#include "AI3DAT2MVS.h"
#include "FreeImage.h"

using namespace MVS;

// D E F I N E S ///////////////////////////////////////////////////

#define APPNAME _T("InterfaceAI3DAT")
#define MVS_EXT _T(".mvs")
#define AI3DATA_CAMERAS_NAME  _T("block.ai3dat")
#define AI3DATA_INFO_NAME _T("block.info")
#define AI3DATA_VPC_NAME _T("block.vpc")


// S T R U C T S ///////////////////////////////////////////////////
namespace OPT {
	bool b3Dnovator2AI3DAT; // conversion direction
	bool bNormalizeIntrinsics;
	String strInputFileName;
	String strOutputFileName;
	String strImageFolder;
	unsigned nArchiveType;
	int nProcessPriority;
	unsigned nMaxThreads;
	String strConfigFileName;
	boost::program_options::variables_map vm;
} // namespace OPT

// initialize and parse the command line parameters
bool Initialize(size_t argc, LPCTSTR* argv)
{
	// initialize log and console
	OPEN_LOG();
	OPEN_LOGCONSOLE();

	// group of options allowed only on command line
	boost::program_options::options_description generic("Generic options");
	generic.add_options()
		("help,h", "produce this help message")
		("working-folder,w", boost::program_options::value<std::string>(&WORKING_FOLDER), "working directory (default current directory)")
		("config-file,c", boost::program_options::value<std::string>(&OPT::strConfigFileName)->default_value(APPNAME _T(".cfg")), "file name containing program options")
		("archive-type", boost::program_options::value(&OPT::nArchiveType)->default_value(2), "project archive type: 0-text, 1-binary, 2-compressed binary")
		("process-priority", boost::program_options::value(&OPT::nProcessPriority)->default_value(-1), "process priority (below normal by default)")
		("max-threads", boost::program_options::value(&OPT::nMaxThreads)->default_value(0), "maximum number of threads (0 for using all available cores)")
#if TD_VERBOSE != TD_VERBOSE_OFF
		("verbosity,v", boost::program_options::value(&g_nVerbosityLevel)->default_value(
#if TD_VERBOSE == TD_VERBOSE_DEBUG
			3
#else
			2
#endif
		), "verbosity level")
#endif
		;

	// group of options allowed both on command line and in config file
	boost::program_options::options_description config("Main options");
	config.add_options()
		("input-file,i", boost::program_options::value<std::string>(&OPT::strInputFileName), "input AI3DAT folder containing block.ai3dat and block.vpc")
		("output-file,o", boost::program_options::value<std::string>(&OPT::strOutputFileName), "output filename for storing the MVS project")
		("normalize,f", boost::program_options::value(&OPT::bNormalizeIntrinsics)->default_value(false), "normalize intrinsics while exporting to MVS format")
		;

	boost::program_options::options_description cmdline_options;
	cmdline_options.add(generic).add(config);

	boost::program_options::options_description config_file_options;
	config_file_options.add(config);

	boost::program_options::positional_options_description p;
	p.add("input-file", -1);

	try {
		// parse command line options
		boost::program_options::store(boost::program_options::command_line_parser((int)argc, argv).options(cmdline_options).positional(p).run(), OPT::vm);
		boost::program_options::notify(OPT::vm);
		INIT_WORKING_FOLDER;
		// parse configuration file
		std::ifstream ifs(MAKE_PATH_SAFE(OPT::strConfigFileName));
		if (ifs) {
			boost::program_options::store(parse_config_file(ifs, config_file_options), OPT::vm);
			boost::program_options::notify(OPT::vm);
		}
	}
	catch (const std::exception& e) {
		LOG(e.what());
		return false;
	}

	// initialize the log file
	OPEN_LOGFILE(MAKE_PATH(APPNAME _T("-") + Util::getUniqueName(0) + _T(".log")));

	// print application details: version and command line
	Util::LogBuild();
	LOG(_T("Command line: ") APPNAME _T("%s"), Util::CommandLineToString(argc, argv).c_str());

	// validate input
	Util::ensureValidPath(OPT::strInputFileName);
	const String strInputFileNameExt(Util::getFileExt(OPT::strInputFileName).ToLower());
	OPT::b3Dnovator2AI3DAT = (strInputFileNameExt == MVS_EXT);
	const bool bInvalidCommand(OPT::strInputFileName.empty());
	if (OPT::vm.count("help") || bInvalidCommand) {
		boost::program_options::options_description visible("Available options");
		visible.add(generic).add(config);
		GET_LOG() << _T("\n"
			"Import/export 3D reconstruction from AI3DAT. \n"
			"In order to import a scene, run AI3DAT and next undistort the images (only PINHOLE\n"
			"camera model supported for the moment)."
			"\n")
			<< visible;
	}
	if (bInvalidCommand)
		return false;

	// initialize optional options
	Util::ensureValidFolderPath(OPT::strImageFolder);
	Util::ensureValidPath(OPT::strOutputFileName);
	if (OPT::b3Dnovator2AI3DAT) {
		if (OPT::strOutputFileName.empty())
			OPT::strOutputFileName = Util::getFilePath(OPT::strInputFileName);
	}
	else {
		Util::ensureFolderSlash(OPT::strInputFileName);
		if (OPT::strOutputFileName.empty())
			OPT::strOutputFileName = OPT::strInputFileName + _T("scene") MVS_EXT;
		else
			OPT::strImageFolder = Util::getRelativePath(Util::getFilePath(OPT::strOutputFileName), OPT::strInputFileName + OPT::strImageFolder);
	}

	// initialize global options
	Process::setCurrentProcessPriority((Process::Priority)OPT::nProcessPriority);
#ifdef _USE_OPENMP
	if (OPT::nMaxThreads != 0)
		omp_set_num_threads(OPT::nMaxThreads);
#endif

#ifdef _USE_BREAKPAD
	// start memory dumper
	MiniDumper::Create(APPNAME, WORKING_FOLDER);
#endif

	Util::Init();
	return true;
}

// finalize application instance
void Finalize()
{
#if TD_VERBOSE != TD_VERBOSE_OFF
	// print memory statistics
	Util::LogMemoryInfo();
#endif

	CLOSE_LOGFILE();
	CLOSE_LOGCONSOLE();
	CLOSE_LOG();
}


bool UndistortImage(const std::string Ai3datFileFolder)
{//from AI3D::LoadImageForAI3D() and  AI3D::SaveImage() 
	AI3DAT2MVS AI2M;
	AI2M.ReadAi3datImages(Ai3datFileFolder + "block.ai3dat");
	std::vector<Ai3dImage> Ai3dImages = AI2M.GetAi3dImages();
	for (int i = 0; i < Ai3dImages.size(); ++i)
	{
		//load image
		int cols, rows, bands;
		unsigned char* data = nullptr;
		FREE_IMAGE_FORMAT format = FreeImage_GetFileType(Ai3dImages[i].mImagePath.c_str(), 0);
		FIBITMAP* fi_bitmap = FreeImage_Load(format, Ai3dImages[i].mImagePath.c_str(), 0);

		cols = FreeImage_GetWidth(fi_bitmap);
		rows = FreeImage_GetHeight(fi_bitmap);
		const FREE_IMAGE_COLOR_TYPE color_type = FreeImage_GetColorType(fi_bitmap);
		const bool is_grey = color_type == FIC_MINISBLACK && FreeImage_GetBPP(fi_bitmap) == 8;
		const bool is_rgb = color_type == FIC_RGB && FreeImage_GetBPP(fi_bitmap) == 24;
		bands = is_rgb ? 3 : (is_grey ? 1 : 0);
		if (bands == 0)
			return false;
		data = new unsigned char[cols * rows * bands];
		for (int y = 0; y < rows; ++y)
		{
			unsigned char* line = FreeImage_GetScanLine(fi_bitmap, y);
			for (int x = 0; x < cols; ++x)
			{
				if (is_grey)
					data[y * cols + x] = line[x];
				else
				{
					data[(y * cols + x) * 3 + 2] = line[3 * x + FI_RGBA_RED];
					data[(y * cols + x) * 3 + 1] = line[3 * x + FI_RGBA_GREEN];
					data[(y * cols + x) * 3] = line[3 * x + FI_RGBA_BLUE];
				}
			}
		}
		FreeImage_Unload(fi_bitmap);
		// reverse upside down
		unsigned char* reversed_data = new unsigned char[cols * rows * bands];
		if (bands == 3)
		{
			for (int r = 0; r < rows; r++)
			{
				for (int c = 0; c < cols; c++)
				{
					int idx_pixel1 = r * cols + c;
					int idx_pixel2 = (rows - r - 1) * cols + c;
					for (int b = 0; b < 3; b++)
						reversed_data[idx_pixel1 * 3 + b] = data[idx_pixel2 * 3 + b];
				}
			}
		}
		else
		{
			for (int r = 0; r < rows; r++)
			{
				for (int c = 0; c < cols; c++)
				{
					int idx_pixel1 = r * cols + c;
					int idx_pixel2 = (rows - r - 1) * cols + c;
					for (int b = 0; b < 3; b++)
						reversed_data[idx_pixel1 * 3 + b] = data[idx_pixel2];
				}
			}
		}
		//undistort
		AI3D::UndistortImage(Ai3dImages[i].mDistortedParameterVector, Ai3dImages[i].mUndistortedParameterVector, reversed_data);
		int undistored_cols = Ai3dImages[i].mUndistortedParameterVector[0];
		int undistored_rows = Ai3dImages[i].mUndistortedParameterVector[1];
		// reverse again
		unsigned char* reversed_data2 = new unsigned char[undistored_cols * undistored_rows * bands];
		if (bands == 3)
		{
			for (int r = 0; r < undistored_rows; r++)
			{
				for (int c = 0; c < undistored_cols; c++)
				{
					int idx_pixel1 = r * undistored_cols + c;
					int idx_pixel2 = (undistored_rows - r - 1) * undistored_cols + c;
					for (int b = 0; b < 3; b++)
						reversed_data2[idx_pixel1 * 3 + b] = reversed_data[idx_pixel2 * 3 + b];
				}
			}
		}
		else
		{
			for (int r = 0; r < undistored_rows; r++)
			{
				for (int c = 0; c < undistored_cols; c++)
				{
					int idx_pixel1 = r * undistored_cols + c;
					int idx_pixel2 = (rows - r - 1) * undistored_cols + c;
					for (int b = 0; b < 3; b++)
						reversed_data2[idx_pixel1 * 3 + b] = reversed_data[idx_pixel2];
				}
			}
		}
		//save
		if (undistored_cols < 1 || undistored_rows < 1)
			return false;

		FIBITMAP* fi_bitmap_output = nullptr;
		if (is_grey)
		{
			const int nums_bits_per_pixel = 8;
			fi_bitmap_output = FreeImage_Allocate(undistored_cols, undistored_rows, nums_bits_per_pixel);
		}
		else
		{
			const int nums_bits_per_pixel = 24;
			fi_bitmap_output = FreeImage_Allocate(undistored_cols, undistored_rows, nums_bits_per_pixel);
		}
		for (int y = 0; y < undistored_rows; ++y)
		{
			unsigned char* line = FreeImage_GetScanLine(fi_bitmap_output, y);
			for (int x = 0; x < undistored_cols; ++x)
			{
				if (is_grey)
					line[x] = reversed_data2[y * undistored_cols + x];
				else
				{
					line[3 * x + FI_RGBA_RED] = reversed_data2[(y * undistored_cols + x) * 3 + 2];
					line[3 * x + FI_RGBA_GREEN] = reversed_data2[(y * undistored_cols + x) * 3 + 1];
					line[3 * x + FI_RGBA_BLUE] = reversed_data2[(y * undistored_cols + x) * 3];
				}
			}
		}
		FREE_IMAGE_FORMAT save_format = FIF_JPEG;

		std::string SavePath = Ai3dImages[i].mUndistortedImagePath;		bool success = FreeImage_Save(save_format, fi_bitmap_output, SavePath.c_str());
		if (!success)
		{
			std::cout << "save undistort image failed at: " << Ai3dImages[i].mUndistortedImagePath.c_str() << std::endl;
			return false;
		}
		FreeImage_Unload(fi_bitmap_output);

		delete[]data;
		data = nullptr;
		delete[] reversed_data2;
		reversed_data2 = nullptr;
		delete[]reversed_data;
		reversed_data = nullptr;
	}
	return true;
}

bool CutImage(const std::string Ai3datFileFolder)
{
	AI3DAT2MVS AI2M;
	AI2M.ReadAi3datImages(Ai3datFileFolder + "block.ai3dat");
	std::vector<Ai3dImage> Ai3dImages = AI2M.GetAi3dImages();
	for (int i = 0; i < Ai3dImages.size(); ++i)
	{
		cv::Mat UndistortedImage = cv::imread(Ai3dImages[i].mUndistortedImagePath, 1);
		cv::Rect select;
		select = cv::Rect(Ai3dImages[i].mRect.x, Ai3dImages[i].mRect.y, Ai3dImages[i].mRect.width, Ai3dImages[i].mRect.height);
		cv::Mat CuttedImage = UndistortedImage(select);
		bool success = cv::imwrite(Ai3dImages[i].mCuttedImagePath, CuttedImage);
		if (!success)
		{
			std::cout << "save cutted image failed at: " << Ai3dImages[i].mCuttedImagePath.c_str() << std::endl;
			return false;
		}
	}
	return 1;
}

bool ImportSceneFromAI3DAT(const String& strFolder, Interface& scene)
{
	AI3DAT2MVS AI2M;
	LOG_OUT() << "Reading ai3d cameras: " << strFolder << "block.ai3dat" << std::endl;
	AI2M.ReadAi3datImages(strFolder + "block.ai3dat");
	LOG_OUT() << "Reading ai3d points: " << strFolder << "block.vpc" << std::endl;
	AI2M.ReadAi3datPointCloud(strFolder + "block.vpc");
	std::vector<Ai3dImage> Ai3dImages = AI2M.GetAi3dImages();
	std::vector<Ai3dPoint> Ai3dPoints = AI2M.GetAi3dPoints();
	std::vector<std::vector<unsigned short>> Ai3dPointViews = AI2M.GetAi3dPointViews();
	for (int i = 0; i < Ai3dImages.size(); ++i)
	{
		//read images
		Interface::Image image;
		//image.name = Ai3dImages[i].mImagePath;
		//image.name = Ai3dImages[i].mUndistortedImagePath;
		image.name = Ai3dImages[i].mCuttedImagePath;
		image.cameraID = 0;
		image.platformID = i;
		image.ID = i;
		image.poseID = 0;
		scene.images.push_back(image);

		//read cameras
		Interface::Platform platform;
		platform.name = String::FormatString(_T("platform%03u"), i);
		Interface::Platform::Camera camera;
		camera.name = "PINHOLE";  //colmapCamera.model;
		camera.K = Interface::Mat33d::eye();
		camera.R = Interface::Mat33d::eye();

		Interface::Platform::Pose pose;
		pose.C = Interface::Pos3d(Ai3dImages[i].mAi3dCamera.C[0], Ai3dImages[i].mAi3dCamera.C[1], Ai3dImages[i].mAi3dCamera.C[2]);

		for (int m = 0; m < 3; ++m)
		{
			for (int n = 0; n < 3; ++n)
			{
				camera.K(m, n) = Ai3dImages[i].mAi3dCamera.K[m * 3 + n];
				pose.R(m, n) = Ai3dImages[i].mAi3dCamera.R[m * 3 + n];
			}
		}
		EnsureRotationMatrix((Matrix3x3d&)pose.R);
		uint32_t width = Ai3dImages[i].mCols;
		uint32_t height = Ai3dImages[i].mRows;
		if (OPT::bNormalizeIntrinsics) {
			// normalize camera intrinsics
			const REAL fScale(REAL(1) / Camera::GetNormalizationScale(width, height));
			camera.K(0, 0) *= fScale;
			camera.K(1, 1) *= fScale;
			camera.K(0, 2) *= fScale;
			camera.K(1, 2) *= fScale;
		}
		else {
			camera.width = width;
			camera.height = height;
		}
		platform.cameras.push_back(camera);
		platform.poses.push_back(pose);
		scene.platforms.push_back(platform);
	}

	if (Ai3dPoints.size() != Ai3dPointViews.size())
	{
		LOG_OUT() << "pointcloud and visibility have different size." << std::endl;
		return false;
	}
	for (int i = 0; i < Ai3dPoints.size(); ++i)
	{
		//read points
		Interface::Vertex vertex;
		vertex.X.x = Ai3dPoints[i].x;
		vertex.X.y = Ai3dPoints[i].y;
		vertex.X.z = Ai3dPoints[i].z;
		for (int j = 0; j < Ai3dPointViews[i].size(); ++j)
		{
			Interface::Vertex::View view;
			view.imageID = Ai3dPointViews[i].at(j);
			view.confidence = 0;
			vertex.views.emplace_back(view);
		}
		std::sort(vertex.views.begin(), vertex.views.end(),
			[](const Interface::Vertex::View& view0, const Interface::Vertex::View& view1) { return view0.imageID < view1.imageID; });
		if (vertex.views.size() > 1)
			scene.vertices.emplace_back(std::move(vertex));
	}
	return true;
}

int main(int argc, LPCTSTR* argv)
{
#ifdef _DEBUGINFO
	// set _crtBreakAlloc index to stop in <dbgheap.c> at allocation
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);// | _CRTDBG_CHECK_ALWAYS_DF);
#endif

	if (!Initialize(argc, argv))
		return EXIT_FAILURE;
	TD_TIMER_START();

	//read AI3DAT input data
	Interface scene;
	if (!ImportSceneFromAI3DAT(MAKE_PATH_SAFE(OPT::strInputFileName), scene))
		return EXIT_FAILURE;

	//UndistortImage(OPT::strInputFileName);
	//CutImage(OPT::strInputFileName);

	// write MVS input data
	Util::ensureFolder(Util::getFullPath(MAKE_PATH_FULL(WORKING_FOLDER_FULL, OPT::strOutputFileName)));
	if (!ARCHIVE::SerializeSave(scene, MAKE_PATH_SAFE(OPT::strOutputFileName), (uint32_t)OPT::bNormalizeIntrinsics ? 0 : 1))
		return EXIT_FAILURE;
	VERBOSE("Exported data: %u images & %u vertices (%s)", scene.images.size(), scene.vertices.size(), TD_TIMER_GET_FMT().c_str());


	Finalize();
	return EXIT_SUCCESS;
}
/*----------------------------------------------------------------*/
