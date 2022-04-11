#pragma once
#ifndef KERNEL_CUH
#define KERNEL_CUH
#include <vector>

namespace AI3D
{
	void UndistortCamera(std::vector<float>& distored_camera_parameter_vector, std::vector<float>& undistored_camera_parameter_vector);
	void UndistortImage(std::vector<float>& distored_camera_parameter_vector, std::vector<float>& undistored_camera_parameter_vector, unsigned char*& image_data);

}
#endif