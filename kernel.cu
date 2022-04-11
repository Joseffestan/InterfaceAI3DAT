//#include "ai3d_log.h"
#include "kernel.cuh"
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/transform_scan.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>
#include <float.h>
#include <fstream>
#include <iostream>

__device__ __constant__ float guass_kernel[9];
namespace AI3D
{
	__forceinline__ bool CudaSynchronize(const char* str_function_name)
	{
		cudaError_t cudaStatus;
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching %s!\n", cudaStatus, str_function_name);
			return false;
		}
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "%s launch failed: %s\n", cudaGetErrorString(cudaStatus), str_function_name);
			return false;
		}
		return true;
	}



	/******************************************影像纠正畸变*****************************************************/
//SimpleRadialCameraModel "f, cx, cy, k";
	__host__ __device__ void Distortion4SimpleRadialCameraModel(const float* params, const float u, const float v, float& du, float& dv)
	{
		const float k = params[5];

		const float u2 = u * u;
		const float v2 = v * v;
		const float r2 = u2 + v2;
		const float radial = k * r2;
		du = u * radial;
		dv = v * radial;
	}

	__host__ __device__ void IterativeUndistortion4SimpleRadialCameraModel(const float* params, float& u, float& v)
	{
		// Number of iterations for iterative undistortion, 100 should be enough
		// even for complex camera models with higher order terms.
		const int kNumUndistortionIterations = 100;
		const double kUndistortionEpsilon = 1e-10;

		float uu = u;
		float vv = v;
		float du;
		float dv;

		for (int i = 0; i < kNumUndistortionIterations; ++i)
		{
			Distortion4SimpleRadialCameraModel(params, uu, vv, du, dv);
			const float uu_prev = uu;
			const float vv_prev = vv;
			uu = u - du;
			vv = v - dv;
			if (std::abs(uu_prev - uu) < kUndistortionEpsilon &&
				std::abs(vv_prev - vv) < kUndistortionEpsilon) {
				break;
			}
		}

		u = uu;
		v = vv;
	}

	__host__ __device__ void WorldToImage4SimpleRadialCameraModel(const float* params, const float u, const float v, float& x, float& y)
	{
		const float f = params[2];
		const float c1 = params[3];
		const float c2 = params[4];

		// Distortion
		float du, dv;
		Distortion4SimpleRadialCameraModel(params, u, v, du, dv);
		x = u + du;
		y = v + dv;
		// Transform to image coordinates
		x = f * x + c1;
		y = f * y + c2;
	}

	__host__ __device__ void ImageToWorld4SimpleRadialCameraModel(const float* params, const float x, const float y, float& u, float& v)
	{
		const float f = params[2];
		const float c1 = params[3];
		const float c2 = params[4];

		// Lift points to normalized plane
		u = (x - c1) / f;
		v = (y - c2) / f;

		IterativeUndistortion4SimpleRadialCameraModel(params, u, v);
	}

	//OpenCVCameraModel "fx, fy, cx, cy, k1, k2, p1, p2"
	__host__ __device__ void Distortion4OpenCVCameraModel(const float* params, const float u, const float v, float& du, float& dv)
	{
		const float k1 = params[6];
		const float k2 = params[7];
		const float p1 = params[8];
		const float p2 = params[9];

		const float u2 = u * u;
		const float uv = u * v;
		const float v2 = v * v;
		const float r2 = u2 + v2;
		const float radial = k1 * r2 + k2 * r2 * r2;
		du = u * radial + 2.f * p1 * uv + p2 * (r2 + 2.f * u2);
		dv = v * radial + 2.f * p2 * uv + p1 * (r2 + 2.f * v2);
	}

	__host__ __device__ void IterativeUndistortion4OpenCVCameraModel(const float* params, float& u, float& v)
	{
		// Number of iterations for iterative undistortion, 100 should be enough
		// even for complex camera models with higher order terms.
		const int kNumUndistortionIterations = 100;
		const double kUndistortionEpsilon = 1e-10;

		float uu = u;
		float vv = v;
		float du;
		float dv;

		for (int i = 0; i < kNumUndistortionIterations; ++i)
		{
			Distortion4OpenCVCameraModel(params, uu, vv, du, dv);
			const float uu_prev = uu;
			const float vv_prev = vv;
			uu = u - du;
			vv = v - dv;
			if (std::abs(uu_prev - uu) < kUndistortionEpsilon &&
				std::abs(vv_prev - vv) < kUndistortionEpsilon) {
				break;
			}
		}

		u = uu;
		v = vv;
	}

	__host__ __device__ void ImageToWorld4OpenCVCameraModel(const float* params, const float x, const float y, float& u, float& v)
	{
		const float f1 = params[2];
		const float f2 = params[3];
		const float c1 = params[4];
		const float c2 = params[5];

		u = (x - c1) / f1;
		v = (y - c2) / f2;

		IterativeUndistortion4OpenCVCameraModel(params, u, v);
	}

	__host__ __device__ void WorldToImage4OpenCVCameraModel(const float* params, const float u, const float v, float& x, float& y)
	{
		const float f1 = params[2];
		const float f2 = params[3];
		const float c1 = params[4];
		const float c2 = params[5];

		// Distortion
		float du, dv;
		Distortion4OpenCVCameraModel(params, u, v, du, dv);
		x = u + du;
		y = v + dv;

		// Transform to image coordinates
		x = f1 * x + c1;
		y = f2 * y + c2;
	}


	/*__global__ void KerUndistort4OpenCVCameraModel(//reverse image in this kernel
		const float* distored_camera_parameter_array,
		const float* undistored_camera_parameter_array,
		cudaTextureObject_t distorted_image_texture_object,
		cudaSurfaceObject_t undistorted_image_surface_object
	)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int cols_undistored = undistored_camera_parameter_array[0];
		int rows_undistored = undistored_camera_parameter_array[1];
		if (x < 0 || x > cols_undistored - 1 || y < 0 || y > rows_undistored - 1)
			return;

		float u, v;
		ImageToWorld4SimpleRadialCameraModel(undistored_camera_parameter_array, (float)x + 0.5f, (float)y + 0.5f, u, v);

		float x_new, y_new;
		WorldToImage4OpenCVCameraModel(distored_camera_parameter_array, u, v, x_new, y_new);

		int cols_distored = distored_camera_parameter_array[0];
		int rows_distored = distored_camera_parameter_array[1];

		const int x_tmp = floor(x_new);
		const int y_tmp = floor(rows_distored - 1 - y_new);
		if (x_tmp < 0 || x_tmp > cols_distored - 1 || y_tmp < 0 || y_tmp > rows_distored - 1)
		{
			uchar4 uchar4_zero = { 0,0,0,0 };
			surf2Dwrite(uchar4_zero, undistorted_image_surface_object, x * 4, rows_undistored - y - 1, cudaBoundaryModeZero);
			return;
		}

		//插值的时候，有自带的往左上方的(0.5,0.5)坐标偏移的,因此需要向右下方移动0.5
		float4 color_f = tex2D<float4>(distorted_image_texture_object, x_new, (float)rows_distored - y_new);
		uchar4 color = { round(color_f.x * 255.f), round(color_f.y * 255.f), round(color_f.z * 255.f), round(color_f.w * 255.f) };
		//float4 color = tex2D(refTex1, x_new - 0.5f, rows_distored - y_new - 1.5f);
		surf2Dwrite(color, undistorted_image_surface_object, x * 4, rows_undistored - y - 1, cudaBoundaryModeZero);
	}*/


	__global__ void NewKerUndistort4OpenCVCameraModel(
		const float* distored_camera_parameter_array,
		const float* undistored_camera_parameter_array,
		cudaTextureObject_t distorted_image_texture_object,
		cudaSurfaceObject_t undistorted_image_surface_object
	)
	{
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int cols_undistored = undistored_camera_parameter_array[0];
		int rows_undistored = undistored_camera_parameter_array[1];
		if (x < 0 || x > cols_undistored - 1 || y < 0 || y > rows_undistored - 1)
			return;

		float u, v;
		ImageToWorld4SimpleRadialCameraModel(undistored_camera_parameter_array, (float)x + 0.5f, (float)y + 0.5f, u, v);

		float x_new, y_new;
		WorldToImage4OpenCVCameraModel(distored_camera_parameter_array, u, v, x_new, y_new);

		int cols_distored = distored_camera_parameter_array[0];
		int rows_distored = distored_camera_parameter_array[1];

		const int x_tmp = floor(x_new);
		const int y_tmp = floor(y_new);
		if (x_tmp < 0 || x_tmp > cols_distored - 1 || y_tmp < 0 || y_tmp > rows_distored - 1)
		{
			uchar4 uchar4_zero = { 0,0,0,0 };
			surf2Dwrite(uchar4_zero, undistorted_image_surface_object, x * 4, y, cudaBoundaryModeZero);
			return;
		}

		//插值的时候，有自带的往左上方的(0.5,0.5)坐标偏移的,因此需要向右下方移动0.5
		//参考COLMAP函数WarpImageBetweenCameras：
		// for (int y = 0; y < target_image->Height(); ++y) {
		// image_point.y() = y + 0.5;
		// for (int x = 0; x < target_image->Width(); ++x) {
		//   image_point.x() = x + 0.5;
		//   // Camera models assume that the upper left pixel center is (0.5, 0.5).
		//   const Eigen::Vector2d world_point =
		//       scaled_target_camera.ImageToWorld(image_point);
		//   const Eigen::Vector2d source_point =
		//       source_camera.WorldToImage(world_point);
		//   BitmapColor<float> color;
		//   if (source_image.InterpolateBilinear(source_point.x() - 0.5,
		//                                        source_point.y() - 0.5, &color)) {
		//     target_image->SetPixel(x, y, color.Cast<uint8_t>());
		//   } else {
		//     target_image->SetPixel(x, y, BitmapColor<uint8_t>(0));
		//   }
		//在计算图像坐标转换之前，它先将图像坐标+0.5，从源图像上采样的坐标之前再-0.5
		//再参考https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching
		//又因为tex2D会自动地减去0.5的坐标，所以这里就不需要再减了。
		float4 color_f = tex2D<float4>(distorted_image_texture_object, float(x_new) - 0.5 + 0.5, float(y_new) - 0.5 + 0.5);
		uchar4 color = { round(color_f.x * 255.f), round(color_f.y * 255.f), round(color_f.z * 255.f), round(color_f.w * 255.f) };

		surf2Dwrite(color, undistorted_image_surface_object, x * 4, y, cudaBoundaryModeZero);
	}

	void Undistort4OpenCVCameraModel(std::vector<float>& distored_camera_parameter_vector, std::vector<float>& undistored_camera_parameter_vector, unsigned char*& image_data)
	{
		//开辟显存空间
		int distored_cols = distored_camera_parameter_vector[0];
		int distored_rows = distored_camera_parameter_vector[1];

		int undistored_cols = undistored_camera_parameter_vector[0];
		int undistored_rows = undistored_camera_parameter_vector[1];

		float* distored_camera_parameter_array;
		cudaMalloc((void**)&distored_camera_parameter_array, distored_camera_parameter_vector.size() * sizeof(float));
		cudaMemcpy(distored_camera_parameter_array, distored_camera_parameter_vector.data(),
			distored_camera_parameter_vector.size() * sizeof(float), cudaMemcpyHostToDevice);

		float* undistored_camera_parameter_array;
		cudaMalloc((void**)&undistored_camera_parameter_array, undistored_camera_parameter_vector.size() * sizeof(float));
		cudaMemcpy(undistored_camera_parameter_array, undistored_camera_parameter_vector.data(),
			undistored_camera_parameter_vector.size() * sizeof(float), cudaMemcpyHostToDevice);


		cudaArray_t distorted_image_cuda_array;
		cudaArray_t undistorted_image_cuda_array;
		cudaTextureObject_t distorted_image_texture_object;
		cudaSurfaceObject_t undistorted_image_surface_object;

		uchar4* image_data_host = new uchar4[distored_cols * distored_rows];
		for (int i = 0; i < distored_cols * distored_rows; i++)
		{
			image_data_host[i].x = image_data[i * 3];
			image_data_host[i].y = image_data[i * 3 + 1];
			image_data_host[i].z = image_data[i * 3 + 2];
			image_data_host[i].w = 0;
		}

		cudaChannelFormatDesc uchar4_channel_desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		cudaMallocArray(&distorted_image_cuda_array, &uchar4_channel_desc, distored_cols, distored_rows);
		cudaMallocArray(&undistorted_image_cuda_array, &uchar4_channel_desc, undistored_cols, undistored_rows);
		cudaMemcpy2DToArray(distorted_image_cuda_array, 0, 0, image_data_host, distored_cols * 4, distored_cols * 4, distored_rows, cudaMemcpyHostToDevice);
		delete[] image_data_host;

		struct cudaTextureDesc texture_desc;
		memset(&texture_desc, 0, sizeof(texture_desc));
		texture_desc.addressMode[0] = cudaAddressModeClamp;
		texture_desc.addressMode[1] = cudaAddressModeClamp;
		texture_desc.filterMode = cudaFilterModeLinear;
		texture_desc.readMode = cudaReadModeNormalizedFloat;
		texture_desc.normalizedCoords = false;

		struct cudaResourceDesc resource_desc;
		memset(&resource_desc, 0, sizeof(resource_desc));
		resource_desc.resType = cudaResourceTypeArray;

		resource_desc.res.array.array = distorted_image_cuda_array;
		cudaCreateTextureObject(&distorted_image_texture_object, &resource_desc, &texture_desc, NULL);

		resource_desc.res.array.array = undistorted_image_cuda_array;
		cudaCreateSurfaceObject(&undistorted_image_surface_object, &resource_desc);

		//影像纠正畸变
		dim3 threads(16, 16);
		dim3 grids((undistored_cols + threads.x - 1) / threads.x, (undistored_rows + threads.y - 1) / threads.y);
		NewKerUndistort4OpenCVCameraModel << <grids, threads >> > (
			distored_camera_parameter_array,
			undistored_camera_parameter_array,
			distorted_image_texture_object,
			undistorted_image_surface_object
			);
		if (!CudaSynchronize("KerUndistort4OpenCVCameraModel"))
			return;

		uchar4* undistored_image_data_host = new uchar4[undistored_cols * undistored_rows];
		cudaMemcpy2DFromArray(undistored_image_data_host, undistored_cols * 4, undistorted_image_cuda_array,
			0, 0, undistored_cols * 4, undistored_rows, cudaMemcpyDeviceToHost);
		delete[] image_data;
		image_data = new unsigned char[undistored_cols * undistored_rows * 3];
		for (int i = 0; i < undistored_cols * undistored_rows; i++)
		{
			image_data[i * 3] = undistored_image_data_host[i].x;
			image_data[i * 3 + 1] = undistored_image_data_host[i].y;
			image_data[i * 3 + 2] = undistored_image_data_host[i].z;
		}

		//释放显存
		cudaDestroySurfaceObject(distorted_image_texture_object);
		cudaDestroySurfaceObject(undistorted_image_surface_object);
		cudaFreeArray(distorted_image_cuda_array);
		cudaFreeArray(undistorted_image_cuda_array);
		cudaFree(distored_camera_parameter_array);
		cudaFree(undistored_camera_parameter_array);
		delete[] undistored_image_data_host;
	}

	float Clip(const float& value, const float& low, const float& high)
	{
		float tmp = value < high ? value : high;
		float result = tmp > low ? tmp : low;
		return result;
	}

	void UndistortCamera(std::vector<float>& source_camera_parameter_vector, std::vector<float>& target_camera_parameter_vector)
	{
		int cols = source_camera_parameter_vector[0];
		int rows = source_camera_parameter_vector[1];
		target_camera_parameter_vector[0] = source_camera_parameter_vector[0]; //cols
		target_camera_parameter_vector[1] = source_camera_parameter_vector[1]; //rows
		target_camera_parameter_vector[2] = (source_camera_parameter_vector[2] + source_camera_parameter_vector[3]) * 0.5f; //平均焦距
		target_camera_parameter_vector[3] = source_camera_parameter_vector[4]; //cx
		target_camera_parameter_vector[4] = source_camera_parameter_vector[5]; //cy
		target_camera_parameter_vector[5] = 0.f;

		float left_min_x = std::numeric_limits<float>::max();
		float left_max_x = std::numeric_limits<float>::lowest();
		float right_min_x = std::numeric_limits<float>::max();
		float right_max_x = std::numeric_limits<float>::lowest();

		for (int r = 0; r < rows; ++r)
		{
			// Left border.
			float2 world_point1;
			ImageToWorld4OpenCVCameraModel(source_camera_parameter_vector.data(), 0.5f, (float)r + 0.5f, world_point1.x, world_point1.y);


			float2 undistorted_point1;
			WorldToImage4SimpleRadialCameraModel(target_camera_parameter_vector.data(), world_point1.x, world_point1.y, undistorted_point1.x, undistorted_point1.y);
			//printf("undistorted_point1 %f %f\n", undistorted_point1.x, undistorted_point1.y);


			left_min_x = left_min_x < undistorted_point1.x ? left_min_x : undistorted_point1.x;
			left_max_x = left_max_x > undistorted_point1.x ? left_max_x : undistorted_point1.x;

			// Right border.
			float2 world_point2;
			ImageToWorld4OpenCVCameraModel(source_camera_parameter_vector.data(), (float)cols - 0.5f, (float)r + 0.5f, world_point2.x, world_point2.y);
			float2 undistorted_point2;
			WorldToImage4SimpleRadialCameraModel(target_camera_parameter_vector.data(), world_point2.x, world_point2.y, undistorted_point2.x, undistorted_point2.y);
			//printf("undistorted_point2 %f %f\n", undistorted_point2.x, undistorted_point2.y);
			right_min_x = right_min_x < undistorted_point2.x ? right_min_x : undistorted_point2.x;
			right_max_x = right_max_x > undistorted_point2.x ? right_max_x : undistorted_point2.x;
		}

		// Determine min, max coordinates along left / right image border.
		float top_min_y = std::numeric_limits<float>::max();
		float top_max_y = std::numeric_limits<float>::lowest();
		float bottom_min_y = std::numeric_limits<float>::max();
		float bottom_max_y = std::numeric_limits<float>::lowest();

		for (int c = 0; c < cols; ++c)
		{
			// Top border.
			float2 world_point1;
			ImageToWorld4OpenCVCameraModel(source_camera_parameter_vector.data(), float(c) + 0.5f, 0.5f, world_point1.x, world_point1.y);
			float2 undistorted_point1;
			WorldToImage4SimpleRadialCameraModel(target_camera_parameter_vector.data(), world_point1.x, world_point1.y, undistorted_point1.x, undistorted_point1.y);
			//printf("undistorted_point1 %f %f\n", undistorted_point1.x, undistorted_point1.y);

			top_min_y = top_min_y < undistorted_point1.y ? top_min_y : undistorted_point1.y;
			top_max_y = top_max_y > undistorted_point1.y ? top_max_y : undistorted_point1.y;

			// Bottom border.
			float2 world_point2;
			ImageToWorld4OpenCVCameraModel(source_camera_parameter_vector.data(), (float)c + 0.5f, (float)rows - 0.5f, world_point2.x, world_point2.y);
			float2 undistorted_point2;
			WorldToImage4SimpleRadialCameraModel(target_camera_parameter_vector.data(), world_point2.x, world_point2.y, undistorted_point2.x, undistorted_point2.y);
			//printf("undistorted_point2 %f %f\n", undistorted_point2.x, undistorted_point2.y);
			bottom_min_y = bottom_min_y < undistorted_point2.y ? bottom_min_y : undistorted_point2.y;
			bottom_max_y = bottom_max_y > undistorted_point2.y ? bottom_max_y : undistorted_point2.y;
		}
		const float cx = target_camera_parameter_vector[3]; //cx
		const float cy = target_camera_parameter_vector[4]; //cy

		// Scale such that undistorted image contains all pixels of distorted image
		float min_scale_x1 = cx / (cx - left_min_x);
		float min_scale_x2 = (cols - 0.5f - cx) / (right_max_x - cx);
		const float min_scale_x = min_scale_x1 < min_scale_x2 ? min_scale_x1 : min_scale_x2;

		float min_scale_y1 = cy / (cy - top_min_y);
		float min_scale_y2 = (rows - 0.5f - cy) / (bottom_max_y - cy);
		const float min_scale_y = min_scale_y1 < min_scale_y2 ? min_scale_y1 : min_scale_y2;

		// Scale such that there are no blank pixels in undistorted image
		float max_scale_x1 = cx / (cx - left_max_x);
		float max_scale_x2 = ((float)cols - 0.5f - cx) / (right_min_x - cx);
		const float max_scale_x = max_scale_x1 > max_scale_x2 ? max_scale_x1 : max_scale_x2;

		float max_scale_y1 = cy / (cy - top_max_y);
		float max_scale_y2 = ((float)rows - 0.5f - cy) / (bottom_min_y - cy);
		const float max_scale_y = max_scale_y1 > max_scale_y2 ? max_scale_y1 : max_scale_y2;
		//printf(" max_scale_y %f %f \n", max_scale_y1, max_scale_y2);

		// Interpolate scale according to blank_pixels.
		float blank_pixels = 0.f;
		float scale_x = 1.f / (min_scale_x * blank_pixels + max_scale_x * (1.f - blank_pixels));
		float scale_y = 1.f / (min_scale_y * blank_pixels + max_scale_y * (1.f - blank_pixels));
		//printf("scale_x0 %f %f \n", scale_x, scale_y);

		// Clip the scaling factors.
		float min_scale = 0.2f;
		float max_scale = 2.f;
		scale_x = Clip(scale_x, min_scale, max_scale);
		scale_y = Clip(scale_y, min_scale, max_scale);
		//printf("scale_x %f %f \n", scale_x, scale_y);

		target_camera_parameter_vector[0] = static_cast<int>(1.f > scale_x * cols ? 1.f : scale_x * cols); //cols
		target_camera_parameter_vector[1] = static_cast<int>(1.f > scale_y * rows ? 1.f : scale_y * rows); //rows

		//printf("target_camera_parameter_vector %f %f \n", target_camera_parameter_vector[0], target_camera_parameter_vector[1]);

		target_camera_parameter_vector[3] = (target_camera_parameter_vector[0] - 1.f) / 2.f; //cx
		target_camera_parameter_vector[4] = (target_camera_parameter_vector[1] - 1.f) / 2.f; //cy
	}

	void UndistortImage(std::vector<float>& distored_camera_parameter_vector, std::vector<float>& undistored_camera_parameter_vector, unsigned char*& image_data)
	{
		Undistort4OpenCVCameraModel(distored_camera_parameter_vector, undistored_camera_parameter_vector, image_data);
		/*distored_camera_parameter_vector[0] = undistored_camera_parameter_vector[0];
		distored_camera_parameter_vector[1] = undistored_camera_parameter_vector[1];*/
	}
}