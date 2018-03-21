#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "customBlender.h"

#include "opencv2\highgui.hpp"

int divUp(int a, int b){
	return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void feedCUDA_kernel(uchar* img, uchar* mask, uchar* dst, uchar* dst_mask, int dx, int dy, int width, int height, int img_step, int dst_step, int mask_step, int mask_dst_step)
{
	int x, y, pixel, pixelOut, pixel_mask, pixelOut_mask;
	y = blockIdx.y * blockDim.y + threadIdx.y;
	x = blockIdx.x * blockDim.x + threadIdx.x;
	if (y >= height){
		//printf("Out of image. y = %d  height = %d \n", y, height);
		return;
	}
	if (x >= width){
		//printf("Out of image. x = %d  width = %d \n", x, width);
		return;
	}

	// MASK
	pixel_mask = (y * (mask_step)) + x;
	pixelOut_mask = ((y + dy) * (mask_dst_step)) + (x + dx);

	// DST 8 BIT
	// Get pixel index. 3 is the num of channel, dx and dy are the deltas
	//pixel = (y * (img_step)) +  3 * x;
	//pixelOut = ((y + dy) * (dst_step)) + (3*(x + dx));
	//// ------ uchar
	//const uchar img_px = img[pixel];
	//dst[pixelOut] = img[pixel];
	//dst[pixelOut + 1] = img[pixel + 1];
	//dst[pixelOut + 2] = img[pixel + 2];
	//// ------ uchar3
	////const uchar3 img_px = img[pixel];
	////dst[pixelOut] = make_uchar3(img_px.x, img_px.y, img_px.z);

	// DST 16 BIT
	// Get pixel index. 3 is the num of channel, dx and dy are the deltas
	pixel = (y * (img_step)) + 6 * x;
	pixelOut = ((y + dy) * (dst_step)) + (6 * (x + dx));
	// ------ uchar
	const uchar img_px = img[pixel];
	if (mask[pixel_mask]) {
		dst[pixelOut] = img[pixel];
		dst[pixelOut + 1] = img[pixel + 1];
		dst[pixelOut + 2] = img[pixel + 2];
		dst[pixelOut + 3] = img[pixel + 3];
		dst[pixelOut + 4] = img[pixel + 4];
		dst[pixelOut + 5] = img[pixel + 5];
	}

	// MASK
	dst_mask[pixelOut_mask] |= mask[pixel_mask] ;

	return;
}

extern "C" void feedCUDA(cv::cuda::GpuMat img, cv::cuda::GpuMat mask, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& dst_mask, int dx, int dy)
{
	dim3 blockDim(32, 32, 1);
	dim3 gridDim(divUp(img.cols, blockDim.x), divUp(img.rows, blockDim.y), 1);

	feedCUDA_kernel << <gridDim, blockDim >> >((uchar*)img.data, mask.data, (uchar*)dst.data, dst_mask.data, dx, dy, img.cols, img.rows, img.step, dst.step, mask.step, dst_mask.step);

//	printf("width = %d, height = %d, step = %d, widthdst = %d, heightdst = %d, stepdst = %d \n ", img.cols, img.rows, img.step, dst.cols, dst.rows, dst.step);
//	printf("width = %d, height = %d, step = %d, widthdst = %d, heightdst = %d, stepdst = %d \n ", mask.cols, mask.rows, mask.step, dst_mask.cols, dst_mask.rows, dst_mask.step);

	//cv::imshow("img in feedCUDA", cv::Mat(img));
	//cv::imwrite("img.png", cv::Mat(img));
	//cv::waitKey(0);

	//cv::Mat temp1, temp2;
	//mask.download(temp1);
	//dst_mask.download(temp2);

	//cv::imshow("mask in feedCUDA", cv::Mat(mask));
	//cv::imwrite("mask.png", cv::Mat(mask));
	//cv::waitKey(0);

	//cv::imshow("dst_mask in feedCUDA", cv::Mat(dst_mask));
	//cv::imwrite("dst_mask.png", cv::Mat(dst_mask));
	//cv::waitKey(0);

	cudaDeviceSynchronize();
}
