#include "customBlender.h"
#include "opencv2\highgui.hpp"

extern "C" {
	void feedCUDA(cv::cuda::GpuMat img, cv::cuda::GpuMat mask, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& dst_mask, int dx, int dy);
}

CustomBlender::CustomBlender(void)
{
}


CustomBlender::~CustomBlender(void)
{
}


void CustomBlender::prepare(const std::vector<cv::Point> &corners, const std::vector<cv::Size> &sizes)
{
	prepare(cv::detail::resultRoi(corners, sizes));
}


void CustomBlender::prepare(cv::Rect dst_roi)
{
	
	//dst_ = cv::cuda::GpuMat(dst_roi.size(), CV_8UC3);
	dst_ = cv::cuda::GpuMat(dst_roi.size(), CV_16SC3);
	dst_.setTo(cv::Scalar::all(0));
	dst_mask_ = cv::cuda::GpuMat(dst_roi.size(), CV_8U);
	dst_mask_.setTo(cv::Scalar::all(0));
	dst_roi_ = dst_roi;
}


void CustomBlender::feed(cv::cuda::GpuMat _img, cv::cuda::GpuMat _mask, cv::Point tl)
{

	// No CUDA Kernel
	//Mat img, mask, dst, dst_mask;
	//_img.download(img);
	//_mask.download(mask);
	//dst_.download(dst);
	//dst_mask_.download(dst_mask);
	//int dx = tl.x - dst_roi_.x;
	//int dy = tl.y - dst_roi_.y;
	//for (int y = 0; y < img.rows; ++y)
	//{
	//	const Point3_<short> *src_row = img.ptr<Point3_<short> >(y);
	//	Point3_<short> *dst_row = dst.ptr<Point3_<short> >(dy + y);
	//	const uchar *mask_row = mask.ptr<uchar>(y);
	//	uchar *dst_mask_row = dst_mask.ptr<uchar>(dy + y);

	//	for (int x = 0; x < img.cols; ++x)
	//	{
	//		if (mask_row[x])
	//			dst_row[dx + x] = src_row[x];
	//		dst_mask_row[dx + x] |= mask_row[x];
	//	}
	//}
	//dst_.upload(dst);
	//dst_mask_.upload(dst_mask);

	//cv::Mat temp1;
	//dst.convertTo(temp1, CV_8UC3);
	//cv::imshow("dst in feedCUDA", cv::Mat(temp1));
	//cv::imwrite("dst.png", cv::Mat(temp1));
	//cv::waitKey(0);


	// with CUDA KERNEL
	//CV_Assert(_img.type() == CV_8UC3);
	CV_Assert(_img.type() == CV_16SC3);
	CV_Assert(_mask.type() == CV_8U);
	int dx = tl.x - dst_roi_.x;
	int dy = tl.y - dst_roi_.y;
	feedCUDA(_img, _mask, dst_, dst_mask_, dx, dy);
}


void CustomBlender::blend(cv::cuda::GpuMat &dst, cv::cuda::GpuMat &dst_mask)
{


	cv::cuda::GpuMat mask;
	cv::cuda::compare(dst_mask_, 0, mask, cv::CMP_EQ);
	dst_.setTo(cv::Scalar::all(0), mask);
	dst_.copyTo(dst);
	dst_mask_.copyTo(dst_mask);


	//cv::Mat tempdst, tempdstmask, tempmask;
	//dst_.download(tempdst);
	//dst_mask_.download(tempdstmask);
	//mask.download(tempmask);

	dst_.release();
	dst_mask_.release();
}
