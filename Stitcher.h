#pragma once

#include <iostream>
#include "opencv2/opencv.hpp"
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

#ifdef WITH_CUDA
	#include "customBlender.h"
#endif

#include <time.h>
#ifdef WITH_CUDA
	#include <opencv2/cudawarping.hpp>
	#include <opencv2/cudaarithm.hpp>
#endif


class Stitcher
{
public:
	enum FeatureTypes {
		SURF,
		ORB,
		SIFT
	};

	enum BaCostFunctions {
		REPROJ,
		RAY,
		AFFINE,
		NONE
	};

	enum WarpingTypes {
		PLANE,
		CYLINDRICAL,
		WARP_AFFINE,
		SPHERICAL,
		FISHEYE,
		STEREOGRAPHIC,
		COMPRESSEDPLANEA2B1,
		COMPRESSEDPLANEA1_5B1,
		COMPRESSEDPLANEPORTRAITA2B1,
		COMPRESSEDPLANEPORTRAITA1_5B1,
		PANINIA2B1,
		PANINIA1_5B1,
		PANINIPORTRAITA2B1,
		PANINIPORTRAITA1_5B1,
		MERCATOR,
		TRANSVERSEMERCATOR
	};

	enum SeamFindTypes {
		NO,
		VORONOI,
		GC_COLOR,
		GC_COLORGRAD,
		DP_COLOR,
		DP_COLORGRAD
	};

	struct StitchingParams {
		StitchingParams() {
			work_megapix = 0.6f;
			seam_megapix = 0.1f;
			compose_megapix = -1.f;
			conf_thresh = 1.f;
			features_type = SURF;
			ba_cost_func = RAY;
			ba_refine_mask = "xxxxx";
			do_wave_correct = false;
			wave_correct = cv::detail::WAVE_CORRECT_HORIZ;
			warp_type = SPHERICAL;
			expos_comp_type = cv::detail::ExposureCompensator::NO;
			//expos_comp_type = cv::detail::ExposureCompensator::GAIN_BLOCKS;
			match_conf = 0.7f;
			seam_find_type = NO;
			//seam_find_type = DP_COLORGRAD;
			blend_type = cv::detail::Blender::NO;
			//blend_type = cv::detail::Blender::MULTI_BAND;
			//blend_strength = 5.f;
			blend_strength = 0.f;

		}
		double work_megapix;
		double seam_megapix;
		double compose_megapix;
		float conf_thresh;
		Stitcher::FeatureTypes features_type;
		Stitcher::BaCostFunctions ba_cost_func;
		std::string ba_refine_mask;
		bool do_wave_correct;
		cv::detail::WaveCorrectKind wave_correct;
		Stitcher::WarpingTypes warp_type;
		int expos_comp_type;
		float match_conf;
		Stitcher::SeamFindTypes seam_find_type;
		int blend_type;
		float blend_strength;
		bool show_overview_in_panorama;
	};


	Stitcher(std::vector<cv::Mat> images);
	bool ComputeRegistration(StitchingParams params);
	bool ComputeRegistration(bool showOverviewInPanorama, FeatureTypes featureTypes = SURF, double workMegapix = 0.6,
		float matchConf = 0.3f, float confThresh = 1.f,
		BaCostFunctions baCostFunc = RAY, std::string baRefineMask = "xxxxx",
		bool doWaveCorrect = false, cv::detail::WaveCorrectKind waveCorrection = cv::detail::WAVE_CORRECT_HORIZ,
		WarpingTypes warpType = SPHERICAL, double seamMegapix = 0.1,
		SeamFindTypes seamFindType = GC_COLOR,
		int exposCompType = cv::detail::ExposureCompensator::GAIN_BLOCKS, double composeMegapix = -1, int blendType = cv::detail::Blender::MULTI_BAND, float blendStrength = 5);
	bool ComputeCompositing(std::vector<cv::Mat> images, cv::Mat& result, cv::Mat& resultMask, bool showOverviewInPanorama, bool lowQualityStitching);
#ifdef WITH_CUDA
	//version GPU
	bool ComputeCompositing(std::vector<cv::cuda::GpuMat> images, cv::Mat& result, cv::Mat& resultMask, bool showOverviewInPanorama, bool lowQualityStitching);
#endif

	bool FindFeatures(FeatureTypes featureTypes = SURF, double workMegapix = 0.6);
	bool MatchFeatures(float matchConf = 0.3f, float confThresh = 1.f);
	bool EstimateCameraParameters();
	bool BundleAdjustment(BaCostFunctions baCostFunc = RAY, std::string baRefineMask = "xxxxx");
	bool WaveCorrection(cv::detail::WaveCorrectKind waveCorrection = cv::detail::WAVE_CORRECT_HORIZ);
	bool WarpAux(WarpingTypes warpType = SPHERICAL, double seamMegapix = 0.1);
	bool SeamFinding(SeamFindTypes seamFindType = GC_COLOR);
	//bool Prepare(int exposCompType = cv::detail::ExposureCompensator::GAIN_BLOCKS, double composeMegapix = -1, int blendType = cv::detail::Blender::MULTI_BAND, float blendStrength = 5);
	bool Prepare(int exposCompType = 0, double composeMegapix = -1, int blendType = 0, float blendStrength = 0);
	bool Compose(std::vector<cv::Mat> images, cv::Mat& result, cv::Mat& resultMask, bool showOverviewInPanorama, bool lowQualityStitching);
	//version GPU
	bool Compose(std::vector<cv::cuda::GpuMat> images, cv::Mat& result, cv::Mat& resultMask, bool showOverviewInPanorama, bool lowQualityStitching);

	~Stitcher(void);

	void Serialize(char* buffer, int& size);
	bool Deserialize(const char* buffer, int size);
	void DeleteBlender();

	int x_left;
	int x_right;
	int y_top;
	int y_bottom;


//#if OUTPUT_ON_FILE
//	static std::ofstream logFile;
//#endif

private:
	bool PrepareExposureCompensator(int exposCompType = cv::detail::ExposureCompensator::GAIN_BLOCKS);
	bool PrepareWarp();
	bool PrepareBlend(int blendType = cv::detail::Blender::MULTI_BAND, float blendStrength = 5);

	int _numImagesTot, _numImages;
	std::vector<cv::Mat> _images;
	//version GPU
	std::vector<cv::cuda::GpuMat> _images_gpu;

	std::vector<cv::detail::ImageFeatures> _features;
	std::vector<cv::detail::MatchesInfo> _pairwiseMatches;
	std::vector<int> _validIndices;
	std::vector<cv::detail::CameraParams> _cameras;
	std::vector<cv::Point> _corners;
	//std::vector<cv::Mat> _masksWarpedSeam;
	std::vector<cv::UMat> _masksWarpedSeam;
	//std::vector<cv::Mat> _imagesWarped;
	std::vector<cv::UMat> _imagesWarped;
	std::vector<cv::Size> _sizes;
	cv::Ptr<cv::WarperCreator> _warperCreator;
	cv::Ptr<cv::detail::ExposureCompensator> _compensator;


#ifdef WITH_CUDA
	CustomBlender* _customBlender;
#else
	cv::Ptr<cv::detail::Blender> _blender;
#endif


	std::vector<cv::Mat> _xmap, _ymap;

	//_xmap, _ymap per gpu
#ifdef WITH_CUDA
	std::vector<cv::cuda::GpuMat> _xmap_gpu, _ymap_gpu;
#endif

	std::vector<cv::Mat> _masksWarpedCompose;

	bool _isWorkScaleSet;
	bool _isSeamScaleSet;
	bool _isComposeScaleSet;

	double _workScale;
	double _seamScale;
	double _composeScale;
	float _warpedImageScale;
	double _seamWorkAspect;

	int _blendType;
	float _blendStrength;
	int _exposCompType;

	float _confThresh;

};
