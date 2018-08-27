#include "Stitcher.h"

using namespace std;
using namespace cv;
using namespace cv::detail;

#define LOGLN(msg) std::cout << msg << std::endl

#define ENABLE_LOG 0


//#if OUTPUT_ON_FILE
//std::ofstream Stitcher::logFile;
//#endif

Stitcher::Stitcher(std::vector<Mat> images)
{
	_images = images;
	_numImagesTot = (int)images.size();
	_numImages = _numImagesTot;

	_isWorkScaleSet = false;
	_isSeamScaleSet = false;
	_isComposeScaleSet = false;
	_workScale = 1;
	_seamScale = 1;
	_composeScale = 1;
	_warpedImageScale = -1;
	_seamWorkAspect = 1;

	_blendType = -1;
	_blendStrength = -1;
	_exposCompType = -1;
}


Stitcher::~Stitcher(void)
{
	for (int img_idx = 0; img_idx < _masksWarpedSeam.size(); ++img_idx)
	{
		_masksWarpedSeam[img_idx].release();
	}
	_masksWarpedSeam.clear();
	for (int img_idx = 0; img_idx < _masksWarpedSeam.size(); ++img_idx)
	{
		_images[img_idx].release();
	}
	_images.clear();
	for (int img_idx = 0; img_idx < _masksWarpedSeam.size(); ++img_idx)
	{
		_masksWarpedCompose[img_idx].release();
	}
	_masksWarpedCompose.clear();
	_numImagesTot = 0;
	_numImages = 0;

	_corners.clear();
	for (int img_idx = 0; img_idx < _imagesWarped.size(); ++img_idx)
	{
		_imagesWarped[img_idx].release();
	}
	_imagesWarped.clear();
}

bool Stitcher::ComputeRegistration(StitchingParams params)
{
	return ComputeRegistration(params.show_overview_in_panorama, params.features_type, params.work_megapix,
		params.match_conf, params.conf_thresh,
		params.ba_cost_func, params.ba_refine_mask,
		params.do_wave_correct, params.wave_correct,
		params.warp_type, params.seam_megapix,
		params.seam_find_type,
		params.expos_comp_type, params.compose_megapix, params.blend_type, params.blend_strength);
}
bool Stitcher::ComputeRegistration(bool showOverviewInPanorama, FeatureTypes featureTypes, double workMegapix,
	float matchConf, float confThresh,
	BaCostFunctions baCostFunc, std::string baRefineMask,
	bool doWaveCorrect, cv::detail::WaveCorrectKind waveCorrection,
	WarpingTypes warpType, double seamMegapix,
	SeamFindTypes seamFindType,
	int exposCompType, double composeMegapix, int blendType, float blendStrength)
{
#if ENABLE_LOG
	int64 t = getTickCount();
#endif

	if (!FindFeatures(featureTypes, workMegapix))
	{
		return false;
	}

	if (!MatchFeatures(matchConf, confThresh))
	{
		return false;
	}

	if (!EstimateCameraParameters())
	{
		return false;
	}

	if (!BundleAdjustment(baCostFunc, baRefineMask))
	{
		return false;
	}

	if (doWaveCorrect)
		if (!WaveCorrection(waveCorrection))
		{
			return false;
		}

	if (!WarpAux(warpType, seamMegapix))
	{
		return false;
	}

	if (!SeamFinding(seamFindType))
	{
		return false;
	}

	if (!Prepare(exposCompType, composeMegapix, blendType, blendStrength))
	{
		return false;
	}
#if ENABLE_LOG
	LOGLN("Stitching registration, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
#endif
	return true;
}


#ifdef WITH_CUDA
bool Stitcher::ComputeCompositing(std::vector<cv::cuda::GpuMat> images, cv::Mat& result, cv::Mat& resultMask, bool showOverviewInPanorama, bool lowQualityStitching)
{
#if ENABLE_LOG
	int64 t = getTickCount();
#endif

	bool res = Compose(images, result, resultMask, showOverviewInPanorama, lowQualityStitching);

#if ENABLE_LOG
	if (res)
	{
		LOGLN("Stitching compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	}
#endif

	return res;
}
#else
	bool Stitcher::ComputeCompositing(std::vector<cv::Mat> images, cv::Mat& result, cv::Mat& resultMask, bool showOverviewInPanorama, bool lowQualityStitching)
	{
	#if ENABLE_LOG
		int64 t = getTickCount();
	#endif

		bool res = Compose(images, result, resultMask, showOverviewInPanorama, lowQualityStitching);

	#if ENABLE_LOG
		if (res)
		{
			LOGLN("Stitching compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		}
	#endif

		return res;
	}
#endif

bool Stitcher::FindFeatures(FeatureTypes featureTypes, double workMegapix)
{
	if (_numImagesTot < 2)
	{
		LOGLN("[Stitcher] ERROR: Not enough images");
		return false;
	}

	Ptr<FeaturesFinder> finder;
	switch (featureTypes)
	{
	case SURF:
	{
#ifdef WITH_CUDA
	#ifdef HAVE_OPENCV_XFEATURES2D
		if (cuda::getCudaEnabledDeviceCount() > 0)
			finder = makePtr<SurfFeaturesFinderGpu>();
		else
	#endif
#endif
			finder = makePtr<SurfFeaturesFinder>();
		break;
	}
	case ORB:
	{
				finder = makePtr<OrbFeaturesFinder>();
				break;
	}
	default:
		LOGLN("[Stitcher] ERROR: unknown feature type");
		return false;
	}

	_features.resize(_numImagesTot);

	Mat full_img, img;

	for (int i = 0; i < _numImagesTot; ++i)
	{
		full_img = _images[i];

		if (full_img.empty())
		{
			LOGLN("[Stitcher] ERROR: Empty image #" << i);
			return false;
		}
		if (workMegapix < 0)
		{
			img = full_img;
			_workScale = 1;
			_isWorkScaleSet = true;
		}
		else
		{
			if (!_isWorkScaleSet)
			{
				_workScale = min(1.0, sqrt(workMegapix * 1e6 / full_img.size().area()));
				_isWorkScaleSet = true;
			}
			cv::resize(full_img, img, Size(), _workScale, _workScale);
		}

		(*finder)(img, _features[i]);
		_features[i].img_idx = i;
		LOGLN("Features in image #" << i+1 << ": " << _features[i].keypoints.size());
		
		// Debug features results
		//Mat out;
		//drawKeypoints(img, _features[i].keypoints, out);
		//imshow("output keypoints", out);
		//waitKey(0);
		//out.release();
		//destroyWindow("output keypoints");
		// ----------------
	}

	finder->collectGarbage();
	full_img.release();
	img.release();
	return true;
}


bool Stitcher::MatchFeatures(float matchConf, float confThresh)
{
	if (_features.size() == 0)
	{
		LOGLN("[Stitcher] ERROR: No features. Run FindFeatures before");
		return false;
	}

	_confThresh = confThresh;

	// Find the maximum matchConf
	std::vector<cv::detail::ImageFeatures> temp_features = _features;
	float temp_matchConf = matchConf;
	_numImages = 0;
	while (_numImages < 2)
	{
		_features = temp_features;

		//LOGLN("[Stitcher]  Total matches found: " << _pairwiseMatches.size());
		Ptr<FeaturesMatcher> matcher;
		//TODO: parametrizzare rangewidth e matcherType
		int range_width = -1;
		string matcher_type="";
		if (matcher_type == "affine")
			matcher = makePtr<AffineBestOf2NearestMatcher>(false, true, matchConf);
		else if (range_width==-1)
			matcher = makePtr<BestOf2NearestMatcher>(true, matchConf);
		else
			matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, true, matchConf);
		(*matcher)(_features, _pairwiseMatches);
		LOGLN("[Stitcher]  Total matches found: " << _pairwiseMatches.size());
		matcher->collectGarbage();

#ifdef WRITE_IMAGES
		// Debug features results
		Mat out1, imgresize0, imgresize1, imgresize2, imgresize3, imgresize4, out2, out3, out4;
		imwrite("img0.png", _images[0]);
		imwrite("img1.png", _images[1]);
		imwrite("img2.png", _images[2]);
		imwrite("img3.png", _images[3]);
		imwrite("img4.png", _images[4]);
		resize(_images[0], imgresize0, Size(), _workScale, _workScale);
		resize(_images[1], imgresize1, Size(), _workScale, _workScale);
		resize(_images[2], imgresize2, Size(), _workScale, _workScale);
		resize(_images[3], imgresize3, Size(), _workScale, _workScale);
		resize(_images[4], imgresize4, Size(), _workScale, _workScale);
		drawMatches(imgresize0, _features[0].keypoints, imgresize1, _features[1].keypoints, _pairwiseMatches[1].matches, out1, Scalar(0, 255, 0));
		drawMatches(imgresize0, _features[0].keypoints, imgresize2, _features[2].keypoints, _pairwiseMatches[2].matches, out2, Scalar(0, 255, 0));
		drawMatches(imgresize0, _features[0].keypoints, imgresize3, _features[3].keypoints, _pairwiseMatches[3].matches, out3, Scalar(0, 255, 0));
		drawMatches(imgresize0, _features[0].keypoints, imgresize4, _features[4].keypoints, _pairwiseMatches[4].matches, out4, Scalar(0, 255, 0));
		

		resize(out1, out1, Size(600, 400));
		resize(out2, out2, Size(600, 400));
		resize(out3, out3, Size(600, 400));
		resize(out4, out4, Size(600, 400));
		imshow("matches1", out1);
		imshow("matches2", out2);
		imshow("matches3", out3);
		imshow("matches4", out4);
		waitKey(0);
		imwrite("matches1.png", out1);
		imwrite("matches2.png", out2);
		imwrite("matches3.png", out3);
		imwrite("matches4.png", out4);
		imgresize0.release();
		imgresize1.release();
		imgresize2.release();
		imgresize3.release();
		imgresize4.release();
		out1.release();
		out2.release();
		out3.release();
		out4.release();
		destroyWindow("matches1");
		destroyWindow("matches2");
		destroyWindow("matches3");
		destroyWindow("matches4");
#endif // WRITE_IMAGES

		// Leave only images we are sure are from the same panorama
		_validIndices = leaveBiggestComponent(_features, _pairwiseMatches, confThresh);
		LOGLN("[Stitcher]  Total matches found after leaveBiggestComponent: " << _pairwiseMatches.size());

		//LOG("[Stitcher]  Inlier images: ");
		//for (int i = 0; i < _validIndices.size(); i++)
		//	LOG("" << _validIndices[i] << "  ");
		//LOGLN("");

		vector<Mat> img_subset;
		vector<string> img_names_subset;
		vector<Size> full_img_sizes_subset;

		for (size_t i = 0; i < _validIndices.size(); ++i)
		{
			img_subset.push_back(_images[_validIndices[i]]);
		}
		//_images = img_subset;
		//_numImages = static_cast<int>(_images.size());
		_numImages = static_cast<int>(img_subset.size());

		if (temp_matchConf < 0){
			return false;
		}


		// Check if we still have enough images
		if (_numImages < 2)
		{
			LOGLN("[Stitcher] ERROR: Not enough valid images after match. Total matches found: " << _pairwiseMatches.size());
			LOGLN("[Stitcher] temp_matchConf is: " << temp_matchConf);
			temp_matchConf -= 0.1;
			continue;
			//return false;
		}
		else{
			_images = img_subset;
			LOGLN("[Stitcher] OK, temp_matchConf is: " << temp_matchConf);
			break;
		}
	}
	return true;
}

bool Stitcher::EstimateCameraParameters()
{
	if (_features.size() == 0)
	{
		LOGLN("[Stitcher] ERROR: No features. Run FindFeatures before");
		return false;
	}
	if (_pairwiseMatches.size() == 0)
	{
		LOGLN("[Stitcher] ERROR: No matches. Run MatchFeatures before");
		return false;
	}

	Ptr<Estimator> estimator;
	String estimator_type = "";
	if (estimator_type == "affine")
		estimator = makePtr<AffineBasedEstimator>();
	else
		estimator = makePtr<HomographyBasedEstimator>();

    if (!(*estimator)(_features, _pairwiseMatches, _cameras))
    {
        cout << "[Stitcher] Homography estimation failed.\n";
        return false;
    }

	for (size_t i = 0; i < _cameras.size(); ++i)
	{
		Mat R;
		_cameras[i].R.convertTo(R, CV_32F);
		_cameras[i].R = R;
		//LOGLN("Initial intrinsics #" << _validIndices[i]+1 << ":\n" << _cameras[i].K());
	}
	return true;
}

bool Stitcher::BundleAdjustment(BaCostFunctions baCostFunc, std::string baRefineMask)
{
	if (_cameras.size() == 0)
	{
		LOGLN("[Stitcher] ERROR: No camera parameters. Run EstimateCameraParameters before");
		return false;
	}
	if (_features.size() == 0)
	{
		LOGLN("[Stitcher] ERROR: No fetures. Run FindFeatures before");
		return false;
	}
	if (_pairwiseMatches.size() == 0)
	{
		LOGLN("[Stitcher] ERROR: No matches. Run MatchFeatures before");
		return false;
	}

	Ptr<detail::BundleAdjusterBase> adjuster;
	switch (baCostFunc)
	{
	case REPROJ:
		adjuster = makePtr<detail::BundleAdjusterReproj>();
		break;
	case RAY:
		adjuster = makePtr<detail::BundleAdjusterRay>();
		break;
	case AFFINE:
		adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
		break;
	case NONE:
		adjuster = makePtr<NoBundleAdjuster>();
		break;
	default:
		LOGLN("[Stitcher] ERROR: Unknown ba cost function");
		return false;
	}

	adjuster->setConfThresh(_confThresh);

	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (baRefineMask[0] == 'x') refine_mask(0, 0) = 1;
	if (baRefineMask[1] == 'x') refine_mask(0, 1) = 1;
	if (baRefineMask[2] == 'x') refine_mask(0, 2) = 1;
	if (baRefineMask[3] == 'x') refine_mask(1, 1) = 1;
	if (baRefineMask[4] == 'x') refine_mask(1, 2) = 1;

	adjuster->setRefinementMask(refine_mask);

    if (!(*adjuster)(_features, _pairwiseMatches, _cameras))
    {
        cout << "Camera parameters adjusting failed.\n";
        return false;
    }

	return true;

}

bool Stitcher::WaveCorrection(WaveCorrectKind waveCorrection)
{
	if (_cameras.size() == 0)
	{
		LOGLN("[Stitcher] ERROR: No camera parameters. Run EstimateCameraParameters before");
		return false;
	}

	vector<Mat> rmats;
	for (size_t i = 0; i < _cameras.size(); ++i)
		rmats.push_back(_cameras[i].R);
	waveCorrect(rmats, waveCorrection);
	for (size_t i = 0; i < _cameras.size(); ++i)
		_cameras[i].R = rmats[i];

	return true;
}


bool Stitcher::WarpAux(WarpingTypes warpType, double seamMegapix)
{
	if (_cameras.size() == 0)
	{
		LOGLN("[Stitcher] ERROR: No camera parameters. Run EstimateCameraParameters before");
		return false;
	}

	// Find median focal length
	vector<double> focals;
	for (size_t i = 0; i < _cameras.size(); ++i)
	{
		focals.push_back(_cameras[i].focal);
	}
	sort(focals.begin(), focals.end());

	if (focals.size() % 2 == 1)
		_warpedImageScale = static_cast<float>(focals[focals.size() / 2]);
	else
		_warpedImageScale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	_corners.resize(_numImages);
	_masksWarpedSeam.resize(_numImages);
	_imagesWarped.resize(_numImages);
	vector<Mat> masks(_numImages);

	// Warp images and their masks
	if (_warperCreator.empty())
	{
#ifdef HAVE_OPENCV_CUDAWARPING
		if (cuda::getCudaEnabledDeviceCount() > 0)
		{
			switch (warpType)
			{
			case PLANE:
				_warperCreator = makePtr<cv::PlaneWarperGpu>();
				break;
			case CYLINDRICAL:
				_warperCreator = makePtr<cv::CylindricalWarperGpu>();
				break;
			case SPHERICAL:
				_warperCreator = makePtr<cv::SphericalWarperGpu>();
				break;
			default:
				LOGLN("[Stitcher] ERROR: Unknown warp type");
				return false;
			}
		}
		else
#endif
		{
			switch (warpType)
			{
			case PLANE:
				_warperCreator = makePtr<cv::PlaneWarper>();
				break;
			case WARP_AFFINE:
				_warperCreator = makePtr<cv::AffineWarper>();
				break;
			case CYLINDRICAL:
				_warperCreator = makePtr<cv::CylindricalWarper>();
				break;
			case SPHERICAL:
				_warperCreator = makePtr<cv::SphericalWarper>();
				break;
			case FISHEYE:
				_warperCreator = makePtr<cv::FisheyeWarper>();
				break;
			case STEREOGRAPHIC:
				_warperCreator = makePtr<cv::StereographicWarper>();
				break;
			case COMPRESSEDPLANEA2B1:
				_warperCreator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
				break;
			case COMPRESSEDPLANEA1_5B1:
				_warperCreator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
				break;
			case COMPRESSEDPLANEPORTRAITA2B1:
				_warperCreator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
				break;
			case COMPRESSEDPLANEPORTRAITA1_5B1:
				_warperCreator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
				break;
			case PANINIA2B1:
				_warperCreator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
				break;
			case PANINIA1_5B1:
				_warperCreator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
				break;
			case PANINIPORTRAITA2B1:
				_warperCreator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
				break;
			case PANINIPORTRAITA1_5B1:
				_warperCreator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
				break;
			case MERCATOR:
				_warperCreator = makePtr<cv::MercatorWarper>();
				break;
			case TRANSVERSEMERCATOR:
				_warperCreator = makePtr<cv::TransverseMercatorWarper>();
				break;
			default:
				LOGLN("[Stitcher] ERROR: Unknown warp type");
				return false;
			}
		}
		if (_warperCreator.empty())
		{
			LOGLN("[Stitcher] ERROR: Can not create warper creator");
			return false;
		}
	}

	if (!_isSeamScaleSet)
	{
		_seamScale = min(1.0, sqrt(seamMegapix * 1e6 / _images[0].size().area()));
		_seamWorkAspect = _seamScale / _workScale;
		_isSeamScaleSet = true;
	}

	Ptr<RotationWarper> warper = _warperCreator->create(static_cast<float>(_warpedImageScale * _seamWorkAspect));
	Mat img;
	for (int i = 0; i < _numImages; ++i)
	{
		cv::resize(_images[i], img, Size(), _seamScale, _seamScale);
		//_images[i] = img.clone();

		// Prepare images masks
		masks[i].create(img.size(), CV_8U);
		masks[i].setTo(Scalar::all(255));

		Mat_<float> K;
		_cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)_seamWorkAspect;
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;

		_corners[i] = warper->warp(img, K, _cameras[i].R, INTER_LINEAR, BORDER_REFLECT, _imagesWarped[i]);

		//LOGLN("_warpedImageScale: " << _warpedImageScale);
		//LOGLN("_seamWorkAspect: " << _seamWorkAspect);

		//LOGLN("warp #" << i << " K:\n" << K);
		//LOGLN("warp #" << i << " R:\n" << _cameras[i].R);

		//std::stringstream ss;
		//ss << "masks_" << i << ".jpg";
		//imwrite(ss.str(), masks[i]);

		warper->warp(masks[i], K, _cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, _masksWarpedSeam[i]);

		//ss.str("");
		//ss << "mask_s_warp" << i << ".jpg";
		//imwrite(ss.str(), _masksWarpedSeam[i]);

	}

	masks.clear();
	img.release();

	return true;
}


bool Stitcher::SeamFinding(SeamFindTypes seamFindType)
{
	if (_imagesWarped.size() <= 0)
	{
		LOGLN("[Stitcher] ERROR: No warped images. Run WarpAux before");
		return false;
	}
	if (_corners.size() <= 0)
	{
		LOGLN("[Stitcher] ERROR: No mask corners. Run WarpAux before");
		return false;
	}
	if (_masksWarpedSeam.size() <= 0)
	{
		LOGLN("[Stitcher] ERROR: No warped masks. Run WarpAux before");
		return false;
	}

	Ptr<SeamFinder> seam_finder;
	switch (seamFindType)
	{
	case NO:
		seam_finder = makePtr<detail::NoSeamFinder>();
		break;
	case VORONOI:
		seam_finder = makePtr<detail::VoronoiSeamFinder>();
		break;
	case GC_COLOR:
#ifdef HAVE_OPENCV_CUDALEGACY
		if (cuda::getCudaEnabledDeviceCount() > 0)
			seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
		else
#endif
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
		break;
	case GC_COLORGRAD:
#ifdef HAVE_OPENCV_CUDALEGACY
		if (cuda::getCudaEnabledDeviceCount() > 0)
			seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		else
#endif
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		break;
	case DP_COLOR:
		seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
		break;
	case DP_COLORGRAD:
		seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
		break;
	default:
		LOGLN("[Stitcher] ERROR: Unknown seam find type");
		return false;
	}
	if (seam_finder.empty())
	{
		LOGLN("[Stitcher] ERROR: Can not create seam finder");
		return false;
	}

	vector<UMat> images_warped_f(_numImages);


	for (int i = 0; i < _numImages; ++i)
		_imagesWarped[i].convertTo(images_warped_f[i], CV_32F);

	//for(int i = 0; i < _masksWarpedSeam.size(); i++)
	//{
	//	std::stringstream ss;
	//	ss << "mask_s_preSeam" << i << ".jpg";
	//	imwrite(ss.str(), _masksWarpedSeam[i]);
	//}

	seam_finder->find(images_warped_f, _corners, _masksWarpedSeam);


	// Release unused memory
	images_warped_f.clear();

	return true;
}

bool Stitcher::Prepare(int exposCompType, double composeMegapix, int blendType, float blendStrength)
{
	if (!_isComposeScaleSet)
	{
		if (composeMegapix > 0)
			_composeScale = min(1.0, sqrt(composeMegapix * 1e6 / _images[0].size().area()));
		_isComposeScaleSet = true;
	}

#if ENABLE_LOG
	LOGLN("Prepare exposure compensator...");
	int64 t = getTickCount();
#endif
	bool result = PrepareExposureCompensator(exposCompType);

#if ENABLE_LOG
	LOGLN("Prepare exposure compensator, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	LOGLN("Prepare warp...");
	t = getTickCount();
#endif
	result &= PrepareWarp();

#if ENABLE_LOG
	LOGLN("Prepare warp, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
	LOGLN("Prepare blend...");
	t = getTickCount();
#endif
	result &= PrepareBlend(blendType, blendStrength);

#if ENABLE_LOG
	LOGLN("Prepare blend, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
#endif
	return result;
}


bool Stitcher::PrepareExposureCompensator(int exposCompType)
{
	if (_corners.size() == 0)
	{
		LOGLN("[Stitcher] ERROR: No mask corners. Run WarpAux before");
		return false;
	}
	if (_imagesWarped.size() == 0)
	{
		LOGLN("[Stitcher] ERROR: No warped images. Run WarpAux before");
		return false;
	}
	if (_masksWarpedSeam.size() == 0)
	{
		LOGLN("[Stitcher] ERROR: No warped masks. Run WarpAux before");
		return false;
	}

	_exposCompType = exposCompType;

	_compensator = ExposureCompensator::createDefault(exposCompType);
	_compensator->feed(_corners, _imagesWarped, _masksWarpedSeam);

	return true;
}

bool Stitcher::PrepareWarp()
{
	if (_warperCreator.empty())
	{
		LOGLN("[Stitcher] ERROR: No instantiated warper creator. Run WarpAux before");
		return false;
	}
	if (_cameras.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No camera parameters. Run EstimateCameraParameters before");
		return false;
	}

	double compose_work_aspect = 1;

	// Compute relative scales
	//compose_seam_aspect = compose_scale / seam_scale;
	compose_work_aspect = _composeScale / _workScale;

	// Update warped image scale
	_warpedImageScale *= static_cast<float>(compose_work_aspect);
	Ptr<RotationWarper> warper = _warperCreator->create(_warpedImageScale);

	_sizes.resize(_numImages);
	_corners.resize(_numImages);

	// Update corners and sizes
	for (int i = 0; i < _numImages; ++i)
	{
		// Update intrinsics
		_cameras[i].focal *= compose_work_aspect;
		_cameras[i].ppx *= compose_work_aspect;
		_cameras[i].ppy *= compose_work_aspect;

		// Update corner and size
		Size sz = _images[i].size();
		if (std::abs(_composeScale - 1) > 1e-1)
		{
			sz.width = cvRound(sz.width * _composeScale);
			sz.height = cvRound(sz.height * _composeScale);
		}

		Mat K;
		_cameras[i].K().convertTo(K, CV_32F);
		Rect roi = warper->warpRoi(sz, K, _cameras[i].R);
		_corners[i] = roi.tl();
		_sizes[i] = roi.size();
	}

	Mat img;
	Mat mask;
	_masksWarpedCompose.resize(_numImages);
	_xmap.resize(_numImages);
	_ymap.resize(_numImages);

#ifdef WITH_CUDA
	_xmap_gpu.resize(_numImages);
	_ymap_gpu.resize(_numImages);
#endif

	for (int img_idx = 0; img_idx < _numImages; ++img_idx)
	{
		if (abs(_composeScale - 1) > 1e-1)
			resize(_images[img_idx], img, Size(), _composeScale, _composeScale);
		else
			img = _images[img_idx];

		Mat K;
		_cameras[img_idx].K().convertTo(K, CV_32F);

		// Prepare warping maps for the current image
		//warper->warp(img, K, _cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
		//LOGLN("#" << img_idx << " K:\n" << K);
		//LOGLN("#" << img_idx << " R:\n" << _cameras[img_idx].R);
		warper->buildMaps(img.size(), K, _cameras[img_idx].R, _xmap[img_idx], _ymap[img_idx]);
#ifdef WITH_CUDA
		_xmap_gpu[img_idx].upload(_xmap[img_idx]);
		_ymap_gpu[img_idx].upload(_ymap[img_idx]);
#endif
		// Warp the current image mask
		mask.create(img.size(), CV_8U);
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, _cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, _masksWarpedCompose[img_idx]);
		//std::stringstream ss;
		//ss << "mask_c" << img_idx << ".jpg";
		//imwrite(ss.str(), _masksWarpedCompose[img_idx]);

		mask.release();
		img.release();
	}
	return true;
}

bool Stitcher::PrepareBlend(int blendType, float blendStrength)
{
	if (_corners.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No mask corners. Run WarpAux before");
		return false;
	}
	if (_sizes.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No warped sizes. Run PrepareWarp before");
		return false;
	}

	_blendType = blendType;
	_blendStrength = blendStrength;

#ifdef WITH_CUDA
	_customBlender = new CustomBlender();
#else
	if (_blender.empty())
	{
		_blender = Blender::createDefault(blendType, true);
		Size dst_sz = resultRoi(_corners, _sizes).size();
		float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blendStrength / 100.f;
		if (blend_width < 1.f)
			_blender = Blender::createDefault(Blender::NO, true); //tryToUseGPU (try_gpu) true
		else if (blendType == Blender::MULTI_BAND)
		{
			MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(_blender.get()));
			mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
			//LOGLN("Multi-band blender, number of bands: " << mb->numBands());
		}
		else if (blendType == Blender::FEATHER)
		{
			FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(_blender.get()));
			fb->setSharpness(1.f / blend_width);
			//LOGLN("Feather blender, sharpness: " << fb->sharpness());
		}
	}
#endif
	return true;
}

#ifdef WITH_CUDA
bool Stitcher::Compose(std::vector<cuda::GpuMat> images, Mat& result, Mat& resultMask, bool showOverviewInPanorama, bool lowQualityStitching)
{

#if ENABLE_LOG
	cout<<"Starting composing with GPU..."<<endl;
#endif

	if (_numImagesTot <= 0)
	{
		LOGLN("[Stitcher] ERROR: Invalid stitcher configuration. Did you invoke the constructor properly?");
		return false;
	}
	if (_validIndices.size() == 0)
	{
		LOGLN("[Stitcher] ERROR: Not valid images. Try running MatchFeatures before");
		return false;
	}
	if (images.size() != _numImagesTot)
	{
		LOGLN("[Stitcher] ERROR: Wrong image number. Expected " << _numImagesTot << ", got " << images.size());
		return false;
	}
#ifndef WITH_CUDA
	if (_blender.empty())
	{
		LOGLN("[Stitcher] ERROR: No instantiated blender. Run PrepareBlend before");
		return false;
	}
#endif
	if (_compensator.empty())
	{
		LOGLN("[Stitcher] ERROR: No instantiated exposure compensator. Run PrepareExposureCompensator before");
		return false;
	}
	if (_composeScale <= 0)
	{
		LOGLN("[Stitcher] ERROR: No compose scale set. Run Prepare before");
		return false;
	}
	if (_xmap.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No warping maps. Run PrepareWarp before");
		return false;
	}
	if (_ymap.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No warping maps. Run PrepareWarp before");
		return false;
	}
	if (_corners.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No mask corners. Run WarpAux before");
		return false;
	}
	if (_sizes.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No warped sizes. Run PrepareWarp before");
		return false;
	}
	if (_masksWarpedCompose.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No warped masks. Run PrepareWarp before");
		return false;
	}
	if (_masksWarpedSeam.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No warped masks. Run WarpAux before");
		return false;
	}

	_images_gpu.clear();
	for (size_t i = 0; i < _validIndices.size(); ++i)
	{
		if (images[_validIndices[i]].empty())
		{
			LOGLN("[Stitcher] ERROR: Invalid image #" << i);
			return false;
		}
		_images_gpu.push_back(images[_validIndices[i]]);
	}

	Mat img_warped, img_warped_s;
	Mat dilated_mask, seam_mask;
	cuda::GpuMat full_img, img;

#if ENABLE_LOG
	clock_t t_begin = clock();
#endif

#ifdef WITH_CUDA
	_customBlender->prepare(_corners, _sizes);
#else
	_blender->prepare(_corners, _sizes);
#endif

#if ENABLE_LOG
	clock_t t_end = clock();
	cout << "Time blender prepare: " << double(t_end - t_begin) / CLOCKS_PER_SEC << endl;

	double remapTime = 0;
	double compTime = 0;
	double blendTime = 0;
	int64 t;
#endif
	clock_t t_begin_ciclo = clock();

	for (int img_idx = 0; img_idx < _numImages; ++img_idx)
	{
		if (!showOverviewInPanorama) {
			if (img_idx == 0) {
				continue;
			}
		}

#if ENABLE_LOG
		cout << "   Working on image: " << img_idx << " " << _images_gpu[img_idx].cols << "x" << _images_gpu[img_idx].rows << endl;
#endif

		full_img = _images_gpu[img_idx];

		if (abs(_composeScale - 1) > 1e-1)
			cuda::resize(full_img, img, Size(), _composeScale, _composeScale);
		else
			img = full_img;
		full_img.release();

		// Warp the current image
		//warper->warp(img, K, _cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

#if ENABLE_LOG
		t = getTickCount();
#endif
		cuda::GpuMat img_warped_gpu;

#if ENABLE_LOG
		t_begin = clock();
#endif

		cuda::remap(img, img_warped_gpu, _xmap_gpu[img_idx], _ymap_gpu[img_idx], INTER_LINEAR, BORDER_REFLECT);

#if ENABLE_LOG
		t_end = clock();
		cout << "      Ciclo::1 - remap img: " << double(t_end - t_begin) / CLOCKS_PER_SEC << endl;

		t_begin = clock();
#endif


//#ifndef WITH_CUDA
		if (!lowQualityStitching)
		{

			img_warped_gpu.download(img_warped);

			//img_warped_gpu.release();
			img.release();

#if ENABLE_LOG
			t_end = clock();
			cout << "      Ciclo::1.1 - download img_warp + some release: " << double(t_end - t_begin) / CLOCKS_PER_SEC << endl;

			remapTime += ((getTickCount() - t) / getTickFrequency());
			t = getTickCount();
			// Compensate exposure
			t_begin = clock();
#endif
			//cv::Mat thresh_result;
			//cv::threshold(_masksWarpedCompose[img_idx], thresh_result, 0, 255, CV_THRESH_BINARY);
			//cv::imshow("test window _masksWarpedCompose", thresh_result);
			//cv::waitKey(0);

			_compensator->apply(img_idx, _corners[img_idx], img_warped, _masksWarpedCompose[img_idx]);

#if ENABLE_LOG
			t_end = clock();
			cout << "      Ciclo::2 - compensate exposure: " << double(t_end - t_begin) / CLOCKS_PER_SEC << endl;
			compTime += ((getTickCount() - t) / getTickFrequency());
#endif

			//img_warped.convertTo(img_warped_s, CV_16S);
			img_warped_gpu.upload(img_warped);

#if ENABLE_LOG
			t_begin = clock();
#endif

			dilate(_masksWarpedSeam[img_idx], dilated_mask, Mat());
			resize(dilated_mask, seam_mask, _masksWarpedCompose[img_idx].size());
			_masksWarpedCompose[img_idx] = seam_mask & _masksWarpedCompose[img_idx];


#if ENABLE_LOG
			t_end = clock();
			cout << "      Ciclo::3 - dilate masks: " << double(t_end - t_begin) / CLOCKS_PER_SEC << endl;
#endif

			dilated_mask.release();
			seam_mask.release();

			// Blend the current image
#if ENABLE_LOG
			t = getTickCount();
			t_begin = clock();
#endif

	}

//#endif

#ifdef WITH_CUDA

		cv::cuda::GpuMat masksWarpedCompose_cuda, img_warped_gpu_s;
		img_warped_gpu.convertTo(img_warped_gpu_s, CV_16SC3);
		masksWarpedCompose_cuda.upload(_masksWarpedCompose[img_idx]);

		_customBlender->feed(img_warped_gpu_s, masksWarpedCompose_cuda, _corners[img_idx]);

		img_warped_gpu.release();
		img_warped_gpu_s.release();
		masksWarpedCompose_cuda.release();
#else
		_blender->feed(img_warped_s, _masksWarpedCompose[img_idx], _corners[img_idx]);
#endif

#if ENABLE_LOG
		t_end = clock();
		cout << "      Ciclo::4 - blender feed: " << double(t_end - t_begin) / CLOCKS_PER_SEC << endl;

		blendTime += ((getTickCount() - t) / getTickFrequency());
#endif

		img_warped_s.release();

	}

#if ENABLE_LOG
	clock_t t_end_ciclo = clock();
	cout << "Totale tempo ciclo: " << double(t_end_ciclo - t_begin_ciclo) / CLOCKS_PER_SEC << endl;

	t = getTickCount();
	t_begin = clock();
#endif

#ifdef WITH_CUDA
	cv::cuda::GpuMat result_cuda, resultMask_cuda;
	result_cuda.upload(result);
	resultMask_cuda.upload(resultMask);
	_customBlender->blend(result_cuda, resultMask_cuda);

	result_cuda.download(result);
	resultMask_cuda.download(resultMask);
#else
	_blender->blend(result, resultMask);
#endif

#if ENABLE_LOG
	t_end = clock();
	cout << "time blender::blend: " << double(t_end - t_begin) / CLOCKS_PER_SEC << endl;

	LOGLN("      Remap, time: " << remapTime << " sec");
	LOGLN("      Exposure compensation, time: " << compTime << " sec");
	blendTime += ((getTickCount() - t) / getTickFrequency());
	LOGLN("      Blend, time: " << blendTime << " sec");
#endif

	//cv::Mat result_normal;
	//result.convertTo(result_normal, CV_8UC3);
	//cv::imwrite("result.png", cv::Mat(result_normal));
	//cv::imshow("result", cv::Mat(result_normal));
	//cv::waitKey(0);

	//cv::Mat result_mask_temp;
	//cv::threshold(resultMask, result_mask_temp, 0, 255, CV_THRESH_BINARY);
	//cv::imwrite("result_mask_temp.png", cv::Mat(result_mask_temp));
	//cv::imshow("result_mask_temp", cv::Mat(result_mask_temp));
	//cv::waitKey(0);

	return true;
}
#else
bool Stitcher::Compose(std::vector<Mat> images, Mat& result, Mat& resultMask, bool showOverviewInPanorama)
{
	if (_numImagesTot <= 0)
	{
		LOGLN("[Stitcher] ERROR: Invalid stitcher configuration. Did you invoke the constructor properly?");
		return false;
	}
	if (_validIndices.size() == 0)
	{
		LOGLN("[Stitcher] ERROR: Not valid images. Try running MatchFeatures before");
		return false;
	}
	if (images.size() != _numImagesTot)
	{
		LOGLN("[Stitcher] ERROR: Wrong image number. Expected " << _numImagesTot << ", got " << images.size());
		return false;
	}
	if (_blender.empty())
	{
		LOGLN("[Stitcher] ERROR: No instantiated blender. Run PrepareBlend before");
		return false;
	}
	if (_compensator.empty())
	{
		LOGLN("[Stitcher] ERROR: No instantiated exposure compensator. Run PrepareExposureCompensator before");
		return false;
	}
	if (_composeScale <= 0)
	{
		LOGLN("[Stitcher] ERROR: No compose scale set. Run Prepare before");
		return false;
	}
	if (_xmap.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No warping maps. Run PrepareWarp before");
		return false;
	}
	if (_ymap.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No warping maps. Run PrepareWarp before");
		return false;
	}
	if (_corners.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No mask corners. Run WarpAux before");
		return false;
	}
	if (_sizes.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No warped sizes. Run PrepareWarp before");
		return false;
	}
	if (_masksWarpedCompose.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No warped masks. Run PrepareWarp before");
		return false;
	}
	if (_masksWarpedSeam.size() < _numImages)
	{
		LOGLN("[Stitcher] ERROR: No warped masks. Run WarpAux before");
		return false;
	}

	_images.clear();
	for (size_t i = 0; i < _validIndices.size(); ++i)
	{
		if (images[_validIndices[i]].empty())
		{
			LOGLN("[Stitcher] ERROR: Invalid image #" << i);
			return false;
		}
		_images.push_back(images[_validIndices[i]]);
	}

	Mat img_warped, img_warped_s;
	Mat dilated_mask, seam_mask;
	Mat full_img, img;

#if ENABLE_LOG
	clock_t t_begin = clock();
#endif

	_blender->prepare(_corners, _sizes);

#if ENABLE_LOG
	clock_t t_end = clock();
	cout << "time blender prepare: " << double(t_end - t_begin) / CLOCKS_PER_SEC << endl;
#endif

#if ENABLE_LOG
	double remapTime = 0;
	double compTime = 0;
	double blendTime = 0;
	int64 t;
#endif
	clock_t t_begin_ciclo = clock();

	//	vector<thread> th_composing;
	//	for (int thread_idx=0;thread_idx<_numImages;++thread_idx){
	//		th_composing.push_back(thread(ThreadFeedBlending,thread_idx));
	//	}
	//
	//	//TODO: thread join.

	for (int img_idx = 0; img_idx < _numImages; ++img_idx)
	{
		if (!showOverviewInPanorama) {
			if (img_idx == 0) {
				continue;
			}
		}

		full_img = _images[img_idx];

		if (abs(_composeScale - 1) > 1e-1)
			cv::resize(full_img, img, Size(), _composeScale, _composeScale);
		else
			img = full_img;
		full_img.release();

		// Warp the current image
		//warper->warp(img, K, _cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

#if ENABLE_LOG
		t = getTickCount();
#endif

#if ENABLE_LOG
		t_begin = clock();
#endif

		remap(img, img_warped, _xmap[img_idx], _ymap[img_idx], INTER_LINEAR, BORDER_REFLECT);
#if ENABLE_LOG
		t_end = clock();
		std::cout << "time remap gpu: " << (t_end - t_begin) / (double)CLOCKS_PER_SEC << std::endl;
#endif	

		img.release();

#if ENABLE_LOG
		remapTime += ((getTickCount() - t) / getTickFrequency());
		t = getTickCount();
#endif
		// Compensate exposure
		//	std::cout << "inizio compensator apply" << std::endl;
		_compensator->apply(img_idx, _corners[img_idx], img_warped, _masksWarpedCompose[img_idx]);

#if ENABLE_LOG
		compTime += ((getTickCount() - t) / getTickFrequency());
#endif
		img_warped.convertTo(img_warped_s, CV_16S);

		dilate(_masksWarpedSeam[img_idx], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, _masksWarpedCompose[img_idx].size());
		_masksWarpedCompose[img_idx] = seam_mask & _masksWarpedCompose[img_idx];
		dilated_mask.release();
		seam_mask.release();

		// Blend the current image
#if ENABLE_LOG
		t = getTickCount();
		std::cout << "inizio feed" << std::endl;
#endif

		_blender->feed(img_warped_s, _masksWarpedCompose[img_idx], _corners[img_idx]);

#if ENABLE_LOG
		blendTime += ((getTickCount() - t) / getTickFrequency());
#endif
		img_warped_s.release();
	}


#if ENABLE_LOG
	clock_t t_end_ciclo = clock();
	cout << "time ciclo immagini: " << double(t_end_ciclo - t_begin_ciclo) / CLOCKS_PER_SEC << endl;

	t = getTickCount();
	t_begin = clock();
#endif

	_blender->blend(result, resultMask);

#if ENABLE_LOG	
	t_end = clock();
	cout << "time blender blend: " << double(t_end - t_begin) / CLOCKS_PER_SEC << endl;

	LOGLN("Remap, time: " << remapTime << " sec");
	LOGLN("Exposure compensation, time: " << compTime << " sec");
	blendTime += ((getTickCount() - t) / getTickFrequency());
	LOGLN("Blend, time: " << blendTime << " sec");
#endif

	return true;

}
#endif

void Stitcher::Serialize(char* buffer, int& size)
{
//#if ENABLE_LOG
//	int64 t_tot = getTickCount();
//#endif

	int t;
	unsigned char* p;
	size = 0;

	// _numImagesTot
	memcpy(buffer + size, &_numImagesTot, sizeof(int));
	size += sizeof(int);

	// _validIndices
	size_t n = _validIndices.size();
	memcpy(buffer + size, &n, sizeof(size_t));
	size += sizeof(size_t);
	for (auto v : _validIndices)
	{
		memcpy(buffer+size, &v, sizeof(int));
		size += sizeof(int);
	}

	// blender
	// Save only blendType, blendStrength, corners and sizes and then re-alloc the blender object
	memcpy(buffer + size, &_blendType, sizeof(int));
	size += sizeof(int);
	memcpy(buffer + size, &_blendStrength, sizeof(float));
	size += sizeof(float);
	// corners
	n = _corners.size();
	memcpy(buffer + size, &n, sizeof(size_t));
	size += sizeof(size_t);
	for (auto c : _corners)
	{
		t = c.x;
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		t = c.y;
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
	}
	// sizes
	n = _sizes.size();
	memcpy(buffer + size, &n, sizeof(size_t));
	size += sizeof(size_t);
	for (auto s : _sizes)
	{
		t = s.height;
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		t = s.width;
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
	}

	// compensator
	// Save only exposCompType, corners, imagesWarped and maskWarpedSeam and then re-alloc the compensator object
	memcpy(buffer + size, &_exposCompType, sizeof(int));
	size += sizeof(int);
	n = _imagesWarped.size();
	memcpy(buffer + size, &n, sizeof(size_t));
	size += sizeof(size_t);
	for (auto m : _imagesWarped)
	{
		// For each image, save w, h, type, data
		t = m.cols;
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		t = m.rows;
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		t = m.type();
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		cv::Mat tmp_m = m.getMat(0);
		p = tmp_m.data; //m.data;
		memcpy(buffer + size, p, m.rows*m.step);
		size += m.rows*(int)m.step;
	}

	// mask warped seam
	n = _masksWarpedSeam.size();
	memcpy(buffer + size, &n, sizeof(size_t));
	size += sizeof(size_t);
	for (auto m : _masksWarpedSeam)
	{
		// For each image, save w, h, type, data
		t = m.cols;
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		t = m.rows;
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		t = m.type();
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		cv::Mat tmp_m = m.getMat(0);
		p = tmp_m.data; //m.data;
		memcpy(buffer + size, p, m.rows*m.step);
		size += m.rows*(int)m.step;
	}
	
	// compose scale
	memcpy(buffer+size, &_composeScale, sizeof(double));
	size += sizeof(double);

	// xmap, ymap
	n = _xmap.size();
	memcpy(buffer + size, &n, sizeof(size_t));
	size += sizeof(size_t);
	for (auto m : _xmap)
	{
		// For each image, save w, h, type, data
		t = m.cols;
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		t = m.rows;
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		t = m.type();
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		p = m.data;
		memcpy(buffer + size, p, m.rows*m.step);
		size += m.rows*(int)m.step;
	}

	n = _ymap.size();
	memcpy(buffer + size, &n, sizeof(size_t));
	size += sizeof(size_t);
	for (auto m : _ymap)
	{
		// For each image, save w, h, type, data
		t = m.cols;
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		t = m.rows;
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		t = m.type();
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		p = m.data;
		memcpy(buffer + size, p, m.rows*m.step);
		size += m.rows*(int)m.step;
	}

	// mask warped compose
	n = _masksWarpedCompose.size();
	memcpy(buffer + size, &n, sizeof(size_t));
	size += sizeof(size_t);
	for (auto m : _masksWarpedCompose)
	{
		// For each image, save w, h, type, data
		t = m.cols;
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		t = m.rows;
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		t = m.type();
		memcpy(buffer + size, &t, sizeof(int));
		size += sizeof(int);
		p = m.data;
		memcpy(buffer + size, p, m.rows*m.step);
		size += m.rows*(int)m.step;
	}

//#if ENABLE_LOG
//	LOGLN("Stitcher serialization, time: " << ((getTickCount() - t_tot) / getTickFrequency()) << " sec");
//#endif
}

bool Stitcher::Deserialize(const char* buffer, int size)
{
//#if ENABLE_LOG
//	int64 t_tot = getTickCount();
//#endif

	size_t n;
	int t;
	int w, h;

	// _numImagesTot
	if (size < sizeof(int))
		return false;
	memcpy(&_numImagesTot, buffer, sizeof(int));
	buffer += sizeof(int);
	size -= sizeof(int);

	// _validIndices
	if (size < sizeof(size_t))
		return false;
	memcpy(&n, buffer, sizeof(size_t));
	buffer += sizeof(size_t);
	size -= sizeof(size_t);
	if (size < n*sizeof(int))
		return false;
	_validIndices.clear();
	for (int i = 0; i < n; i++)
	{
		memcpy(&t, buffer, sizeof(int));
		_validIndices.push_back(t);
		buffer += sizeof(int);
		size -= sizeof(int);
	}
	if (n > 0)
		_numImages = (int)n;

	// blender
	// Save only blendType, blendStrength, corners and sizes and then re-alloc the blender object
	if (size < 2*sizeof(int)+sizeof(float))
		return false;
	memcpy(&_blendType, buffer, sizeof(int));
	buffer += sizeof(int);
	size -= sizeof(int);
	memcpy(&_blendStrength, buffer, sizeof(float));
	buffer += sizeof(float);
	size -= sizeof(float);
	// corners
	if (size < sizeof(size_t))
		return false;
	memcpy(&n, buffer, sizeof(size_t));
	buffer += sizeof(size_t);
	size -= sizeof(size_t);
	if (size < n * 2 * sizeof(int))
		return false;
	Point c;
	_corners.clear();
	for (int i = 0; i < n; i++)
	{
		memcpy(&t, buffer, sizeof(int));
		c.x = t;
		buffer += sizeof(int);
		size -= sizeof(int);
		memcpy(&t, buffer, sizeof(int));
		c.y = t;
		buffer += sizeof(int);
		size -= sizeof(int);
		_corners.push_back(c);
	}
	// sizes
	if (size < sizeof(size_t))
		return false;
	memcpy(&n, buffer, sizeof(size_t));
	buffer += sizeof(size_t);
	size -= sizeof(size_t);
	if (size < n * 2 * sizeof(int))
		return false;
	Size s;
	_sizes.clear();
	for (int i = 0; i < n; i++)
	{
		memcpy(&t, buffer, sizeof(int));
		s.height = t;
		buffer += sizeof(int);
		size -= sizeof(int);
		memcpy(&t, buffer, sizeof(int));
		s.width = t;
		buffer += sizeof(int);
		size -= sizeof(int);
		_sizes.push_back(s);
	}
#if ENABLE_LOG
	LOGLN("Prepare blend...");
	int64 tick = getTickCount();
#endif
	bool result = PrepareBlend(_blendType, _blendStrength);
#if ENABLE_LOG
	LOGLN("Prepare blend, time: " << ((getTickCount() - tick) / getTickFrequency()) << " sec");
#endif
	if (!result)
	{
#ifdef WITH_CUDA
		_customBlender->~CustomBlender();
#else
		//_blender = NULL;
		_blender.release();
#endif
		return false;
	}

	// compensator
	// Save only exposCompType, corners, imagesWarped and maskWarpedSeam and then re-alloc the compensator object
	if (size < sizeof(int))
		return false;
	memcpy(&_exposCompType, buffer, sizeof(int));
	buffer += sizeof(int);
	size -= sizeof(int);
	if (size < sizeof(size_t))
		return false;
	memcpy(&n, buffer, sizeof(size_t));
	buffer += sizeof(size_t);
	size -= sizeof(size_t);
	_imagesWarped.clear();
	for (int i = 0; i < n; i++)
	{
		if (size < 3*sizeof(int))
			return false;
		// For each image, read w, h, type, data
		memcpy(&w, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		memcpy(&h, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		memcpy(&t, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		cv::Mat m(h, w, t);
		if (size < m.rows*m.step)
		{
			m.release();
			return false;
		}
		memcpy(m.data, buffer, m.rows*m.step);
		buffer += m.rows*m.step;
		size -= m.rows*(int)m.step;
		cv::UMat tmp_m = m.getUMat(0);
		//_imagesWarped.push_back(m);
		_imagesWarped.push_back(tmp_m);
	}
	// mask warped seam
	if (size < sizeof(size_t))
		return false;
	memcpy(&n, buffer, sizeof(size_t));
	buffer += sizeof(size_t);
	size -= sizeof(size_t);
	_masksWarpedSeam.clear();
	for (int i = 0; i < n; i++)
	{
		if (size < 3 * sizeof(int))
			return false;
		// For each image, read w, h, type, data
		memcpy(&w, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		memcpy(&h, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		memcpy(&t, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		cv::Mat m(h, w, t);
		if (size < m.rows*m.step)
		{
			m.release();
			return false;
		}
		memcpy(m.data, buffer, m.rows*m.step);
		buffer += m.rows*m.step;
		size -= m.rows*(int)m.step;
		cv::UMat tmp_m = m.getUMat(0);
		//_masksWarpedSeam.push_back(m);
		_masksWarpedSeam.push_back(tmp_m);
	}
#if ENABLE_LOG
	LOGLN("Prepare exposure compensator...");
	tick = getTickCount();
#endif
	result = PrepareExposureCompensator(_exposCompType);
#if ENABLE_LOG
	LOGLN("Prepare exposure compensator, time: " << ((getTickCount() - tick) / getTickFrequency()) << " sec");
#endif

	if (!result)
	{
//		_compensator = NULL;
		_compensator.release();
		return false;
	}

	// compose scale
	if (size < sizeof(double))
		return false;
	memcpy(&_composeScale, buffer, sizeof(double));
	buffer += sizeof(double);
	size -= sizeof(double);

	// xmap, ymap
	if (size < sizeof(size_t))
		return false;
	memcpy(&n, buffer, sizeof(size_t));
	buffer += sizeof(size_t);
	size -= sizeof(size_t);
	_xmap.clear();

#ifdef WITH_CUDA
	_xmap_gpu.clear();
#endif
	//aggiunta gpumat
	for (int i = 0; i < n; i++)
	{
		if (size < 3 * sizeof(int))
			return false;
		// For each image, read w, h, type, data
		memcpy(&w, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		memcpy(&h, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		memcpy(&t, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		cv::Mat m(h, w, t);
		if (size < m.rows*m.step)
		{
			m.release();
			return false;
		}
		memcpy(m.data, buffer, m.rows*m.step);
		buffer += m.rows*m.step;
		size -= m.rows*(int)m.step;
		_xmap.push_back(m);
	#ifdef WITH_CUDA
		cuda::GpuMat m_gpu(m);
		_xmap_gpu.push_back(m_gpu);
	#endif
	}

	if (size < sizeof(size_t))
		return false;
	memcpy(&n, buffer, sizeof(size_t));
	buffer += sizeof(size_t);
	size -= sizeof(size_t);
	_ymap.clear();
#ifdef WITH_CUDA
	_ymap_gpu.clear();
#endif
	for (int i = 0; i < n; i++)
	{
		if (size < 3 * sizeof(int))
			return false;
		// For each image, read w, h, type, data
		memcpy(&w, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		memcpy(&h, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		memcpy(&t, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		cv::Mat m(h, w, t);
		if (size < m.rows*m.step)
		{
			m.release();
			return false;
		}
		memcpy(m.data, buffer, m.rows*m.step);
		buffer += m.rows*m.step;
		size -= m.rows*(int)m.step;
		_ymap.push_back(m);
#ifdef WITH_CUDA
		cuda::GpuMat m_gpu(m);
		_ymap_gpu.push_back(m_gpu);
#endif
	}

	// mask warped compose
	if (size < sizeof(size_t))
		return false;
	memcpy(&n, buffer, sizeof(size_t));
	buffer += sizeof(size_t);
	size -= sizeof(size_t);
	_masksWarpedCompose.clear();
	for (int i = 0; i < n; i++)
	{
		if (size < 3 * sizeof(int))
			return false;
		// For each image, read w, h, type, data
		memcpy(&w, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		memcpy(&h, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		memcpy(&t, buffer, sizeof(int));
		buffer += sizeof(int);
		size -= sizeof(int);
		cv::Mat m(h, w, t);
		if (size < m.rows*m.step)
		{
			m.release();
			return false;
		}
		memcpy(m.data, buffer, m.rows*m.step);
		buffer += m.rows*m.step;
		size -= m.rows*(int)m.step;
		_masksWarpedCompose.push_back(m);
	}
//#if ENABLE_LOG
//	LOGLN("Stitcher deserialization, time: " << ((getTickCount() - t_tot) / getTickFrequency()) << " sec");
//#endif
	return true;
}

void Stitcher::DeleteBlender() {
	delete _customBlender;
}
