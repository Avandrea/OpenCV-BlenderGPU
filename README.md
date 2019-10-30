# OpenCV-BlenderGPU

This is an implementation of opencv blenders class using CUDA and gpu.  
The original OpenCV stitcher class does not implement the functions prepare, feed and blend using CUDA. The class customBlender allow to execute those functions using the gpu.  
Note that the input images for the functions feed() and blend() must be GpuMat.

### Prerequisites

* OpenCV 3.2 is required (it will probably work with higher versions, but not tested)  
* CUDA 8 is required (it will probably work with higher versions, but not tested)

## Authors

* **Andrea Avigni**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


////////////////// WIP
Update with new versions
cuda 10.1 update 2
OpenCV 4.0.0

BUILD
Cmake -> configure visual studio 2017 Win64
add extra modules path
check WITH_CUDA and BUILD_EXAMPLES and press configure again
remove unwanted CUDA_ARCH_BIN (in case of gtx 1080 leave 6.1) and set CUDA_ARCH_PTX to 7.5
generate

Open Visual Studio solution and build in release

If cudev fails for an external dependancy add ..\..\lib\Release\opencv_core400[d].lib to the dependancies of the project cudev


TEST
try to use stitching sample
set OPENCV_SAMPLES_DATA_PATH to the path of the images img1 e img2
Go to C:\Libs\opencv-4.0.0\build\bin\Release
Launch .\example_cpp_stitching.exe --d3 --mode panorama img1.jpeg img2.jpeg

EXAMPLE
include opencv add to dependancies
add opencv bin to environmental path