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




