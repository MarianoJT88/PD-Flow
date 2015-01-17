			Primal-Dual Scene Flow for RGB-D cameras
========================================================================
This code contain an algorithm implemented on GPU to compute scene flow in real-time.
It has been tested on Windows 7 and Ubuntu 14.04.


				Project configuration and generation
------------------------------------------------------------------------
A CMakeLists.txt file is included to detect dependencies and generate the project automatically.
For external dependencies are used, but not all are required (only CUDA is required, and a GPU with CUDA capability of at least 2.0). The project builds a "cudalib" file embedding the scene flow algorithm and 2 different applications to evaluate and visualize it.

The first one, called "Scene-Flow-Impair" reads two RGB-D frames from the same directory of the executable file and generates an OpenCV RGB visualization of the scene flow together with a .txt file where the scene flow estimate is saved. 
This application requires CUDA and OpenCV to work, and is built or not depending on the CMake variable "BUILD_EVALUATOR".

The second one, called "Scene-Flow-Visualization" computes scene flow in real-time using the images provided by an RGB-D camera, and shows a 3D visualization of the motion field.
This application requires CUDA, MRPT and OpenNI2 to work, and is built or not depending on the CMake variable "BUILD_RT_VISUALIZATION".

CUDA, OpenCV and MRPT should be easily found by CMake, but OpenNI2 might be troublesome and could require the user to modify the CMakeList.txt file according to its own configuration of OpenNI2.

In case you don't know MRPT I encourage you to have a look at its website here: http://www.mrpt.org/
Detailed instructions about how to install it can be found here: http://www.mrpt.org/download-mrpt/


							Compiling
-----------------------------------------------------------------------
The compiling process should be straightforward. I guess the only problem might arise from headers that are included but the compiler cannot find. In this case you should find those files on your computer and include them with the correct address (for your machine).


								Usage
-----------------------------------------------------------------------
Both apps can read command line arguments. By adding the argument "--help" you will obtain a list of the accepted input arguments:

--help: To show the list of arguments...

--cam_mode cm: To open the RGB-D camera with VGA (cm = 1) or QVGA (cm = 2) resolution.
				Only for "Scene-Flow-Visualization"
				
--fps f: The desired scene flow frame rate (Hz) 
				Only for "Scene-Flow-Visualization"
				
--rows r: Number of rows at the finest level of the pyramid
			Options: r=15, r=30, r=60, r=120, r=240, r=480 (if VGA)
			

The coarsest level of the pyramid is always 15 x 20, so the total number of coarse-to-fine levels will depend on the parameter "rows". By default, rows = 240 and there are 5 levels.

The images that "Scene-Flow-Impair" must be named as "I1" (first intensity image), "I2" (second intensity image", "Z1" (first depth image) and "Z2" (second depth image) and must be located in the same directory than the executable. Furthermore, they must be saved with the following format:

intensity images - 8 bit in PNG format. Resolution of VGA or QVGA
					Use cv::Mat image_name(height, width, CV_8U) and     cv::imwrite(filename, image_name) to store them
					
depth images - 16 bit monochrome in PNG, scaled by 5000. Resolution of VGA or QVGA
					Use cv::Mat image_name(height, width, CV_16U) and
					cv::imwrite(filename, image_name) to store them.


The algorithm convergence is set to a fixed number of iterations, which increases with the level resolution. You can change it at your convenience with the variable num_max_iter[].

The number of Threads and Blocks that the GPU utilizes is also set to a fixed value. You can adapt it to your own device to get the best performance. In any case, the number of threads should be higher than 25.


The provided code is published under the General Public License Version 3 (GPL v3). More information can be found in the "GPU LICENSE.txt" file also included in the repository


