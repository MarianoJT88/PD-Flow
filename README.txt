===========================================================================================
                        Primal-Dual Scene Flow for RGB-D cameras
===========================================================================================
This code contains an algorithm implemented on GPU to compute scene flow in real-time.
It has been tested on Windows 7 and Ubuntu 14.04.


                             Configuration and Generation
-------------------------------------------------------------------------------------------
A CMakeLists.txt file is included to detect external dependencies and generate the project automatically. CUDA, OpenCV, MRPT and OpenNI2 are used, but not all are required (only CUDA is required, together with a GPU with CUDA capability of at least 2.0). 
The project builds a "cudalib" embedding the scene flow algorithm and 2 different applications to evaluate and visualize it.

The first one, called "Scene-Flow-Impair" reads two RGB-D and generates an OpenCV RGB visualization of the scene flow together with a .txt file where the scene flow estimate is saved. 
This application requires CUDA and OpenCV to work, and is built or not depending on the CMake variable "BUILD_EVALUATOR".

The second one, called "Scene-Flow-Visualization" computes scene flow in real-time from video stream provided by an RGB-D camera, and shows a 3D visualization of the motion field.
This application requires CUDA, MRPT and OpenNI2 to work, and is built or not depending on the CMake variable "BUILD_RT_VISUALIZATION".

CUDA, OpenCV and MRPT should be easily found by CMake, but OpenNI2 might be troublesome and could the user might need to modify the CMakeList.txt file according to its own configuration of OpenNI2.

In case you don't know MRPT I encourage you to have a look at its website here: http://www.mrpt.org/
Detailed instructions about how to install it (or some of its modules) can be found here: http://www.mrpt.org/download-mrpt/


                                   Compiling
-------------------------------------------------------------------------------------------
The compiling process should be straightforward. The only problem might arise from headers that are included but the compiler cannot find. In this case you should find those files on your computer and include them with the correct path (for your machine).


                                     Usage
-------------------------------------------------------------------------------------------
Both apps can read command line arguments. By adding the argument "--help" you will obtain a list of the accepted input arguments:

--help: To show the list of arguments...

--cam_mode cm: To open the RGB-D camera with VGA (cm = 1) or QVGA (cm = 2) resolution.
               Only for "Scene-Flow-Visualization"
				
--fps f: The desired scene flow frame rate (Hz), it  might be not achievable!! 
         Only for "Scene-Flow-Visualization"
				
--rows r: Number of rows at the finest level of the pyramid
          Options: r=15, r=30, r=60, r=120, r=240, r=480 (if VGA)

The coarsest level of the pyramid is always 15 x 20, so the total number of coarse-to-fine levels will depend on the parameter "rows". By default, rows = 240.

The images that "Scene-Flow-Impair" reads can be specified on the command line 

--i1 <filename> : The file name of the first intensity image. Defaults to i1.png

--i2 <filename> : The file name of the second intensity image. Defaults to i2.png

--z1 <filename> : The file name of the first depth image. Defaults to z1.png

--z2 <filename> : The file name of the second depth image. Defaults to z2.png

Note these names are case sensitive and the files must be located in the same directory as the executable unless an absolute path is specified. Furthermore, they must be saved with the following format:

intensity images - 8 bit in PNG. Resolution of VGA or QVGA
                   Clue: Use cv::Mat image_name(height, width, CV_8U) and
                         cv::imwrite(filename, image_name) to store them
					
depth images - 16 bit monochrome in PNG, scaled by 5000. Resolution of VGA or QVGA
               Clue: Use cv::Mat image_name(height, width, CV_16U) and
                     cv::imwrite(filename, image_name) to store them.
                     Multiply the real depth by 5000.

Scene-Flow_Impair writes outputs to two files, a text version containing the full scene flow and an image representation. The root of these files can be specified using 

--out <root>    : The output file name root. Defaults to pdflow. 
                  The results will be written to <root>_resultsNN.txt and root_representationNN.png 
                  where NN is a two digit number.

--no-show       : Supresses display of images so that the code can be run in batch mode. Output files are still generated


The algorithm convergence is set to a fixed number of iterations at each level of the coarse-to-fine scheme, and depends on the amount of levels and the level itself. If necessary, it can be changed by modifying the variable num_max_iter[].

The number of Threads and Blocks that the GPU utilizes is also set to a fixed value. You can adapt it to your own device (GPU) to get the best performance. In any case, the number of threads should be higher than 25.


The provided code is published under the General Public License Version 3 (GPL v3). More information can be found in the "GPU LICENSE.txt" also included in the repository


