/*****************************************************************************
**				Primal-Dual Scene Flow for RGB-D cameras					**
**				----------------------------------------					**
**																			**
**	Copyright(c) 2015, Mariano Jaimez Tarifa, University of Malaga			**
**	Copyright(c) 2015, Mohamed Souiai, Technical University of Munich		**
**	Copyright(c) 2015, MAPIR group, University of Malaga					**
**	Copyright(c) 2015, Computer Vision group, Tech. University of Munich	**
**																			**
**  This program is free software: you can redistribute it and/or modify	**
**  it under the terms of the GNU General Public License (version 3) as		**
**	published by the Free Software Foundation.								**
**																			**
**  This program is distributed in the hope that it will be useful, but		**
**	WITHOUT ANY WARRANTY; without even the implied warranty of				**
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the			**
**  GNU General Public License for more details.							**
**																			**
**  You should have received a copy of the GNU General Public License		**
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.	**
**																			**
*****************************************************************************/

#ifdef _WIN32
    #include <opencv2/core.hpp>
    #include <opencv2/highgui.hpp>
	#include <io.h>
#elif __linux
    #include <opencv2/core/core.hpp>
    #include <opencv2/highgui/highgui.hpp>
	#include <unistd.h>
#endif

#include <fstream>
#include <string.h>
#include "pdflow_cudalib.h"
#include "legend_pdflow.xpm"

#ifdef _WIN32
    #define M_PI 3.14159265f
    #define M_LOG2E 1.44269504088896340736f //log2(e)
    inline float log2(const float x){ return  log(x) * M_LOG2E; }

#elif __linux
    inline int stoi(char *c) {return int(std::strtol(c,NULL,10));}

#endif

//==================================================================
//					PD-Flow class (using openCV)
//==================================================================

class PD_flow_opencv {
public:

    unsigned int cam_mode;	// (1 - 640 x 480, 2 - 320 x 240)
    unsigned int ctf_levels;//Number of levels used in the coarse-to-fine scheme (always dividing by two)
    unsigned int num_max_iter[6];  //Number of iterations at every pyramid level (primal-dual solver)
    float g_mask[25];
	
    //Matrices that store the original images
	cv::Mat intensity1;
	cv::Mat depth1;
	cv::Mat intensity2;
	cv::Mat depth2;

	//Aux pointers to copy the RGBD images to CUDA 
	float *I, *Z;

    //Motion field
	float *dxp, *dyp, *dzp;

    //Camera properties
    float fovh;     //In radians
    float fovv;     //In radians

    //Max resolution of the coarse-to-fine scheme.
    unsigned int rows;
    unsigned int cols;

	//Resolution of the original images
	unsigned int width;
	unsigned int height;

    //Optimization Parameters
    float mu, lambda_i, lambda_d;

    //Cuda
    CSF_cuda csf_host, *csf_device;

    // Filenames
    const char *intensity_filename_1;
    const char *intensity_filename_2;
    const char *depth_filename_1;
    const char *depth_filename_2;
    const char *output_filename_root;

	//Methods
	bool loadRGBDFrames();
    void createImagePyramidGPU();
    void solveSceneFlowGPU();
    void freeGPUMemory();
    void initializeCUDA();
	void showImages();
    cv::Mat createImage() const;
    void saveResults( const cv::Mat& image) const;
	void showAndSaveResults();

    PD_flow_opencv( unsigned int rows_config, 
                    const char *intensity_filename_1="i1.png", 
                    const char *intensity_filename_2="i2.png", 
                    const char *depth_filename_1="z1.png", 
                    const char *depth_filename_2="z2.png", 
                    const char* output_filename_root="pdflow");
};



