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

#if BUILD_PLATFORM == WINDOWS_BUILD
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#elif BUILD_PLATFORM == LINUX_BUILD
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#endif

#include "pdflow_cudalib.h"
#include "legend_pdflow.xpm"
#include <ostream>


#if BUILD_PLATFORM == WINDOWS_BUILD

#define M_PI 3.14159265f
#define M_LOG2E 1.44269504088896340736f //log2(e)
inline float log2(const float x){ return  log(x) * M_LOG2E; }

#endif


class PD_flow_opencv {
public:

    float len_disp;         //In meters
    unsigned int cam_mode;	// (1 - 640 x 480, 2 - 320 x 240, 4 - 160 x 120)
    unsigned int ctf_levels;//Number of levels used in the coarse-to-fine scheme (always dividing by two)
    unsigned int num_max_iter[6];  //Max number of iterations distributed homogeneously between all levels
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
    float f_dist;	//In meters
    float fovh;     //Here it is expressed in radians
    float fovv;     //Here it is expressed in radians

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

	//Methods
	bool loadRGBDFrames();
    void createImagePyramidGPU();
    void solveSceneFlowGPU();
    void freeGPUMemory();
    void initializeCUDA();
	void showImages();
	void showAndSaveResults();

    PD_flow_opencv(unsigned int rows_config);
};



