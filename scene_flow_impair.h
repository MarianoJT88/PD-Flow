
#include <mrpt/utils.h>
#include <mrpt/system.h>
#include <mrpt/gui/CDisplayWindow3D.h>
#include <mrpt/opengl.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/src/Core/Matrix.h>
#include "pdflow_cudalib.h"
#include "legend_pdflow.xpm"
#include <stdlib.h>
#include <ostream>

#define M_LOG2E 1.44269504088896340736 //log2(e)

inline float log2(const float x){
    return  log(x) * M_LOG2E;
}


using namespace mrpt;
using namespace mrpt::math;
using namespace mrpt::utils;
using namespace std;
using mrpt::poses::CPose3D;
using Eigen::MatrixXf;


class PD_flow_opencv {
public:

    float len_disp;         //In meters
    unsigned int cam_mode;	// (1 - 640 x 480, 2 - 320 x 240, 4 - 160 x 120)
    unsigned int ctf_levels;//Number of levels used in the coarse-to-fine scheme (always dividing by two)
    unsigned int num_max_iter[6];  //Max number of iterations distributed homogeneously between all levels
    float g_mask[25];
	
    //Matrices that store the original images with the image resolution
	cv::Mat intensity1;
	cv::Mat depth1;
	cv::Mat intensity2;
	cv::Mat depth2;
	float *I, *Z;
    MatrixXf colour_wf;
    MatrixXf depth_wf;

    //Matrices that store the images downsampled
    vector<MatrixXf> colour;
    vector<MatrixXf> colour_old;
    vector<MatrixXf> depth;
    vector<MatrixXf> depth_old;
    vector<MatrixXf> xx;
    vector<MatrixXf> xx_old;
    vector<MatrixXf> yy;
    vector<MatrixXf> yy_old;

    //Motion field
    vector<MatrixXf> dx;
    vector<MatrixXf> dy;
    vector<MatrixXf> dz;
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

    //Visual
    gui::CDisplayWindow3D       window;
    opengl::COpenGLScenePtr		scene;
    utils::CImage				image;

    //Cuda
    CSF_cuda csf_host, *csf_device;

	//Methods
	bool loadRGBDFrames();
    void createImagePyramidGPU();
    void solveSceneFlowGPU();
    void freeGPUMemory();
    void initializeCUDA();
	void initializeScene();
	void updateScene();
	void showImages();
	void showAndSaveResults();

    PD_flow_opencv(unsigned int rows_config);
};



