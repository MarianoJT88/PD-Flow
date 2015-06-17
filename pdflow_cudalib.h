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

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define N_blocks 256
#define N_threads 128

//Warning!!!!!! Number of threads should be higher than 25 - See definition of "computePyramidLevel()"

#define ELEM_SWAP(a,b) { register float t=(a);(a)=(b);(b)=t; }

//--------------------------------------------------------------
//				Host-Device class for the PD-Flow
//--------------------------------------------------------------
class CSF_cuda {
public:

    //Parameters
    unsigned int rows;
    unsigned int cols;
    unsigned int ctf_levels;
    unsigned int cam_mode;
    float lambda_i;
    float lambda_d;
    float mu;
    float fovh;
    float fovv;

    //Local values
    unsigned int local_level;
    unsigned int level_image;
    unsigned int rows_i, cols_i;

    //Compute gaussian mask
    float *g_mask_dev;

    //Last frames read
    float *colour_wf_dev, *depth_wf_dev;

    //Images and coordinates (pointers to pointers...)
    float *depth_old_dev[8], *xx_old_dev[8], *yy_old_dev[8];
    float *depth_dev[8], *xx_dev[8], *yy_dev[8];
    float *colour_old_dev[8], *colour_dev[8];

    //Motion field
    float *dx_dev, *dy_dev, *dz_dev;

    //Intensity and depth gradients without warping
    float *dcu_aux_dev, *dcv_aux_dev;
    float *ddu_aux_dev, *ddv_aux_dev;

    //Warped intensity and depth gradients
    float *dct_dev, *dcu_dev, *dcv_dev;
    float *ddt_dev, *ddu_dev, *ddv_dev;

    //Motion field gradients
    float *gradu1_dev, *gradu2_dev;
    float *gradv1_dev, *gradv2_dev;
    float *gradw1_dev, *gradw2_dev;

    //Divergence of dual variables
    float *divpu_dev, *divpv_dev, *divpw_dev;

    //Primal and dual step sizes
    float *sigma_pd_dev, *sigma_puvx_dev, *sigma_puvy_dev;
    float *sigma_pwx_dev, *sigma_pwy_dev;
    float *tau_u_dev, *tau_v_dev, *tau_w_dev;

    //Weights for data terms and regularization
    float *mu_uv_dev;
    float *ri_dev, *rj_dev;
	float *ri_2_dev, *rj_2_dev;

    //Primal-dual acceleration and previous solution
    float *du_acc_dev, *dv_acc_dev, *dw_acc_dev;
    float *du_prev_dev, *dv_prev_dev;

    //Aux variables for gaussian upsampling
    float *du_upsamp_dev, *dv_upsamp_dev, *dw_upsamp_dev;
    float *pd_upsamp_dev;
    float *puu_upsamp_dev, *puv_upsamp_dev;
    float *pvu_upsamp_dev, *pvv_upsamp_dev;
    float *pwu_upsamp_dev, *pwv_upsamp_dev;

    //Solution of the previous level (only store one)
    float *du_l_dev, *dv_l_dev, *dw_l_dev;
    float *pd_l_dev;
    float *puu_l_dev, *puv_l_dev;
    float *pvu_l_dev, *pvv_l_dev;
    float *pwu_l_dev, *pwv_l_dev;

    //Primal and dual variables
    float *du_new_dev;
    float *dv_new_dev;
    float *dw_new_dev;
    float *pd_dev;
    float *puu_dev, *puv_dev;
    float *pvu_dev, *pvv_dev;
    float *pwu_dev, *pwv_dev;


    __host__ void allocateDevMemory();
    __host__ void allocateMemoryNewLevel(unsigned int rows_loc, unsigned int cols_loc, unsigned int level_i, unsigned int level_image_i);
    __host__ void readParameters(unsigned int rows_host, unsigned int cols_host, float lambda_i_host, float lambda_d_host, float mu_host,
								 float *g_mask, unsigned int levels_host, unsigned int cam_mode_host, float fovh_host, float fovv_host);

    __host__ void copyNewFrames(float *colour_wf, float *depth_wf);
    __host__ void freeDeviceMemory();
    __host__ void freeLevelVariables();
	__host__ void copyAllSolutions(float *dx, float *dy, float *dz, float *depth, float *depth_old, float *colour, float *colour_old, float *xx, float *xx_old, float *yy, float *yy_old);
	__host__ void copyMotionField(float *dx, float *dy, float *dz);

    __device__ void computePyramidLevel(unsigned int index, unsigned int level);
    __device__ void assignZeros(unsigned int index);
    __device__ void upsampleCopyPrevSolution(unsigned int index);
    __device__ void upsampleFilterPrevSolution(unsigned int index);
    __device__ void computeImGradients(unsigned int index);
    __device__ void performWarping(unsigned int index);
    __device__ void computeRij(unsigned int index);
    __device__ void computeMu(unsigned int index);
    __device__ void computeStepSizes(unsigned int index);
    __device__ void updateDualVariables(unsigned int index);
    __device__ void updatePrimalVariables(unsigned int index);
    __device__ void computeDivergence(unsigned int index);
    __device__ void computeGradient(unsigned int index);
    __device__ void saturateVariables(unsigned int index);
    __device__ void filterSolution(unsigned int index);
    __device__ float interpolatePixel(float *mat, float ind_u, float ind_v);
	__device__ float interpolatePixelDepth(float *mat, float ind_u, float ind_v);
    __device__ void computeMotionField(unsigned int index);
};

//-------------------------------------------------------------------------------------------------------
// Bridges between Cuda-related functions and the code compiled with the standard compiler (without CUDA)
//-------------------------------------------------------------------------------------------------------
CSF_cuda *ObjectToDevice(CSF_cuda *csf_host);
void GaussianPyramidBridge(CSF_cuda *csf, unsigned int levels, unsigned int cam_mode);
void AssignZerosBridge(CSF_cuda *csf);
void UpsampleBridge(CSF_cuda *csf);
void ImageGradientsBridge(CSF_cuda *csf);
void RijBridge(CSF_cuda *csf);
void WarpingBridge(CSF_cuda *csf);
void MuAndStepSizesBridge(CSF_cuda *csf);
void DualVariablesBridge(CSF_cuda *csf);
void PrimalVariablesBridge(CSF_cuda *csf);
void DivergenceBridge(CSF_cuda *csf);
void GradientBridge(CSF_cuda *csf);
void FilterBridge(CSF_cuda *csf);
void MotionFieldBridge(CSF_cuda *csf);
void DebugBridge(CSF_cuda *csf_device);
void BridgeBack(CSF_cuda *csf_host, CSF_cuda *csf_device);

//-------------------------------------------------------------------------------------------------------
//				Kernels corresponding to the main steps of the algorithm (and aux kernels)
//-------------------------------------------------------------------------------------------------------
__global__ void ComputePyramidLevelKernel (CSF_cuda *csf, unsigned int level);
__global__ void AssignZerosKernel (CSF_cuda *csf);
__global__ void UpsampleCopyKernel (CSF_cuda *csf);
__global__ void UpsampleFilterKernel (CSF_cuda *csf);
__global__ void RijKernel(CSF_cuda *csf);
__global__ void ComputeImGradients (CSF_cuda *csf);
__global__ void PerformWarping(CSF_cuda *csf);
__global__ void MuAndStepSizesKernel(CSF_cuda *csf);
__global__ void DualIteration(CSF_cuda *csf);
__global__ void PrimalIteration(CSF_cuda *csf);
__global__ void DivergenceComputation(CSF_cuda *csf);
__global__ void GradientComputation(CSF_cuda *csf);
__global__ void SaturateSolution(CSF_cuda *csf);
__global__ void FilterSolution(CSF_cuda *csf);
__global__ void MotionFieldKernel (CSF_cuda *csf);
__global__ void DebugKernel(CSF_cuda *csf);

//-----------------------------------------------------------------------------
//				Aux functions for the weighted median filter
//-----------------------------------------------------------------------------

//Sorting - Structure field and presence
struct fieldAndPresence {
	float field;
	float pres;
};

//Naive implementation of bubbleSort (applied to very small arrays)
__device__ void bubbleSortDev(fieldAndPresence array[], unsigned int num_elem);



