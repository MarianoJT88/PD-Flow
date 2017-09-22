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

#include "scene_flow_impair.h"

bool  fileExists(const std::string& path)
{
    return 0 == access(path.c_str(), 0x00 ); // 0x00 = Check for existence only!
}

PD_flow_opencv::PD_flow_opencv(unsigned int rows_config, 
	const char *intensity_filename_1, 
	const char *intensity_filename_2,
	const char *depth_filename_1,
	const char *depth_filename_2,
	const char* output_filename_root) {

    rows = rows_config;      //Maximum size of the coarse-to-fine scheme - Default 240 (QVGA)
    cols = rows*320/240;
    ctf_levels = static_cast<unsigned int>(log2(float(rows/15))) + 1;
    fovh = M_PI*62.5f/180.f;
    fovv = M_PI*48.5f/180.f;

	//Iterations of the primal-dual solver at each pyramid level.
	//Maximum value set to 100 at the finest level
	for (int i=5; i>=0; i--)
	{
		if (i >= ctf_levels - 1)
			num_max_iter[i] = 100;	
		else
			num_max_iter[i] = num_max_iter[i+1]-15;
	}

    //Compute gaussian mask
	int v_mask[5] = {1,4,6,4,1};
    for (unsigned int i=0; i<5; i++)
        for (unsigned int j=0; j<5; j++)
            g_mask[i+5*j] = float(v_mask[i]*v_mask[j])/256.f;


    //Reserve memory for the scene flow estimate (the finest)
	dxp = (float *) malloc(sizeof(float)*rows*cols);
	dyp = (float *) malloc(sizeof(float)*rows*cols);
	dzp = (float *) malloc(sizeof(float)*rows*cols);

    //Parameters of the variational method
    lambda_i = 0.04f;
    lambda_d = 0.35f;
    mu = 75.f;

    // Set file names
    this->intensity_filename_1 = intensity_filename_1;
    this->intensity_filename_2 = intensity_filename_2;
    this->depth_filename_1 = depth_filename_1;
    this->depth_filename_2 = depth_filename_2;
    this->output_filename_root = output_filename_root;
}


void PD_flow_opencv::createImagePyramidGPU()
{
    //Copy new frames to the scene flow object
    csf_host.copyNewFrames(I, Z);

    //Copy scene flow object to device
    csf_device = ObjectToDevice(&csf_host);

    unsigned int pyr_levels = static_cast<unsigned int>(log2(float(width/cols))) + ctf_levels;
    GaussianPyramidBridge(csf_device, pyr_levels, cam_mode);

    //Copy scene flow object back to host
    BridgeBack(&csf_host, csf_device);
}

void PD_flow_opencv::solveSceneFlowGPU()
{
    unsigned int s;
    unsigned int cols_i, rows_i;
    unsigned int level_image;
    unsigned int num_iter;

    //For every level (coarse-to-fine)
    for (unsigned int i=0; i<ctf_levels; i++)
    {
        s = static_cast<unsigned int>(pow(2.f,int(ctf_levels-(i+1))));
        cols_i = cols/s;
        rows_i = rows/s;
        level_image = ctf_levels - i + static_cast<unsigned int>(log2(float(width/cols))) - 1;

        //=========================================================================
        //                              Cuda - Begin
        //=========================================================================

        //Cuda allocate memory
        csf_host.allocateMemoryNewLevel(rows_i, cols_i, i, level_image);

        //Cuda copy object to device
        csf_device = ObjectToDevice(&csf_host);

        //Assign zeros to the corresponding variables
        AssignZerosBridge(csf_device);

        //Upsample previous solution
        if (i>0)
            UpsampleBridge(csf_device);

        //Compute connectivity (Rij)
		RijBridge(csf_device);
		
		//Compute colour and depth derivatives
        ImageGradientsBridge(csf_device);
        WarpingBridge(csf_device);

        //Compute mu_uv and step sizes for the primal-dual algorithm
        MuAndStepSizesBridge(csf_device);

        //Primal-Dual solver
		for (num_iter = 0; num_iter < num_max_iter[i]; num_iter++)
        {
            GradientBridge(csf_device);
            DualVariablesBridge(csf_device);
            DivergenceBridge(csf_device);
            PrimalVariablesBridge(csf_device);
        }

        //Filter solution
        FilterBridge(csf_device);

        //Compute the motion field
        MotionFieldBridge(csf_device);

        //BridgeBack to host
        BridgeBack(&csf_host, csf_device);

        //Free memory of variables associated to this level
        csf_host.freeLevelVariables();

        //Copy motion field to CPU
		csf_host.copyMotionField(dxp, dyp, dzp);

		//For debugging
        //DebugBridge(csf_device);

        //=========================================================================
        //                              Cuda - end
        //=========================================================================
    }
}

void PD_flow_opencv::freeGPUMemory()
{
    csf_host.freeDeviceMemory();
}

void PD_flow_opencv::initializeCUDA()
{
	//Read one image to know the image resolution
	intensity1 = cv::imread(intensity_filename_1, CV_LOAD_IMAGE_GRAYSCALE);

	width = intensity1.cols;
	height = intensity1.rows;
	if (height == 240) {cam_mode = 2;}
	else			   {cam_mode = 1;}

	I = (float *) malloc(sizeof(float)*width*height);
	Z = (float *) malloc(sizeof(float)*width*height);   
	
	//Read parameters
    csf_host.readParameters(rows, cols, lambda_i, lambda_d, mu, g_mask, ctf_levels, cam_mode, fovh, fovv);

    //Allocate memory
    csf_host.allocateDevMemory();
}

void PD_flow_opencv::showImages()
{
	const unsigned int dispx = intensity1.cols + 20;
	const unsigned int dispy = intensity1.rows + 20;

	//Show images with OpenCV windows
	cv::namedWindow("I1", cv::WINDOW_AUTOSIZE);
	cv::moveWindow("I1",10,10);
	cv::imshow("I1", intensity1);

	cv::namedWindow("Z1", cv::WINDOW_AUTOSIZE);
	cv::moveWindow("Z1",dispx,10);
	cv::imshow("Z1", depth1);

	cv::namedWindow("I2", cv::WINDOW_AUTOSIZE);
	cv::moveWindow("I2",10,dispy);
	cv::imshow("I2", intensity2);

	cv::namedWindow("Z2", cv::WINDOW_AUTOSIZE);
	cv::moveWindow("Z2",dispx,dispy);
	cv::imshow("Z2", depth2);
}

bool PD_flow_opencv::loadRGBDFrames()
{
	cv::Mat depth_float;

	//First intensity image
	intensity1 = cv::imread(intensity_filename_1, CV_LOAD_IMAGE_GRAYSCALE);
	if (intensity1.empty())
	{
		printf("\nThe first intensity image (%s) cannot be found, please check that it is in the correct folder \n", intensity_filename_1);
		return 0;
	}

	for (unsigned int u=0; u<width; u++)
		for (unsigned int v=0; v<height; v++)
			I[v + u*height] = float(intensity1.at<unsigned char>(v,u));

	//First depth image
	depth1 = cv::imread(depth_filename_1, -1);
	if (depth1.empty())
	{
		printf("\nThe first depth image (%s) cannot be found, please check that it is in the correct folder \n", depth_filename_1);
		return 0;
	}

	depth1.convertTo(depth_float, CV_32FC1, 1.0 / 5000.0);
	for (unsigned int v=0; v<height; v++)
		for (unsigned int u=0; u<width; u++)
			Z[v + u*height] = depth_float.at<float>(v,u);

	createImagePyramidGPU();


	//Second intensity image
	intensity2 = cv::imread(intensity_filename_2, CV_LOAD_IMAGE_GRAYSCALE);
	if (intensity2.empty())
	{
		printf("\nThe second intensity image (%s) cannot be found, please check that it is in the correct folder \n", intensity_filename_2);
		return 0;
	}

	for (unsigned int v=0; v<height; v++)
		for (unsigned int u=0; u<width; u++)
			I[v + u*height] = float(intensity2.at<unsigned char>(v,u));

	//Second depth image
	depth2 = cv::imread(depth_filename_2, -1);
	if (depth2.empty())
	{
		printf("\nThe second depth image (%s) cannot be found, please check that they are in the correct folder \n", depth_filename_2);
		return 0;
	}
	depth2.convertTo(depth_float, CV_32FC1, 1.0 / 5000.0);
	for (unsigned int v=0; v<height; v++)
		for (unsigned int u=0; u<width; u++)
			Z[v + u*height] = depth_float.at<float>(v,u);

	createImagePyramidGPU();

	return 1;
}

// Create the image
cv::Mat PD_flow_opencv::createImage() const
{
	//Save scene flow as an RGB image (one colour per direction)
	cv::Mat sf_image(rows, cols, CV_8UC3);

    //Compute the max values of the flow (of its components)
	float maxmodx = 0.f, maxmody = 0.f, maxmodz = 0.f;
	for (unsigned int v=0; v<rows; v++)
		for (unsigned int u=0; u<cols; u++)
		{
            if (fabs(dxp[v + u*rows]) > maxmodx)
                maxmodx = fabs(dxp[v + u*rows]);
            if (fabs(dyp[v + u*rows]) > maxmody)
                maxmody = fabs(dyp[v + u*rows]);
            if (fabs(dzp[v + u*rows]) > maxmodz)
                maxmodz = fabs(dzp[v + u*rows]);
		}

	//Create an RGB representation of the scene flow estimate: 
	for (unsigned int v=0; v<rows; v++)
		for (unsigned int u=0; u<cols; u++)
		{
            sf_image.at<cv::Vec3b>(v,u)[0] = static_cast<unsigned char>(255.f*fabs(dxp[v + u*rows])/maxmodx); //Blue - x
            sf_image.at<cv::Vec3b>(v,u)[1] = static_cast<unsigned char>(255.f*fabs(dyp[v + u*rows])/maxmody); //Green - y
            sf_image.at<cv::Vec3b>(v,u)[2] = static_cast<unsigned char>(255.f*fabs(dzp[v + u*rows])/maxmodz); //Red - z
		}
	

	return sf_image;
}


/**
 * Save results without displaying them
 */
void PD_flow_opencv::saveResults( const cv::Mat& sf_image ) const
{
	//Save the scene flow as a text file 
	char	name[500];
	int     nFichero = 0;
	bool    free_name = false;

	while (!free_name)
	{
		nFichero++;
		sprintf(name, "%s_results%02u.txt", output_filename_root, nFichero );
		free_name = !fileExists(name);
	}
	
	std::ofstream f_res;
	f_res.open(name);
	printf("Saving the estimated scene flow to file: %s \n", name);

	//Format: (pixel(row), pixel(col), vx, vy, vz)
	for (unsigned int v=0; v<rows; v++)
		for (unsigned int u=0; u<cols; u++)
		{
			f_res << v << " ";
			f_res << u << " ";
			f_res << dxp[v + u*rows] << " ";
			f_res << dyp[v + u*rows] << " ";
			f_res << dzp[v + u*rows] << std::endl;
		}

	f_res.close();

	//Save the RGB representation of the scene flow
	sprintf(name, "%s_representation%02u.png", output_filename_root, nFichero);
	printf("Saving the visual representation to file: %s \n", name);
	cv::imwrite(name, sf_image);
}


void PD_flow_opencv::showAndSaveResults( )
{
	cv::Mat sf_image = createImage( );

	//Show the scene flow as an RGB image	
	cv::namedWindow("SceneFlow", cv::WINDOW_NORMAL);
    cv::moveWindow("SceneFlow",width - cols/2,height - rows/2);
	cv::imshow("SceneFlow", sf_image);

	saveResults( sf_image );
}
