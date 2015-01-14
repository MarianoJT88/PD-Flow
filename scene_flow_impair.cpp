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

PD_flow_opencv::PD_flow_opencv(unsigned int rows_config)
{     
    rows = rows_config;      //Maximum size of the coarse-to-fine scheme
    cols = rows*320/240;
    ctf_levels = unsigned int(log2(rows/15)) + 1;
    fovh = M_PI*62.5f/180.f;
    fovv = M_PI*45.f/180.f;
    len_disp = 0.022f;
    num_max_iter[0] = 40;
    num_max_iter[1] = 50;
    num_max_iter[2] = 60;
    num_max_iter[3] = 80;
    num_max_iter[4] = 100;
    num_max_iter[5] = 0;

    //Compute gaussian mask
	int v_mask[5] = {1,4,6,4,1};
    for (unsigned int i=0; i<5; i++)
        for (unsigned int j=0; j<5; j++)
            g_mask[i+5*j] = float(v_mask[i]*v_mask[j])/256.f;


    //Resize vectors according to levels
	dxp = (float *) malloc(sizeof(float)*rows*cols);
	dyp = (float *) malloc(sizeof(float)*rows*cols);
	dzp = (float *) malloc(sizeof(float)*rows*cols);


    //Parameters of the variational method
    lambda_i = 0.04f;
    lambda_d = 0.35f;
    mu = 75.f;

    //Camera parameters
    f_dist = 1.f/525.f;     //In meters
}

void PD_flow_opencv::createImagePyramidGPU()
{
    //Cuda copy new frames
    csf_host.copyNewFrames(I, Z);

    //Cuda copy object to device
    csf_device = ObjectToDevice(&csf_host);

    unsigned int pyr_levels = unsigned int(log2(width/cols)) + ctf_levels;
    GaussianPyramidBridge(csf_device, pyr_levels, cam_mode);

    //Cuda copy object back to host
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
        s = pow(2.f,int(ctf_levels-(i+1)));
        cols_i = cols/s;
        rows_i = rows/s;
        level_image = ctf_levels - i + unsigned int(log2(width/cols)) - 1;

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

        //BridgeBack
        BridgeBack(&csf_host, csf_device);

        //Free variables of this level
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
 	char name[100];
	
	//Read one image to know the image resolution
	sprintf(name, "i1.png");
	intensity1 = cv::imread(name, CV_LOAD_IMAGE_GRAYSCALE);

	width = intensity1.cols;
	height = intensity1.rows;
	if (height == 240) {cam_mode = 2;}	//*************************************
	else			   {cam_mode = 1;}

	I = (float *) malloc(sizeof(float)*width*height);
	Z = (float *) malloc(sizeof(float)*width*height);   
	
	//Read parameters
    csf_host.readParameters(rows, cols, lambda_i, lambda_d, mu, g_mask, ctf_levels, len_disp, cam_mode, fovh, fovv, f_dist);

    //Allocate memory
    csf_host.allocateDevMemory();
}

void PD_flow_opencv::showImages()
{
	//Show images
	const unsigned int dispx = intensity1.cols + 20;
	const unsigned int dispy = intensity1.rows + 20;

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

	cv::waitKey(30);
}

bool PD_flow_opencv::loadRGBDFrames()
{
	char name[100];
	cv::Mat depth_float;

	//First intensity image
	sprintf(name, "i1.png");
	intensity1 = cv::imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	if (intensity1.empty())
	{
		printf("\nThe first intensity image (i1) cannot be found, please check that it is in the correct folder \n");
		return 0;
	}

	for (unsigned int u=0; u<width; u++)
		for (unsigned int v=0; v<height; v++)
			I[v + u*height] = float(intensity1.at<unsigned char>(v,u));

	//First depth image
	sprintf(name, "z1.png");
	depth1 = cv::imread(name, -1);
	if (depth1.empty())
	{
		printf("\nThe first depth image (z1) cannot be found, please check that it is in the correct folder \n");
		return 0;
	}

	depth1.convertTo(depth_float, CV_32FC1, 1.0 / 5000.0);
	for (unsigned int v=0; v<height; v++)
		for (unsigned int u=0; u<width; u++)
			Z[v + u*height] = depth_float.at<float>(v,u);

	createImagePyramidGPU();


	//Second intensity image
	sprintf(name, "i2.png");
	intensity2 = cv::imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	if (intensity2.empty())
	{
		printf("\nThe second intensity image (i2) cannot be found, please check that it is in the correct folder \n");
		return 0;
	}

	for (unsigned int v=0; v<height; v++)
		for (unsigned int u=0; u<width; u++)
			I[v + u*height] = float(intensity2.at<unsigned char>(v,u));

	//Second depth image
	sprintf(name, "z2.png");
	depth2 = cv::imread(name, -1);
	if (depth2.empty())
	{
		printf("\nThe second depth image (z2) cannot be found, please check that they are in the correct folder \n");
		return 0;
	}
	depth2.convertTo(depth_float, CV_32FC1, 1.0 / 5000.0);
	for (unsigned int v=0; v<height; v++)
		for (unsigned int u=0; u<width; u++)
			Z[v + u*height] = depth_float.at<float>(v,u);

	createImagePyramidGPU();

	return 1;
}

void PD_flow_opencv::showAndSaveResults()
{
	//Save scene flow as an RGB image (one colour per direction)
	cv::Mat sf_image(rows, cols, CV_8UC3);

	//Compute the max-min values of the flow
	float maxmodx = 0.f, maxmody = 0.f, maxmodz = 0.f;
	for (unsigned int v=0; v<rows; v++)
		for (unsigned int u=0; u<cols; u++)
		{
			if (abs(dxp[v + u*rows]) > maxmodx)
				maxmodx = abs(dxp[v + u*rows]);
			if (abs(dyp[v + u*rows]) > maxmody)
				maxmody = abs(dyp[v + u*rows]);
			if (abs(dzp[v + u*rows]) > maxmodz)
				maxmodz = abs(dzp[v + u*rows]);
		}

	for (unsigned int v=0; v<rows; v++)
		for (unsigned int u=0; u<cols; u++)
		{
			sf_image.at<cv::Vec3b>(v,u)[0] = 255*abs(dxp[v + u*rows])/maxmodx; //Blue
			sf_image.at<cv::Vec3b>(v,u)[1] = 255*abs(dyp[v + u*rows])/maxmody; //Green
			sf_image.at<cv::Vec3b>(v,u)[2] = 255*abs(dzp[v + u*rows])/maxmodz; //Red
		}
	
	//Show the scene flow as an RGB image	
	cv::namedWindow("SceneFlow", cv::WINDOW_NORMAL);
	cv::moveWindow("SceneFlow",10,10);
	cv::imshow("SceneFlow", sf_image);
	cv::waitKey(100000);


	//Save the scene flow as a text file, as well as the RGB generated image. 
	//char	name[100];
	//int     nFichero = 0;
	//bool    free_name = false;

	//while (!free_name)
	//{
	//	nFichero++;
	//	sprintf(name, "pdflow_results%02u.txt", nFichero );
	//	free_name = !system::fileExists(name);
	//}
	//
	//std::ofstream f_res;
	//f_res.open(name);
	//printf("Saving the estimated scene flow to file: %s \n", name);

	////Format: (pixel(row), pixel(col), vx, vy, vz)
	//for (unsigned int v=0; v<rows; v++)
	//	for (unsigned int u=0; u<cols; u++)
	//	{
	//		f_res << v << " ";
	//		f_res << u << " ";
	//		f_res << dx[0](v,u) << " ";
	//		f_res << dy[0](v,u) << " ";
	//		f_res << dz[0](v,u) << endl;
	//	}

	//f_res.close();

	//sprintf(name, "pdflow_representation%02u.png", nFichero);
	//printf("Saving the visual representation to file: %s \n", name);
	//cv::imwrite(name, sf_image);
}