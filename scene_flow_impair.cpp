
#include "scene_flow_impair.h"

PD_flow_opencv::PD_flow_opencv(unsigned int rows_config)
{     
    rows = rows_config;      //Maximum size of the coarse-to-fine scheme
    cols = rows*320/240;
	cam_mode = 1;
    ctf_levels = round(log2(rows/15)) + 1;
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

    //Matrices that store the original and filtered images with the image resolution
    colour_wf.setSize(480/cam_mode,640/cam_mode);
    depth_wf.setSize(480/cam_mode,640/cam_mode);

    //Resize vectors according to levels
    dx.resize(ctf_levels); dy.resize(ctf_levels); dz.resize(ctf_levels);
	dxp = (float *) malloc(sizeof(float)*rows*cols);
	dyp = (float *) malloc(sizeof(float)*rows*cols);
	dzp = (float *) malloc(sizeof(float)*rows*cols);

    const unsigned int width = colour_wf.getColCount();
    const unsigned int height = colour_wf.getRowCount();
    unsigned int s, cols_i, rows_i;

    for (unsigned int i = 0; i<ctf_levels; i++)
    {
        s = pow(2.f,int(ctf_levels-(i+1)));
        cols_i = cols/s; rows_i = rows/s;
        dx[ctf_levels-i-1].setSize(rows_i,cols_i);
        dy[ctf_levels-i-1].setSize(rows_i,cols_i);
        dz[ctf_levels-i-1].setSize(rows_i,cols_i);
    }

    //Resize pyramid
    const unsigned int pyr_levels = round(log2(width/cols)) + ctf_levels;
    colour.resize(pyr_levels);
    colour_old.resize(pyr_levels);
    depth.resize(pyr_levels);
    depth_old.resize(pyr_levels);
    xx.resize(pyr_levels);
    xx_old.resize(pyr_levels);
    yy.resize(pyr_levels);
    yy_old.resize(pyr_levels);

    for (unsigned int i = 0; i<pyr_levels; i++)
    {
        s = pow(2.f,int(i));
        colour[i].resize(height/s, width/s);
        colour_old[i].resize(height/s, width/s);
        colour[i].assign(0.0f);
        colour_old[i].assign(0.0f);
        depth[i].resize(height/s, width/s);
        depth_old[i].resize(height/s, width/s);
        depth[i].assign(0.0f);
        depth_old[i].assign(0.0f);
        xx[i].resize(height/s, width/s);
        xx_old[i].resize(height/s, width/s);
        xx[i].assign(0.0f);
        xx_old[i].assign(0.0f);
        yy[i].resize(height/s, width/s);
        yy_old[i].resize(height/s, width/s);
        yy[i].assign(0.0f);
        yy_old[i].assign(0.0f);
    }

    //Parameters of the variational method
    lambda_i = 0.04f;
    lambda_d = 0.35f;
    mu = 75.f;

    //Camera parameters
    f_dist = 1.f/525.f;     //In meters
}

void PD_flow_opencv::createImagePyramidGPU()
{
    utils::CTicTac clock;
    clock.Tic();

    //Cuda copy new frames
    csf_host.copyNewFrames(I, Z);
	//csf_host.copyNewFrames(colour_wf.data(), depth_wf.data());

    //Cuda copy object to device
    csf_device = ObjectToDevice(&csf_host);

    unsigned int pyr_levels = round(log2(width/cols)) + ctf_levels;
    GaussianPyramidBridge(csf_device, pyr_levels, cam_mode);

    //Cuda copy object back to host
    BridgeBack(&csf_host, csf_device);

	//Execution results
	const float aux_time = 1000.f*clock.Tac();
    //cout << endl << "Time for the pyramid: " << aux_time << "ms";
}

void PD_flow_opencv::solveSceneFlowGPU()
{
    //Define variables
    CTicTac	clock;

    unsigned int s;
    unsigned int cols_i, rows_i;
    unsigned int level_image;
    unsigned int num_iter;

    clock.Tic();

    //For every level (coarse-to-fine)
    for (unsigned int i=0; i<ctf_levels; i++)
    {
        s = pow(2.f,int(ctf_levels-(i+1)));
        cols_i = cols/s;
        rows_i = rows/s;
        level_image = ctf_levels - i + round(log2(width/cols)) - 1;

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
    //Read parameters
    csf_host.readParameters(rows, cols, lambda_i, lambda_d, mu, g_mask, ctf_levels, len_disp, cam_mode, fovh, fovv, f_dist);

    //Allocate memory
    csf_host.allocateDevMemory();
}

void PD_flow_opencv::initializeScene()
{
    global_settings::OCTREE_RENDER_MAX_POINTS_PER_NODE = 10000000;
    window.resize(1000,900);
    window.setPos(900,0);
    window.setCameraZoom(4);
    window.setCameraAzimuthDeg(190);
    window.setCameraElevationDeg(30);
	window.setCameraPointingToPoint(1,0,0);
	
	scene = window.get3DSceneAndLock();

    //Point cloud (final)
    opengl::CPointCloudPtr fpoints_gl = opengl::CPointCloud::Create();
    fpoints_gl->setColor(0, 1, 1);
    fpoints_gl->enablePointSmooth();
    fpoints_gl->setPointSize(3.0);
    fpoints_gl->setPose(CPose3D(0,0,0,0,0,0));
    scene->insert( fpoints_gl );

    //Scene Flow
    opengl::CVectorField3DPtr sf = opengl::CVectorField3D::Create();
    sf->setPointSize(3.0f);
    sf->setLineWidth(2.0f);
    sf->setPointColor(1,0,0);
    sf->setVectorFieldColor(0,0,1);
    sf->enableAntiAliasing();
    sf->setPose(CPose3D(0,0,0,0,0,0));
    scene->insert( sf );

    //Reference frame
    opengl::CSetOfObjectsPtr reference = opengl::stock_objects::CornerXYZ();
    reference->setPose(CPose3D(0,len_disp,0,0,0,0));
	reference->setScale(0.15f);
    scene->insert( reference );

    window.unlockAccess3DScene();
    window.repaint();
}

void PD_flow_opencv::updateScene()
{	
	scene = window.get3DSceneAndLock();

	const unsigned int repr_level = round(log2(colour_wf.getColCount()/cols));

	//Point cloud (final)
    opengl::CPointCloudPtr fpoints_gl = scene->getByClass<opengl::CPointCloud>(0);
    fpoints_gl->clear();
    for (unsigned int v=0; v<rows; v++)
        for (unsigned int u=0; u<cols; u++)
            if (depth[repr_level](v,u) > 0.1f)
                fpoints_gl->insertPoint(depth[repr_level](v,u), xx[repr_level](v,u), yy[repr_level](v,u));


    opengl::CVectorField3DPtr sf = scene->getByClass<opengl::CVectorField3D>(0);
	sf->setPointCoordinates(depth_old[repr_level], xx_old[repr_level], yy_old[repr_level]);
    sf->setVectorField(dx[0], dy[0], dz[0]);

    window.unlockAccess3DScene();
    window.repaint();
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

	initializeScene();
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
		cout << endl << "The first intensity image (i1) cannot be found, please check that it is in the correct folder \n";
		return 0;
	}

	width = intensity1.cols;
	height = intensity1.rows;
	I = (float *) malloc(sizeof(float)*width*height);
	Z = (float *) malloc(sizeof(float)*width*height);

	for (unsigned int u=0; u<width; u++)
		for (unsigned int v=0; v<height; v++)
		{
			colour_wf(height-1-v,u) = intensity1.at<unsigned char>(v,u);
			I[v + u*height] = float(intensity1.at<unsigned char>(v,u));
		}

	//First depth image
	sprintf(name, "z1.png");
	depth1 = cv::imread(name, -1);
	if (depth1.empty())
	{
		cout << endl << "The first depth image (z1) cannot be found, please check that it is in the correct folder \n";
		return 0;
	}

	depth1.convertTo(depth_float, CV_32FC1, 1.0 / 5000.0);
	for (unsigned int v=0; v<depth_wf.rows(); v++)
		for (unsigned int u=0; u<depth_wf.cols(); u++)
		{
			depth_wf(height-1-v,u) = depth_float.at<float>(v,u);
			Z[v + u*height] = depth_float.at<float>(v,u);
		}

	createImagePyramidGPU();


	//Second intensity image
	sprintf(name, "i2.png");
	intensity2 = cv::imread(name, CV_LOAD_IMAGE_GRAYSCALE);
	if (intensity2.empty())
	{
		cout << endl << "The second intensity image (i2) cannot be found, please check that it is in the correct folder \n";
		return 0;
	}

	for (unsigned int v=0; v<colour_wf.rows(); v++)
		for (unsigned int u=0; u<colour_wf.cols(); u++)
		{
			colour_wf(height-1-v,u) = intensity2.at<unsigned char>(v,u);
			I[v + u*height] = float(intensity2.at<unsigned char>(v,u));
		}

	//Second depth image
	sprintf(name, "z2.png");
	depth2 = cv::imread(name, -1);
	if (depth2.empty())
	{
		cout << endl << "The second depth image (z2) cannot be found, please check that they are in the correct folder \n";
		return 0;
	}
	depth2.convertTo(depth_float, CV_32FC1, 1.0 / 5000.0);
	for (unsigned int v=0; v<depth_wf.rows(); v++)
		for (unsigned int u=0; u<depth_wf.cols(); u++)
		{
			depth_wf(height-1-v,u) = depth_float.at<float>(v,u);
			Z[v + u*height] = depth_float.at<float>(v,u);
		}

	createImagePyramidGPU();

	return 1;
}

void PD_flow_opencv::showAndSaveResults()
{
	//Save scene flow as an RGB image (one colour per direction)
	cv::Mat sf_image(rows, cols, CV_8UC3);

	//The max-min values are used by default but it can be set manually if the user wants to
	const float mindx = dx[0].minimum();
	const float maxdx = dx[0].maximum();
	const float mindy = dy[0].minimum();
	const float maxdy = dy[0].maximum();
	const float mindz = dz[0].minimum();
	const float maxdz = dz[0].maximum();

	//const float maxmodx = 0.8*max(abs(mindx), abs(maxdx));
	//const float maxmody = 0.8*max(abs(mindy), abs(maxdy));
	//const float maxmodz = 0.8*max(abs(mindz), abs(maxdz));

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
			//sf_image.at<cv::Vec3b>(v,u)[0] = 255*abs(dx[0](v,u))/maxmodx; //Blue
			//sf_image.at<cv::Vec3b>(v,u)[1] = 255*abs(dy[0](v,u))/maxmody; //Green
			//sf_image.at<cv::Vec3b>(v,u)[2] = 255*abs(dz[0](v,u))/maxmodz; //Red

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