// Author: Mariano Jaimez Tarifa
// Organization: MAPIR, University of Malaga
// Date: January 2014
// License: GNU

#include <stdio.h>
#include <string.h>
#include "scene_flow_impair.h"



// ------------------------------------------------------
//						MAIN
// ------------------------------------------------------

int main(int num_arg, char *argv[])
{	
	//==============================================================================
	//						Read function arguments
	//==============================================================================
	unsigned int rows = 240;	//Default values

	if (num_arg <= 1); //No arguments
	else if ( string(argv[1]) == "--help")
	{
		printf("\n\t       Arguments of the function 'main' \n");
		printf("==============================================================\n\n");
		printf(" --help: Shows this menu... \n\n");
		printf(" --rows r: Number of rows at the finest level of the pyramid. \n");
		printf("\t   Options: r=15, r=30, r=60, r=120, r=240, r=480 (if VGA)\n");
		system::os::getch();
		return 1;
	}
	else
	{
		if ( string(argv[1]) == "--rows")
			rows = stof(string(argv[2]));
	}


	PD_flow_opencv sceneflow(rows);

	//Initialize CUDA
    sceneflow.initializeCUDA();

	bool imloaded = sceneflow.loadRGBDFrames();

	if (imloaded == 1)
	{
		sceneflow.showImages();	
		sceneflow.solveSceneFlowGPU();
		//sceneflow.updateScene();
		sceneflow.showAndSaveResults();
		sceneflow.freeGPUMemory();
	}

	printf("\nPush any key over the scene flow image to finish");
	//mrpt::system::os::getch();

	return 0;
}


