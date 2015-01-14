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
#include <string.h>
#include "scene_flow_impair.h"

using namespace std;


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
		std::getchar();
		return 1;
	}
	else
	{
		if ( string(argv[1]) == "--rows")
			rows = stof(string(argv[2]));
	}

	//==============================================================================
	//								Main operations
	//==============================================================================

	PD_flow_opencv sceneflow(rows);

	//Initialize CUDA and set some internal variables 
    sceneflow.initializeCUDA();

	bool imloaded = sceneflow.loadRGBDFrames();

	if (imloaded == 1)
	{
		sceneflow.showImages();	
		sceneflow.solveSceneFlowGPU();
		sceneflow.showAndSaveResults();
		sceneflow.freeGPUMemory();
		printf("\nPush any key over the scene flow image to finish");
	}

	return 0;
}


