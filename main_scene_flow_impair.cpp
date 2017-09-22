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

using namespace std;




//==================================================================
//  Arguments for running the algorithm - parsed from command line
//==================================================================
struct Launch_args {
    unsigned int    rows;
    unsigned int    show_help;
    const char      *intensity_filename_1;
    const char      *intensity_filename_2;
    const char      *depth_filename_1;
    const char      *depth_filename_2;
    const char      *output_filename_root;
    bool            no_show;
};

/**
 * Parse arguments from the command line. Valid arguments are:
 * --help (in which case other args are ignored)
 * --rows r
 * --i1 <filename> The first RGB image file name. Defaults to i1.png
 * --i2 <filename> The second RGB image file name
 * --z1 <filename> The first depth image file name
 * --z2 <filename> The second depth image file name
 * @param num_arg Number of arguments present
 * @param argv Array of pointers to arguments
 * @param args A launch_args structure which is populated by this method.
 * @return true if arguments were successfully parsed, otherwise false
 */ 
bool parse_arguments( int num_arg, char *argv[], Launch_args& args) {
	// Initialise with defaults
	args.show_help = 0;
	args.rows = 240;
	args.intensity_filename_1 = "i1.png";
	args.intensity_filename_2 = "i2.png";
	args.depth_filename_1 = "z1.png";
	args.depth_filename_2 = "z2.png";
	args.output_filename_root = "pdflow";
	args.no_show = false;

	// Now check what's provided
	bool parsed_ok = true;
	int arg_idx = 1;
	while( parsed_ok && ( arg_idx < num_arg ) ){
		if( strcmp( "--help", argv[arg_idx]) == 0 ) {
			// Stop parsing after seeing help
			args.show_help = 1;
			break;

		} else if ( strcmp( "--rows", argv[arg_idx]) == 0 ) {
			int rows = -1;
			if( ++arg_idx < num_arg ) {
				rows = stoi( argv[arg_idx]);
			}
			if( rows == 15 || rows == 30 || rows == 60 || rows == 120 || rows == 240 || rows == 480 ) {
				args.rows = rows;
			} else {
				parsed_ok = false;
			} 
		} else if ( strcmp( "--i1", argv[arg_idx]) == 0 ) {
			if( ++arg_idx < num_arg ) {
				args.intensity_filename_1 = argv[arg_idx];
			} else {
				parsed_ok = false;
			}
		} else if ( strcmp( "--i2", argv[arg_idx]) == 0 ) {
			if( ++arg_idx < num_arg ) {
				args.intensity_filename_2 = argv[arg_idx];
			} else {
				parsed_ok = false;
			}
		} else if ( strcmp( "--z1", argv[arg_idx]) == 0 ) {
			if( ++arg_idx < num_arg ) {
				args.depth_filename_1 = argv[arg_idx];
			} else {
				parsed_ok = false;
			}
		} else if ( strcmp( "--z2", argv[arg_idx]) == 0 ) {
			if( ++arg_idx < num_arg ) {
				args.depth_filename_2 = argv[arg_idx];
			} else {
				parsed_ok = false;
			}
		} else if ( strcmp( "--out", argv[arg_idx]) == 0 ) {
			if( ++arg_idx < num_arg ) {
				args.output_filename_root = argv[arg_idx];
			} else {
				parsed_ok = false;
			}
		} else if ( strcmp( "--no-show", argv[arg_idx]) == 0 ) {
			args.no_show = true;
		} else {
			parsed_ok = false;
			break;
		}

		arg_idx++;
	}

	return parsed_ok;
}

// ------------------------------------------------------
//						MAIN
// ------------------------------------------------------

int main(int num_arg, char *argv[])
{	
	//==============================================================================
	//								Read arguments
	//==============================================================================
	Launch_args args;
	if( !parse_arguments( num_arg, argv, args ) || args.show_help ) {
		printf("\n\t       Arguments of the function 'main' \n");
		printf("==============================================================\n\n");
		printf(" --help: Shows this menu... \n\n");
		printf(" --rows r: Number of rows at the finest level of the pyramid. \n");
		printf("\t   Options: r=15, r=30, r=60, r=120, r=240, r=480 (if VGA)\n");
 		printf(" --i1 <filename> : The first RGB image file name. Defaults to i1.png\n" );
 		printf(" --i2 <filename> : The second RGB image file name. Defaults to i2.png\n" );
 		printf(" --z1 <filename> : The first depth image file name. Defaults to z1.png\n" );
 		printf(" --z2 <filename> : The second depth image file name. Defaults to z2.png\n" );
 		printf(" --out <filename>: The output file name root. Omit file extension. Defaults to pdflow\n" );
 		printf(" --no-show       : Don't show the output results. Useful for batch processing\n");
        getwchar();
		return 1;
	}

	//==============================================================================
	//								Main operations
	//==============================================================================

	PD_flow_opencv sceneflow(args.rows, 
							args.intensity_filename_1, 
							args.intensity_filename_2, 
							args.depth_filename_1, 
							args.depth_filename_2, 
							args.output_filename_root);

	//Initialize CUDA and set some internal variables 
    sceneflow.initializeCUDA();

	bool imloaded = sceneflow.loadRGBDFrames();

	if (imloaded == 1)
	{
		sceneflow.solveSceneFlowGPU();

		if( args.no_show )
		{
			cv::Mat image = sceneflow.createImage( );
			sceneflow.saveResults( image );
		}
		else
		{
			sceneflow.showImages();
			sceneflow.showAndSaveResults();
			printf("\nPush any key over the scene flow image to finish\n");
			cv::waitKey(0);
		}

		sceneflow.freeGPUMemory();
	}

	return 0;
}


