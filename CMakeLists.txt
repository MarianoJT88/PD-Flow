PROJECT(PD-Flow)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)  # Required by CMake 2.7+
endif(COMMAND cmake_policy)


FIND_PACKAGE(MRPT REQUIRED base gui opengl slam maps)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)

INCLUDE_DIRECTORIES($ENV{OPENNI2_INCLUDE})
LINK_DIRECTORIES($ENV{OPENNI2_LIB})
#SET(OpenNI_lib "/usr/lib/libOpenNI2.so")
SET(OpenNI_lib "C:/Users/Mariano/Programas/OpenNI2_32/Lib/OpenNI2.lib")

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
#set(CUDA_NVCC_FLAGS "-arch=sm_30 -O3 -use_fast_math")
set(CUDA_NVCC_FLAGS "-arch=sm_30")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35)
CUDA_ADD_LIBRARY(pdflow_cudalib pdflow_cudalib.h pdflow_cudalib.cu)
#message( ${CUDA_LIBRARIES} )

		
ADD_EXECUTABLE(Scene-Flow-Visualization  
	main_scene_flow_visualization.cpp
	scene_flow_visualization.cpp
	scene_flow_visualization.h
	legend_pdflow.xpm)
	
TARGET_LINK_LIBRARIES(Scene-Flow-Visualization
		${MRPT_LIBS}
		${CUDA_LIBRARIES}
		${OpenNI_lib}
		pdflow_cudalib)

ADD_EXECUTABLE(Scene-Flow-Impair  
	main_scene_flow_impair.cpp
	scene_flow_impair.cpp
	scene_flow_impair.h
	)
	
TARGET_LINK_LIBRARIES(Scene-Flow-Impair 
		${OpenCV_LIBS}
		${CUDA_LIBRARIES}
		${MRPT_LIBS}
		pdflow_cudalib)		

			
 
# Set optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
ENDIF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")

