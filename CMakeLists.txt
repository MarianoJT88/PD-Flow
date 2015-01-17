PROJECT(PD-Flow)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4) #I have no idea, mine is 3.0.2
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)  # Required by CMake 2.7+
endif(COMMAND cmake_policy)

FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS "-arch=sm_20")

#Build scene flow CUDA lib
CUDA_ADD_LIBRARY(pdflow_cudalib pdflow_cudalib.h pdflow_cudalib.cu)


SET(DEFAULT_BUILD_EVALUATOR ON)
SET(BUILD_EVALUATOR ${DEFAULT_BUILD_EVALUATOR} CACHE BOOL "Build the scene flow evaluator for a RGB-D frame pair that uses OpenCV")

#Build the scene flow evaluator
IF (BUILD_EVALUATOR)
	FIND_PACKAGE(OpenCV REQUIRED)
	
	ADD_EXECUTABLE(Scene-Flow-Impair  
		main_scene_flow_impair.cpp
		scene_flow_impair.cpp
		scene_flow_impair.h)
		
	TARGET_LINK_LIBRARIES(Scene-Flow-Impair 
			${OpenCV_LIBS}
			${CUDA_LIBRARIES}
			pdflow_cudalib)		
ENDIF (BUILD_EVALUATOR)

SET(DEFAULT_BUILD_RT_VISUALIZATION ON)
SET(BUILD_RT_VISUALIZATION ${DEFAULT_BUILD_RT_VISUALIZATION} CACHE BOOL "Build the scene flow estimator with real-time visualization that requires MRPT and OpenNI2")

#Build real-time scene flow with visualization
IF (BUILD_RT_VISUALIZATION)
	FIND_PACKAGE(MRPT REQUIRED base gui opengl)

	IF (WIN32)
		INCLUDE_DIRECTORIES($ENV{OPENNI2_INCLUDE})
		LINK_DIRECTORIES($ENV{OPENNI2_LIB})
		SET(OpenNI_lib "$ENV{OPENNI2_LIB}/OpenNI2.lib")
	ENDIF (WIN32)
	
	IF (UNIX)
		SET(OpenNI2_libdir "/usr/lib")
		INCLUDE_DIRECTORIES("/usr/include/openni2")
		LINK_DIRECTORIES(${OpenNI2_libdir})
		SET(OpenNI_lib "${OpenNI2_libdir}/libOpenNI2.so")
	ENDIF (UNIX)

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
ENDIF (BUILD_RT_VISUALIZATION)			

			
# Set optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
ENDIF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")

