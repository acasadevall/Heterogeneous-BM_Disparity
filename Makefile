TARGET = DisparityMap.exe
TARGET_FPGAOCL = DisparityMap_OCL.exe

ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation)
endif
ifeq ($(wildcard $(INTELFPGAOCLSDKROOT)/host/include/CL/opencl.h),)
$(error Set INTELFPGAOCLSDKROOT to the root directory of the Intel(R) FPGA SDK for OpenCL(TM) software installation.)
endif

# OpenCL compile and link flags.
AOCL_COMPILE_CONFIG := -I/opt/intelFPGA/17.1/hld/host/include -IFPGA/inc
AOCL_LINK_CONFIG := -L/opt/intelFPGA/17.1/hld/host/linux64/lib -lOpenCL
	
# OpenCL compile and link flags.
#AOCL_COMPILE_CONFIG := $(shell aocl compile-config ) -IFPGA/inc
#AOCL_LINK_CONFIG := $(shell aocl link-config )

# Compilation flags
ifeq ($(DEBUG),1)
CXXFLAGS += -g
else
XXFLAGS += 
endif

# Libraries to use, objects to compile
SRCS_FILES := $(wildcard src/*.cpp)
SRCS_FILES_FPGAOCL := $(wildcard src/*.cpp FPGA/src/AOCLUtils/*.cpp)

# OpenCL Compile and Link Flags.
OTHER_LIBS := -lrt -lpthread#-lm

#OpenCV
# opencv_arucoopencv_bgsegmopencv_bioinspiredopencv_calib3dopencv_ccalibopencv_coreopencv_datasetsopencv_dpmopencv_faceopencv_features2dopencv_flannopencv_freetypeopencv_fuzzyopencv_hdfopencv_highguiopencv_imgcodecsopencv_imgprocopencv_line_descriptoropencv_mlopencv_objdetectopencv_optflowopencv_phase_unwrappingopencv_photoopencv_plotopencv_regopencv_rgbdopencv_saliencyopencv_shapeopencv_stereoopencv_stitchingopencv_structured_lightopencv_superresopencv_surface_matchingopencv_textopencv_videoopencv_videoioopencv_videostabopencv_ximgprocopencv_xobjdetectopencv_xphoto
OPENCV_INC := -I/usr/local/include
OPENCV_LIB := -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_video -lopencv_highgui -lopencv_ml -lopencv_calib3d -lopencv_imgcodecs

# OCL Compile and Link Flags.
OPENCL_INC := -I/usr/local/cuda/include/
OPENCL_LIB := -L/usr/local/cuda/lib64/ -lOpenCL

# Make it all!
all :  gpu-ocl fpga-ocl
	
gpu-ocl :
	g++ -std=c++14 $(SRCS_FILES) $(OPENCV_INC) $(OPENCV_LIB) $(OPENCL_INC) $(OPENCL_LIB) -o $(TARGET)
	
fpga-ocl :
	g++ -std=c++14 $(SRCS_FILES_FPGAOCL) $(OPENCV_INC) $(OPENCV_LIB) -fPIC -DFPGA_OCL $(AOCL_COMPILE_CONFIG) $(AOCL_LINK_CONFIG) $(OTHER_LIBS) -o $(TARGET_FPGAOCL)

# Standard make targets
clean :
	@rm -f *.o $(TARGET)
	@rm -f *.o $(TARGET_FPGAOCL)
	
.PHONY : all clean