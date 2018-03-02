/*
 * Copyright (C) 2018 Universitat Autonoma de Barcelona 
 * Arnau Casadevall Saiz <arnau.casadevall@uab.cat>
 * 
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <string>
#include <chrono>
#include <dirent.h>

#include "BM_Disparity.h"
#include "Utils.h"
#include "OpenCL_Interface.h"
#include "File.h"

#define MAX_SOURCE_SIZE (0x100000)

using namespace std::chrono;
using namespace std;
using namespace cv;

static cl_mem left_memobj = NULL;
static cl_mem right_memobj = NULL;
static cl_mem disp_memobj = NULL;

#ifdef FPGA_OCL
std::string OpenCL_Interface::m_kernel_file = "./kernel/DisparityAOCL_640x480";
std::string OpenCL_Interface::m_kernel_name = "BM_Disparity_WorkGroup";
cl_uint dim_item_size = 1;
static const size_t global_item_size[] = {1, 1, 1};
static const size_t local_item_size[] = {1, 1, 1};
#else
std::string OpenCL_Interface::m_kernel_file = "./kernel/BM_Disparity-GPU.cl";
std::string OpenCL_Interface::m_kernel_name = "BM_Disparity";
cl_uint dim_item_size = 2;
static const size_t global_item_size[] = {640, 480, 1};
static const size_t local_item_size[] = {20, 15, 1};
#endif

cl_platform_id OpenCL_Interface::m_platform = NULL;
cl_device_id OpenCL_Interface::m_device = NULL;
cl_context OpenCL_Interface::m_context = NULL;
cl_command_queue OpenCL_Interface::m_command_queue = NULL;
cl_kernel OpenCL_Interface::m_kernel = NULL;
cl_program OpenCL_Interface::m_program = NULL;

bool OpenCL_Interface::m_use_opencl_events = true;

cl_uint OpenCL_Interface::m_dim_item_size = dim_item_size;
const size_t* OpenCL_Interface::m_global_item_size = global_item_size;
const size_t* OpenCL_Interface::m_local_item_size = local_item_size;

/* Global Arguments */
unsigned int max_d = 16;
unsigned int kernel_size = 7;
bool opencl_vs_cpp = false;
bool use_opencl = false;
bool use_opencl_events = false;
bool kernel_info = false;

void helper()
{
    //cout << "Usage: disparity <LeftImage_Path> <RightImage_Path> [-max-d <value>] [-k <value>] [--use-opencl]" << endl;
    cout << "Usage: disparity <path_images> [-max-d <value>] [-k <value>] [--use-opencl] [--kernel-info] [--use-events] [--opencl-vs-cpp]" << endl;

    exit(EXIT_SUCCESS);
}

void parseArg(int argc, char** argv)
{
    if (argc >= 2)
    {
        for (int k = 2; k < argc; k++)
        {
            if (!(strcmp(argv[k], "-k")) || !(strcmp(argv[k], "--k")))
                kernel_size = (unsigned int) atoi(argv[++k]);
            else if (!strcmp(argv[k], "--max-d"))
                max_d = (unsigned int) atoi(argv[++k]);
            else if (!strcmp(argv[k], "--use-opencl"))
                use_opencl = true;
            else if (!strcmp(argv[k], "--kernel-info"))
                kernel_info = true;
            else if (!strcmp(argv[k], "--use-events"))
                use_opencl_events = true;
            else if (!strcmp(argv[k], "--opencl-vs-cpp"))
            {
                opencl_vs_cpp = true;
                use_opencl_events = true;
            }
            else if (!(strcmp(argv[k], "-h")) || !(strcmp(argv[k], "--help")))
                helper();
            else
            {
                printf("[ERROR] Unrecognized option = %s\n", argv[k]);
                helper();
            }
        }

        if ( (kernel_info || use_opencl_events) && !use_opencl)
        {
            printf("[WARNING] You must activate OpenCL if you want you to view the 'Kernel Info' or 'Use OpenCL Events'. Activate OpenCL with --use-opencl\n");
            printf("Executing with C++ ...\n\n");
        }

        if (use_opencl && opencl_vs_cpp)
        {
            use_opencl = false;
            printf("[WARNING] You have indicated the 'Cpp vs OpenCL' method. Do not need to activate OpenCL with --use-opencl\n");
        }

    } else {
        helper();
    }
}

int main(int argc, char** argv)
{
    const char *dir_images = argv[1];
    parseArg(argc, argv);

    // @TODO Automaticate this assignment
    unsigned int width = 640;
    unsigned int height = 480;

    cout << "-------- INFO -------- " << endl;
    cout << "> Max Disparity: " << max_d << endl;
    cout << "> Kernel Size: " << kernel_size << endl;
    cout << "> Width: " << width << endl;
    cout << "> Height: " << height << endl;
    cout << "---------------------- " << endl;

       
    //unsigned char *left_image_uint8 = (unsigned char*) _aligned_malloc(sizeof(unsigned char)*width*height, 4096);
    //unsigned char *right_image_uint8 = (unsigned char*) _aligned_malloc(sizeof(unsigned char)*width*height, 4096);
    //unsigned int *disp_image_uint8_ocl = (unsigned int*) _aligned_malloc(sizeof(unsigned int)*width*height, 4096);
    unsigned int *disp_image_uint8_ocl = new unsigned int[width * height];
    unsigned char *disp_image_uint8_ocl_norm = new unsigned char[width * height];

    // Create OpenCL Interface
    OpenCL_Interface openCL;
    
    if (use_opencl || opencl_vs_cpp)
    {
        openCL.m_use_opencl_events = use_opencl_events;

        if (kernel_info)
            openCL.showInfo();

        #ifdef FPGA_OCL
        openCL.setMemoryBuffer<unsigned char>(left_memobj, width*height, CL_MEM_ALLOC_HOST_PTR);
        openCL.setMemoryBuffer<unsigned char>(right_memobj, width*height, CL_MEM_ALLOC_HOST_PTR);
        openCL.setMemoryBuffer<unsigned int>(disp_memobj, width*height, CL_MEM_ALLOC_HOST_PTR);
        #else
        openCL.setMemoryBuffer<unsigned char>(left_memobj, width*height, CL_MEM_READ_ONLY);
        openCL.setMemoryBuffer<unsigned char>(right_memobj, width*height, CL_MEM_READ_ONLY);
        openCL.setMemoryBuffer<unsigned int>(disp_memobj, width*height, CL_MEM_WRITE_ONLY);
        #endif

        openCL.setKernelArgs(left_memobj, 0);
        openCL.setKernelArgs(right_memobj, 1);
        openCL.setKernelArgs(disp_memobj, 2);
        openCL.setKernelArgs(max_d, 3);
    }

    const char *left_dir = "left";
    const char *right_dir = "right";
    char dir_left_images[PATH_MAX];
    char dir_right_images[PATH_MAX];
    char left_image_file[PATH_MAX];
    char right_image_file[PATH_MAX];

    sprintf(dir_left_images, "%s/%s", dir_images, left_dir); // Left Images
    sprintf(dir_right_images, "%s/%s", dir_images, right_dir); // Right Images

    File imageFiles(dir_left_images);
    std::vector<std::string> list_files = imageFiles.getListFiles();
    //imageFiles.showListFiles(list_files);
    
    int time_elapsed = 0;
    for(int i=0; i< (int) list_files.size(); ++i)
    {
        sprintf(left_image_file, "%s/%s", dir_left_images, list_files[i].c_str()); // Left Image
        sprintf(right_image_file, "%s/%s", dir_right_images, list_files[i].c_str()); // Right Image

        // @TODO Try to avoid OpenCV for read images and show them
        Mat left_image = imread(left_image_file, IMREAD_GRAYSCALE); // grayscale Left image
        Mat right_image = imread(right_image_file, IMREAD_GRAYSCALE); // grayscale Right image

        if (!left_image.data && !right_image.data)
        {
            cerr << "[ERROR] No image data" << endl;
            exit(EXIT_FAILURE);
        }

        assert(left_image.cols == right_image.cols);
        assert(left_image.rows == right_image.rows);

        unsigned int width = (unsigned int) left_image.cols;
        unsigned int height = (unsigned int) left_image.rows;
        
        unsigned char *left_image_uint8 = matToUint8(left_image);
        unsigned char *right_image_uint8 = matToUint8(right_image);
        //left_image_uint8 = matToUint8(left_image);
        //right_image_uint8 = matToUint8(right_image);
        
        //printf("Pointer Left %p\n", left_image_uint8);
        //printf("Pointer Right %p\n", right_image_uint8);

        if (use_opencl)
        {
            /* For each interation */
            high_resolution_clock::time_point t1_ocl = high_resolution_clock::now();
            openCL.enqueueWriteBuffer(left_memobj, left_image_uint8, width*height, CL_TRUE);
            openCL.enqueueWriteBuffer(right_memobj, right_image_uint8, width*height, CL_TRUE);

            openCL.run(disp_memobj, disp_image_uint8_ocl, width*height, CL_TRUE);
            high_resolution_clock::time_point t2_ocl = high_resolution_clock::now();
            
            if (use_opencl_events)
                time_elapsed += openCL.getTotalElapsedTime()*1e-6;
            else
                time_elapsed += duration_cast<milliseconds>(t2_ocl - t1_ocl).count();
            
            if (time_elapsed >= 500)
            {
                if (use_opencl_events)
                    cout << "Time (ms): " << openCL.getTotalElapsedTime()*1e-6 << "  FPS: " << (1.0/openCL.getTotalElapsedTime())*1e9 << endl;
                else{
                    auto duration_ocl = duration_cast<milliseconds>(t2_ocl - t1_ocl).count();
                    cout << "Time (ms): " << duration_ocl << "  FPS: " << (1.0/duration_ocl)*1e3 << endl;
                }
                
                time_elapsed = 0;
            }

            // Norm for OCL
            int max_value = 0;
            int min_value = disp_image_uint8_ocl[0];
            for (int k = 0; k < (width * height); k++)
            {
                if (min_value > disp_image_uint8_ocl[k])
                    min_value = disp_image_uint8_ocl[k];

                if (disp_image_uint8_ocl[k] > max_value)
                    max_value = disp_image_uint8_ocl[k];
            }

            for (int k = 0; k < (width * height); k++)
                disp_image_uint8_ocl_norm[k] = static_cast<unsigned char>(disp_image_uint8_ocl[k] * 255 / max_value);

            //cout << "Max - Min Disparity (OCL) := " << max_value << " - " << min_value;
            //cout << " Done!\n" << endl;

            Mat disp_image_ocl(height, width, CV_8UC1, disp_image_uint8_ocl_norm); // uint8 to Mat

            #ifdef FPGA_OCL
            imshow("Image OpenCL_FPGA", disp_image_ocl);
            #else
            imshow("Image OpenCL_GPU", disp_image_ocl);
            #endif
            waitKey(1);
        }
        else if (opencl_vs_cpp)
        {
            // Turn for OpenCL
            cout << "\nComputing BM Disparity Map OpenCL ..." << endl;
            openCL.enqueueWriteBuffer(left_memobj, left_image_uint8, width*height, CL_TRUE);
            openCL.enqueueWriteBuffer(right_memobj, right_image_uint8, width*height, CL_TRUE);

            openCL.run(disp_memobj, disp_image_uint8_ocl, width*height, CL_TRUE);

            cout << "Time (ms): " << openCL.getTotalElapsedTime()*1e-6 << "  FPS: " << (1.0/openCL.getTotalElapsedTime())*1e9 << endl;

            // Norm for OCL
            int max_value = 0;
            int min_value = disp_image_uint8_ocl[0];
            for (int k = 0; k < (width * height); k++)
            {
                if (min_value > disp_image_uint8_ocl[k])
                    min_value = disp_image_uint8_ocl[k];

                if (disp_image_uint8_ocl[k] > max_value)
                    max_value = disp_image_uint8_ocl[k];
            }

            for (int k = 0; k < (width * height); k++)
                disp_image_uint8_ocl_norm[k] = static_cast<unsigned char>(disp_image_uint8_ocl[k] * 255 / max_value);

            // Turn for C++
            BM_Disparity disparity(width, height, max_d, kernel_size);
            cout << "\nComputing BM Disparity Map C++ ..." << endl;
            high_resolution_clock::time_point t1_cpp = high_resolution_clock::now();
            unsigned char* disp_image_uint8_norm = disparity.computeBM_Dispartity(left_image_uint8, right_image_uint8);
            high_resolution_clock::time_point t2_cpp = high_resolution_clock::now();

            auto duration_cpp = duration_cast<milliseconds>(t2_cpp - t1_cpp).count();
            cout << "Time (ms): " << duration_cpp << "  FPS: " << (1.0/duration_cpp)*1e3 << endl;

            //Mat disp_image_ocl(height, width, CV_8UC1, disp_image_uint8_ocl_norm); // uint8 to Mat
            //Mat disp_image_cpp(height, width, CV_8UC1, disp_image_uint8_norm); // uint8 to Mat

            unsigned char *out_diff = new unsigned char[width*height];
            int out = 0;
            for (int k=0; k<width*height; k++)
            {
                out_diff[k] = abs(disp_image_uint8_ocl_norm[k] - disp_image_uint8_norm[k]);
                out += pow(abs(disp_image_uint8_ocl_norm[k] - disp_image_uint8_norm[k]), 2.0);
            }

            if (!out) cout << "SSD Difference:= " << out << " [OK - MATCH]" << endl;
            if (out) cout << "SSD Difference:= " << out << " [ERROR - MISMATCH]" << endl;
            cout << "---------------------------------------" << endl;

            Mat disp_frame_diff(height, width, CV_8UC1, out_diff); // uint8 to Mat*/

            //imshow("BM Disparity OpenCL", disp_image_ocl);
            //imshow("BM Disparity C++", disp_image_cpp);
            imshow("Diff Image", disp_frame_diff);
            waitKey(10);
        }
        else{
            // C++ computation
            //BM_Disparity disparity(width, height);
            BM_Disparity disparity(width, height, max_d, kernel_size);

            high_resolution_clock::time_point t1_cpp = high_resolution_clock::now();
            unsigned char* disp_image_uint8_norm = disparity.computeBM_Dispartity(left_image_uint8, right_image_uint8);
            high_resolution_clock::time_point t2_cpp = high_resolution_clock::now();

            auto duration_c = duration_cast<milliseconds>(t2_cpp - t1_cpp).count();
            cout << "C++ Time (ms): " << duration_c << "  FPS: " << (1.0/duration_c)*1e3 << endl;

            Mat disp_image_cpp(height, width, CV_8UC1, disp_image_uint8_norm); // uint8 to Mat

            //imwrite("output/DisparityImage_"+image_name+".png", disp_image);
            imshow("Image C++", disp_image_cpp);
            waitKey(10);
        }

        //imwrite(("output/LeftImage_"+image_name+".png").c_str(), left_image);
        //imwrite("output/RightImage_"+image_name+".png", right_image);
        //imwrite("output/FrameDifference.png", disp_frame_diff);
        //imwrite("output/DisparityOCL_"+image_name+".png", disp_image_ocl);

        /*auto duration_c = duration_cast<milliseconds>( t2_c - t1_c ).count();
        auto duration_ocl = duration_cast<milliseconds>( t2_ocl_RB - t1_ocl ).count();
        auto duration_ocl_WB = duration_cast<milliseconds>( t2_ocl_WB - t1_ocl ).count();
        auto duration_ocl_NDRK = duration_cast<microseconds>( t2_ocl_NDRK - t1_ocl_NDRK ).count();
        auto duration_ocl_RB = duration_cast<milliseconds>( t2_ocl_RB - t1_ocl_RB ).count();

        cout << "\n/ ----------- Time Summary ----------- /" << endl;
        cout << "C++ Time (ms): " << duration_c << endl;
        cout << "OpenCL Time (ms): " << duration_ocl << endl;
        cout << "  - OpenCL Time Write Buffer (ms): " << duration_ocl_WB << endl;
        cout << "  - OpenCL Time NDRKernel (us): " << duration_ocl_NDRK << endl;
        cout << "  - OpenCL Time Read Buffer (ms): " << duration_ocl_RB << endl;
        cout << "/ ------------------------------------ /\n" << endl;*/

    }

    if (use_opencl)
    {
        openCL.freeOpenCLMemory(left_memobj);
        openCL.freeOpenCLMemory(right_memobj);
        openCL.freeOpenCLMemory(disp_memobj);
    }

    return 0;
}