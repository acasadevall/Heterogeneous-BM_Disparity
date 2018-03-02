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
#include <string.h>
#include <stdio.h>
#include "OpenCL_Interface.h"

// need deprecated version for OpenCL < 2.0
// clCreateCommandQueueWithProperties() instead of deprecated clCreateCommandQueue()
#define STRING_BUFFER_LEN 1024

// Helper functions to display parameters returned by OpenCL queries
static void device_info_ulong(cl_device_id device, cl_device_info param, const char* name) {
    cl_ulong a;
    clGetDeviceInfo(device, param, sizeof(cl_ulong), &a, NULL);
    printf("%-40s = %lu\n", name, a);
}
static void device_info_uint(cl_device_id device, cl_device_info param, const char* name) {
    cl_uint a;
    clGetDeviceInfo(device, param, sizeof(cl_uint), &a, NULL);
    printf("%-40s = %u\n", name, a);
}
static void device_info_sizet(cl_device_id device, cl_device_info param, const char* name) {
    size_t a;
    clGetDeviceInfo(device, param, sizeof(size_t), &a, NULL);
    printf("%-40s = %zu\n", name, a);
}
static void device_info_bool(cl_device_id device, cl_device_info param, const char* name) {
    cl_bool a;
    clGetDeviceInfo(device, param, sizeof(cl_bool), &a, NULL);
    printf("%-40s = %s\n", name, (a?"true":"false"));
}
static void device_info_string(cl_device_id device, cl_device_info param, const char* name) {
    char a[STRING_BUFFER_LEN];
    clGetDeviceInfo(device, param, STRING_BUFFER_LEN, &a, NULL);
    printf("%-40s = %s\n", name, a);
}
static void kernel_info_sizet(cl_device_id device, cl_kernel kernel, cl_device_info param, const char* name) {
    size_t a;
    clGetKernelWorkGroupInfo(kernel, device, param, sizeof(size_t), &a, NULL);
    printf("%-40s = %zu\n", name, a);
}
static void kernel_info_sizet_array(cl_device_id device, cl_kernel kernel, cl_device_info param, const char* name) {
    size_t a[3];
    clGetKernelWorkGroupInfo(kernel, device, param, 3*sizeof(size_t), (void*) a, NULL);
    printf("%-40s = (%zu, %zu, %zu)\n", name, a[0], a[1], a[2]);
}

OpenCL_Interface::OpenCL_Interface()
{
    
    cl_int status;

    cl_uint num_devices;
    cl_uint num_platforms;
    
    #ifdef FPGA_OCL

        if(!aocl_utils::setCwdToExeDir())
            exit(EXIT_FAILURE);
        // Get the OpenCL platform.
        
        m_platform = aocl_utils::findPlatform("Intel(R) FPGA");
        if(m_platform == NULL)
        {
            printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
            exit(EXIT_FAILURE);
        }

        char char_buffer[STRING_BUFFER_LEN]; 
        printf("Querying platform for info:\n");
        printf("==========================\n");
        clGetPlatformInfo(m_platform, CL_PLATFORM_NAME, STRING_BUFFER_LEN, char_buffer, NULL);
        printf("%-40s = %s\n", "CL_PLATFORM_NAME", char_buffer);
        clGetPlatformInfo(m_platform, CL_PLATFORM_VENDOR, STRING_BUFFER_LEN, char_buffer, NULL);
        printf("%-40s = %s\n", "CL_PLATFORM_VENDOR ", char_buffer);
        clGetPlatformInfo(m_platform, CL_PLATFORM_VERSION, STRING_BUFFER_LEN, char_buffer, NULL);
        printf("%-40s = %s\n\n", "CL_PLATFORM_VERSION ", char_buffer);

        // Query the available OpenCL devices.
        aocl_utils::scoped_array<cl_device_id> devices;

        devices.reset(aocl_utils::getDevices(m_platform, CL_DEVICE_TYPE_ALL, &num_devices));

        // We'll just use the first device.
        m_device = devices[0];  
    #else
        FILE *fp;
        char *source_str;
        size_t source_size;

        /* Load the source code containing the kernel*/
        fp = fopen((m_kernel_file).c_str(), "r");
        if (!fp)
        {
            std::cerr << "Failed to load kernel!" << std::endl;
            exit(EXIT_FAILURE);
        }

        source_str = (char*) malloc(MAX_SOURCE_SIZE);
        source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose(fp);

        /* Get Platform and Device Info */
        status = clGetPlatformIDs(1, &m_platform, &num_platforms);
        status |= clGetDeviceIDs(m_platform, CL_DEVICE_TYPE_GPU, 1, &m_device, &num_devices);
        checkError(status, "Error calling clGetPlatformIDs");

        std::cout << "Detected OpenCL platforms: " << num_platforms << std::endl;
    #endif

    /* Create OpenCL context */
    // @TODO use &oclContextCallback
    m_context = clCreateContext(NULL, 1, &m_device, NULL, NULL, &status);
    checkError(status, "Failed to create Context");

    /* Create Command Queue */
    m_command_queue = clCreateCommandQueue(m_context, m_device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create Comman Queue");
    
    #ifdef FPGA_OCL
        // Create the kernel Program from the binary file
        std::string binary_file = aocl_utils::getBoardBinaryFile(m_kernel_file.c_str(), m_device);
        printf("Using AOCX: %s\n", binary_file.c_str());
        m_program = aocl_utils::createProgramFromBinary(m_context, binary_file.c_str(), &m_device, 1);
        status = clBuildProgram(m_program, 0, NULL, "", NULL, NULL);
        checkError(status, "Failed to build program");
    #else
        /* Create Kernel Program from the source */
        m_program = clCreateProgramWithSource(m_context, 1, (const char **) &source_str, (const size_t *) &source_size, &status);
        checkError(status, "Failed to create Program with Source");
        
        /* Build Kernel Program */
        status = clBuildProgram(m_program, 1, &m_device, NULL, NULL, NULL);
        if (status != CL_SUCCESS) {
            size_t len;
            char buffer[2048];
            clGetProgramBuildInfo(m_program, m_device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
            std::cout << buffer << std::endl;
            checkError(status, "Failed to Build Program");
        }
    
        free(source_str); //malloc of source code
    #endif

    /* Create OpenCL Kernel */
    m_kernel = clCreateKernel(m_program, m_kernel_name.c_str(), &status);
    checkError(status && m_kernel, "Failed to Create Kernel");
    
}

/*OpenCL_Interface::OpenCL_Interface(const size_t* global_item_size, const size_t* local_item_size, cl_uint dim_item_size)
{
    m_global_item_size = global_item_size;
    m_local_item_size = local_item_size;
    m_dim_item_size = dim_item_size;
}*/

OpenCL_Interface::~OpenCL_Interface()
{
    cl_int status;

    status = clFlush(m_command_queue);
    status |= clFinish(m_command_queue);
    status |= clReleaseKernel(m_kernel);
    status |= clReleaseProgram(m_program);
    status |= clReleaseCommandQueue(m_command_queue);
    status |= clReleaseContext(m_context);

    checkError(status, "Failed to Close OpenCL");

    std::cout << "\nOpenCL closed successfully!";
}

// @TODO add the line and file where the error comes from
cl_int OpenCL_Interface::checkError(cl_int status, const char* msg)
{
    if (status != CL_SUCCESS)
    {
        fprintf(stderr, "[ERROR OPENCL] %s\n In %s:%d\n", msg, __FILE__, __LINE__);
        showError(status);
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}

void OpenCL_Interface::freeOpenCLMemory(cl_mem mem)
{
    cl_int status;
    status = clReleaseMemObject(mem);
    checkError(status, "Failed to Release Memory Object");
}

// @TODO: Encapsulate the function DisplayInfio. Now are in Utils.h
void OpenCL_Interface::showInfo()
{
    printf("Using %s (__kernel %s)", (m_kernel_file).c_str(), (m_kernel_name).c_str());
    deviceInfo();
    kernelInfo();
    printf("[USER] GlobalGrup-Size = (%zu, %zu, %zu)\n", m_global_item_size[0], m_global_item_size[1], m_global_item_size[2]);
    printf("[USER] WorkGrup-Size = (%zu, %zu, %zu)\n", m_local_item_size[0], m_local_item_size[1], m_local_item_size[2]);
}

/*
 * Return the OpenCL Event elsapsed time in nanoseconds
 */
cl_ulong OpenCL_Interface::getStartEndTime(cl_event ev)
{
    cl_int status;

    cl_ulong start, end;
    status = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    checkError(status, "Failed to query event start time");
    status = clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    checkError(status, "Failed to query event end time");

    return static_cast<cl_ulong>((end - start));
}

/**
 * Get the Total Elapsed Time obtained with the OpenCL Events
 * @return total_elapsed_time (time in nanoseconds)
 */
cl_ulong OpenCL_Interface::getTotalElapsedTime()
{
    return total_elapsed_time;
}

void OpenCL_Interface::deviceInfo()
{

    /* Query and display OpenCL information on device and runtime environment */

    printf("\nDevice Info:\n");
    printf("========================\n");
    device_info_string(m_device, CL_DEVICE_NAME, "CL_DEVICE_NAME");
    device_info_string(m_device, CL_DEVICE_VENDOR, "CL_DEVICE_VENDOR");
    device_info_uint(m_device, CL_DEVICE_VENDOR_ID, "CL_DEVICE_VENDOR_ID");
    device_info_string(m_device, CL_DEVICE_VERSION, "CL_DEVICE_VERSION");
    device_info_string(m_device, CL_DRIVER_VERSION, "CL_DRIVER_VERSION");
    device_info_uint(m_device, CL_DEVICE_ADDRESS_BITS, "CL_DEVICE_ADDRESS_BITS");
    device_info_bool(m_device, CL_DEVICE_AVAILABLE, "CL_DEVICE_AVAILABLE");
    device_info_bool(m_device, CL_DEVICE_ENDIAN_LITTLE, "CL_DEVICE_ENDIAN_LITTLE");
    device_info_ulong(m_device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHE_SIZE");
    device_info_ulong(m_device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, "CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE");
    device_info_ulong(m_device, CL_DEVICE_GLOBAL_MEM_SIZE, "CL_DEVICE_GLOBAL_MEM_SIZE");
    device_info_bool(m_device, CL_DEVICE_IMAGE_SUPPORT, "CL_DEVICE_IMAGE_SUPPORT");
    device_info_ulong(m_device, CL_DEVICE_LOCAL_MEM_SIZE, "CL_DEVICE_LOCAL_MEM_SIZE");
    device_info_ulong(m_device, CL_DEVICE_MAX_CLOCK_FREQUENCY, "CL_DEVICE_MAX_CLOCK_FREQUENCY");
    device_info_ulong(m_device, CL_DEVICE_MAX_COMPUTE_UNITS, "CL_DEVICE_MAX_COMPUTE_UNITS");
    device_info_ulong(m_device, CL_DEVICE_MAX_CONSTANT_ARGS, "CL_DEVICE_MAX_CONSTANT_ARGS");
    device_info_ulong(m_device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE");
    device_info_uint(m_device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
    device_info_uint(m_device, CL_DEVICE_MAX_WORK_GROUP_SIZE, "CL_DEVICE_MAX_WORK_GROUP_SIZE");
    device_info_uint(m_device, CL_DEVICE_MAX_WORK_ITEM_SIZES, "CL_DEVICE_MAX_WORK_ITEM_SIZES");
    device_info_uint(m_device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS");
    device_info_uint(m_device, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, "CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE");
    device_info_uint(m_device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR");
    device_info_uint(m_device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT");
    device_info_uint(m_device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT");
    device_info_uint(m_device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG");
    device_info_uint(m_device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT");
    device_info_uint(m_device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, "CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE");
    device_info_sizet(m_device, CL_DEVICE_PROFILING_TIMER_RESOLUTION, "CL_DEVICE_PROFILING_TIMER_RESOLUTION");

    cl_command_queue_properties ccp;
    clGetDeviceInfo(m_device, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &ccp, NULL);
    printf("%-40s = %s\n", "Command queue out of order? ", ((ccp & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)?"true":"false"));
    printf("%-40s = %s\n", "Command queue profiling enabled? ", ((ccp & CL_QUEUE_PROFILING_ENABLE)?"true":"false"));

    printf("========================\n");

}

void OpenCL_Interface::kernelInfo()
{
    printf("\nKernel Info:\n");
    printf("========================\n");
    kernel_info_sizet(m_device, m_kernel, CL_KERNEL_WORK_GROUP_SIZE, "CL_KERNEL_WORK_GROUP_SIZE");
    kernel_info_sizet_array(m_device, m_kernel, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, "CL_KERNEL_COMPILE_WORK_GROUP_SIZE");
    kernel_info_sizet(m_device, m_kernel, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE");
    printf("========================\n");
}

void OpenCL_Interface::showError(cl_int error)
{
    switch(error)
    {
        case -1:
            printf("CL_DEVICE_NOT_FOUND ");
            break;
        case -2:
            printf("CL_DEVICE_NOT_AVAILABLE ");
            break;
        case -3:
            printf("CL_COMPILER_NOT_AVAILABLE ");
            break;
        case -4:
            printf("CL_MEM_OBJECT_ALLOCATION_FAILURE ");
            break;
        case -5:
            printf("CL_OUT_OF_RESOURCES ");
            break;
        case -6:
            printf("CL_OUT_OF_HOST_MEMORY ");
            break;
        case -7:
            printf("CL_PROFILING_INFO_NOT_AVAILABLE ");
            break;
        case -8:
            printf("CL_MEM_COPY_OVERLAP ");
            break;
        case -9:
            printf("CL_IMAGE_FORMAT_MISMATCH ");
            break;
        case -10:
            printf("CL_IMAGE_FORMAT_NOT_SUPPORTED ");
            break;
        case -11:
            printf("CL_BUILD_PROGRAM_FAILURE ");
            break;
        case -12:
            printf("CL_MAP_FAILURE ");
            break;
        case -13:
            printf("CL_MISALIGNED_SUB_BUFFER_OFFSET ");
            break;
        case -14:
            printf("CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST ");
            break;
        case -30:
            printf("CL_INVALID_VALUE ");
            break;
        case -31:
            printf("CL_INVALID_DEVICE_TYPE ");
            break;
        case -32:
            printf("CL_INVALID_PLATFORM ");
            break;
        case -33:
            printf("CL_INVALID_DEVICE ");
            break;
        case -34:
            printf("CL_INVALID_CONTEXT ");
            break;
        case -35:
            printf("CL_INVALID_QUEUE_PROPERTIES ");
            break;
        case -36:
            printf("CL_INVALID_COMMAND_QUEUE ");
            break;
        case -37:
            printf("CL_INVALID_HOST_PTR ");
            break;
        case -38:
            printf("CL_INVALID_MEM_OBJECT ");
            break;
        case -39:
            printf("CL_INVALID_IMAGE_FORMAT_DESCRIPTOR ");
            break;
        case -40:
            printf("CL_INVALID_IMAGE_SIZE ");
            break;
        case -41:
            printf("CL_INVALID_SAMPLER ");
            break;
        case -42:
            printf("CL_INVALID_BINARY ");
            break;
        case -43:
            printf("CL_INVALID_BUILD_OPTIONS ");
            break;
        case -44:
            printf("CL_INVALID_PROGRAM ");
            break;
        case -45:
            printf("CL_INVALID_PROGRAM_EXECUTABLE ");
            break;
        case -46:
            printf("CL_INVALID_KERNEL_NAME ");
            break;
        case -47:
            printf("CL_INVALID_KERNEL_DEFINITION ");
            break;
        case -48:
            printf("CL_INVALID_KERNEL ");
            break;
        case -49:
            printf("CL_INVALID_ARG_INDEX ");
            break;
        case -50:
            printf("CL_INVALID_ARG_VALUE ");
            break;
        case -51:
            printf("CL_INVALID_ARG_SIZE ");
            break;
        case -52:
            printf("CL_INVALID_KERNEL_ARGS ");
            break;
        case -53:
            printf("CL_INVALID_WORK_DIMENSION ");
            break;
        case -54:
            printf("CL_INVALID_WORK_GROUP_SIZE ");
            break;
        case -55:
            printf("CL_INVALID_WORK_ITEM_SIZE ");
            break;
        case -56:
            printf("CL_INVALID_GLOBAL_OFFSET ");
            break;
        case -57:
            printf("CL_INVALID_EVENT_WAIT_LIST ");
            break;
        case -58:
            printf("CL_INVALID_EVENT ");
            break;
        case -59:
            printf("CL_INVALID_OPERATION ");
            break;
        case -60:
            printf("CL_INVALID_GL_OBJECT ");
            break;
        case -61:
            printf("CL_INVALID_BUFFER_SIZE ");
            break;
        case -62:
            printf("CL_INVALID_MIP_LEVEL ");
            break;
        case -63:
            printf("CL_INVALID_GLOBAL_WORK_SIZE ");
            break;
        default:
            printf("UNRECOGNIZED ERROR CODE (%d)", error);
    }
}