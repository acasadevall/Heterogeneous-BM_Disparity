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

#ifndef DISPARITYMAP_OPENCL_H
#define DISPARITYMAP_OPENCL_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef FPGA_OCL
#include "CL/opencl.h"
#include "AOCLUtils/aocl_utils.h"
#else
#include <CL/cl.h>
#endif

#include <string>

#define MAX_SOURCE_SIZE (0x100000)

class OpenCL_Interface{

private:
    static std::string m_kernel_file;
    static std::string m_kernel_name;
    static cl_platform_id m_platform;
    static cl_device_id m_device;
    static cl_context m_context;
    static cl_command_queue m_command_queue;
    static cl_kernel m_kernel;
    static cl_program m_program;
    cl_ulong total_elapsed_time;

public:
    static bool m_use_opencl_events;
    static cl_uint m_dim_item_size;
    static const size_t* m_global_item_size;
    static const size_t* m_local_item_size;

    OpenCL_Interface();
    //OpenCL_Interface(size_t* global_item_size, size_t* local_item_size, cl_uint dim_item_size);
    ~OpenCL_Interface();
    
    // @TODO
    //void initOpenCL();
    //void initOpenCL_FPGA();

    cl_ulong getTotalElapsedTime();

    template <class Memory>
    void setKernelArgs(Memory mem, cl_uint arg)
    {
        /* Set OpenCL Kernel Parameters */
        cl_int status;
        status = clSetKernelArg(m_kernel, arg, sizeof(Memory), (void *) &mem);
        checkError(status, "Failed to Set Kernel Arguments");
    }

    template <class Buffer>
    void setMemoryBuffer(cl_mem& memory, size_t size, cl_mem_flags type)
    {
        /* Create Memory Buffer */
        cl_int status;
        memory = clCreateBuffer(m_context, type, size*sizeof(Buffer), NULL, &status);
        checkError(status, "Failed to Allocate Memory Buffer");
    }

    template <class Buffer>
    void enqueueWriteBuffer(cl_mem memory, Buffer* input, size_t size, cl_bool type)
    {
        /* Enqueue Write Buffer */
        cl_int status;
        cl_event event;

        if (!m_use_opencl_events)
            event = NULL;
        
        status = clEnqueueWriteBuffer(m_command_queue, memory, type, 0, size*sizeof(Buffer), (void *) input, 0, NULL, &event);
        checkError(status, "Failed to Enqueue Write Buffer");

        if (m_use_opencl_events)
            std::cout << "Elapsed Write Buffer (us): " << getStartEndTime(event)*1e-3 << std::endl;
    }

    template <class Buffer>
    void run(cl_mem output_mem, Buffer* output, size_t size, cl_bool type)
    {
        /* Execute OpenCL Kernel */

        cl_int status;
        cl_event event_ndr;
        cl_event event_read;
        total_elapsed_time = 0;

        if (!m_use_opencl_events)
        {
            event_ndr = NULL;
            event_read = NULL;
        }

        /* Execute NDRange Kernel */
        status = clEnqueueNDRangeKernel(m_command_queue, m_kernel, m_dim_item_size, NULL, m_global_item_size, m_local_item_size, 0, NULL, &event_ndr);
        //status = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);
        checkError(status, "Failed to Enqueue NDRange Kernel");

        /* Copy results from the memory buffer */
        status = clEnqueueReadBuffer(m_command_queue, output_mem, type, 0, size*sizeof(Buffer), (void *) output, 0, NULL, &event_read);
        checkError(status, "Failed to Enqueue Read Buffer");

        if (m_use_opencl_events)
        {
            cl_ulong eta_ndr = getStartEndTime(event_ndr);
            cl_ulong eta_read = getStartEndTime(event_read);
            
            std::cout << "Elapsed Enqueue NDRange Kernel (us): " << eta_ndr*1e-3 << std::endl;
            std::cout << "Elapsed Read Buffer (us): " << eta_read*1e-3 << std::endl;
            
            total_elapsed_time += (eta_ndr + eta_read);
        }
    }

    void freeOpenCLMemory(cl_mem mem);

    void showInfo();

private:
    cl_ulong getStartEndTime(cl_event ev);
    cl_int checkError(cl_int status, const char* msg);
    void showError(cl_int error);
    void kernelInfo();
    void deviceInfo();

};


#endif //DISPARITYMAP_OPENCL_H
