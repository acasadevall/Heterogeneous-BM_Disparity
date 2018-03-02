#define KERNEL 3
#define HALF_KERNEL 1
#define MAX_D 16
#define LEFT_IM(i,j,w) left_im[i*w + j]
#define RIGHT_IM(i,j,w) right_im[i*w + j]
#define DISP_IM(i,j,w) R[i*w + j]

__attribute((reqd_work_group_size(32,32,1)))
__kernel void BM_Disparity_WorkGroup(__global unsigned char* restrict left_im, __global unsigned char* restrict right_im, __global unsigned int* restrict R)
{
    /* Get index of the work item */
	int glId = get_global_id(0); int glIdy = get_global_id(1);
    int glSizex = get_global_size(0); int glSizey = get_global_size(1);
    int grIdx = get_group_id(0); int grIdy = get_group_id(1);
	int lIdx = get_local_id(0); int lIdy= get_local_id(1);
    int lSizex = get_local_size(0); int lSizey = get_local_size(1);

    /* WORK-GROUP ID */
    int idx = grIdx*lSizex + lIdx;
    int idy = grIdy*lSizey + lIdy;

	//local int disp_block[MAX_D];
	
	
	__local unsigned char left_patch[KERNEL*KERNEL];
	__local unsigned char right_patch[KERNEL*KERNEL];
	__local unsigned int match_cost;
	__local int disp_block[MAX_D];
	

    if ( (idy > HALF_KERNEL) && (idy < (glSizey - HALF_KERNEL)) )
    {
		if ( (idx > HALF_KERNEL) && (idx < (glSizex - HALF_KERNEL)) )
		{
			// Fill Left Patch
			for (int ky=0;ky<KERNEL;ky++)
			{
				int offset_y = idy - HALF_KERNEL + ky;

				#pragma unroll
				for (int kx=0;kx<KERNEL;kx++)
				{
					int offset_x = idx - HALF_KERNEL + kx;
					left_patch[ky*KERNEL+kx] = LEFT_IM(offset_y,offset_x,glSizex);
					barrier(CLK_LOCAL_MEM_FENCE);
				}
			}
			
			//barrier(CLK_LOCAL_MEM_FENCE);
			
            //bool allow = 1;
			unsigned int idx_disp = 0;

            for (int idx_col = idx;  (idx_col>HALF_KERNEL) && (idx_disp<MAX_D); idx_col--)
			{
				//int A = idx_col - HALF_KERNEL + 1;
				//int B = (idx-idx_col) % MAX_D ? 1 : 0;
				//allow = A & B;
				
				
                int shift = idx-idx_col;
				//#pragma unroll
                //#pragma loop_coalesce
                unsigned int idk_blocks = 0;
				
				// Fill Right Patch
				for (int ky=0;ky<KERNEL;ky++)
				{
					int offset_y = idy - HALF_KERNEL + ky;

					#pragma unroll
					for (int kx=0;kx<KERNEL;kx++)
					{
						int offset_x = idx - HALF_KERNEL + kx;
						right_patch[ky*KERNEL+kx] = RIGHT_IM(offset_y,offset_x,glSizex);
						barrier(CLK_LOCAL_MEM_FENCE);
					}
				}				
				
				match_cost = 0;
				barrier(CLK_LOCAL_MEM_FENCE);
				for (int k=0; k<KERNEL*KERNEL; k++)
				{
					match_cost += abs(left_patch[k] - right_patch[k]);
					barrier(CLK_LOCAL_MEM_FENCE);
				}
				
				/*for (int ky=0; ky<KERNEL; ky++)
				{
                    int offset_y = idy - HALF_KERNEL + ky;

                    #pragma unroll
					for (int kx=0;kx<KERNEL;kx++){
						int offset_x = idx - HALF_KERNEL + kx;
						match_cost += abs(left_patch[ky*KERNEL+kx] - right_patch[ky*KERNEL+kx]);
						//match_cost += abs(left_patch[ky*KERNEL+kx] - right_im[offset_y*glSizex + offset_x]);
						//match_cost += abs(LEFT_IM(offset_y,offset_x+shift,glSizex) - RIGHT_IM(offset_y,offset_x,glSizex));
						//match_cost += abs(left_im[offset_y*glSizex + offset_x + shift] - right_im[offset_y*glSizex + offset_x]);
                    }
					
				}*/
				
				disp_block[idx_disp++] = match_cost;
				barrier(CLK_LOCAL_MEM_FENCE);
			}

			int min = disp_block[0];
			int disp = 0;
            
			//#pragma ii 2
			for (int k=0; k<idx_disp; k++)
			{
				if ( disp_block[k] < min )
				{
					min = disp_block[k];
					disp = k;
				}
			}
				
            DISP_IM(idy,idx,glSizex) = disp;
		}
	}

}