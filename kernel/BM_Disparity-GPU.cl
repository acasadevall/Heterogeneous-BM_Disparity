#define KERNEL 7
#define HALF_KERNEL KERNEL/2

//__attribute((reqd_work_group_size(20,15,1)))
__kernel void BM_Disparity(__global unsigned char* restrict left_im, __global unsigned char* restrict right_im, __global unsigned int* restrict disp_im, unsigned int MAX_D)
{

    /* Get index of the work item */
    //int2 globalId = (int2)(get_global_id(0), get_global_id(1));
    int2 globalSize = (int2)(get_global_size(0), get_global_size(1));
    int2 groupId = (int2)(get_group_id(0), get_group_id(1));
    int2 localId = (int2)(get_local_id(0), get_local_id(1));
    int2 localSize = (int2)(get_local_size(0), get_local_size(1));

    /* WORK-GROUP ID */
    unsigned int idx = groupId.x*localSize.x + localId.x;
    unsigned int idy = groupId.y*localSize.y + localId.y;

    if ( (idy >= HALF_KERNEL) && (idy < (globalSize.y - HALF_KERNEL)) )
    {
		if ( (idx >= HALF_KERNEL) && (idx < (globalSize.x - HALF_KERNEL)) )
		{			
			int idx_disp = 0;
			int disp_block[256];
			for (int idx_col = idx; (idx_col>=HALF_KERNEL) && (idx_disp<MAX_D); idx_col--)
			{
				unsigned int match_cost = 0;
				#pragma unroll
				for (int ky=idy-HALF_KERNEL;ky<=(idy+HALF_KERNEL);ky++)
				{
					#pragma unroll
					for (int kx=idx_col-HALF_KERNEL;kx<=(idx_col+HALF_KERNEL);kx++)
						match_cost += abs(left_im[ky*globalSize.x + kx + (idx-idx_col)] - right_im[ky*globalSize.x + kx]);
				}

				disp_block[idx_disp++] = match_cost;
			}

			int min = disp_block[0];
			int disp = 0;
			for (int k=0; k<idx_disp; k++)
			{
				if ( disp_block[k] < min )
				{
					min = disp_block[k];
					disp = k;
				}
			}

			disp_im[idy*globalSize.x + idx] = disp;
		}
	}

}