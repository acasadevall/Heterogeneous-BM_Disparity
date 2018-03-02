//#define KERNEL 7
//#define HALF_KERNEL 3
//#define MAX_D 16
//#define WIDTH 384
//#define HEIGHT 288*/
#define LEFT_IM(i,j,w) left_im[i*w + j]
#define RIGHT_IM(i,j,w) right_im[i*w + j]
#define DISP_IM(i,j,w) disp_im[i*w + j]

__attribute((max_global_work_dim(0)))
__kernel void BM_Disparity_WorkGroup(
	__global unsigned char* restrict left_im, __global unsigned char* restrict right_im, __global unsigned int* restrict disp_im,
	unsigned int const WIDTH, unsigned int const HEIGHT, unsigned int const KERNEL, unsigned int const MAX_D)
{
   	int glIdx = get_global_id(0);
	
	/*__attribute__((numbanks(1), bankwidth(32)))*/
	unsigned char right_patch[512][32];

	
	const int HALF_KERNEL = KERNEL/2;
	
	
	
	//if ( (glIdy >= HALF_KERNEL) && (glIdy < (HEIGHT-HALF_KERNEL)) )
	for (int glIdy=HALF_KERNEL; glIdy<(HEIGHT-HALF_KERNEL); glIdy++)
	{
		
		//if (glIdy == HALF_KERNEL)
		//	printf("Width: %d, Height: %d, Kernel: %d, Half-Kernel: %d, Maxd: %d\n", WIDTH, HEIGHT, KERNEL, HALF_KERNEL, MAX_D);
		
		for (int rx=0; rx<WIDTH; rx++)
		{
			for (int ry=0; ry<KERNEL; ry++)
			{
				// for each index_width of the image, we copy into a local block all the column of the Kernel (Column by Column)
				int offset_y = glIdy + ry - HALF_KERNEL;
				right_patch[rx][ry] = RIGHT_IM(offset_y,rx,WIDTH);
			}
		}
		
		
		//for (int glIdx=HALF_KERNEL; glIdx<(WIDTH-HALF_KERNEL); glIdx++)
		if ( (glIdx >= HALF_KERNEL) && (glIdx < (WIDTH-HALF_KERNEL)) )
		{	
			
			int idx_disp = 0;
			int disp_block[100];
			for (int idx_col=glIdx; (idx_col>=HALF_KERNEL) && (idx_disp<MAX_D); idx_col--)
			{				
				int match_cost = 0;				
				
				if ( (idx_col >= HALF_KERNEL) || (idx_disp<=MAX_D) )
				{
					
				
				// compute the Kernel indexing row by row on the left_image (contiguous pixels in memory)
				
				/*if ( (glIdx == (WIDTH-HALF_KERNEL-1)) && (glIdy==10) && (idx_disp<=0) )
						printf(" - x%d y%d\n",idx_col, glIdy);*/
					
				#pragma unroll
				for (int ky=-HALF_KERNEL;ky<=HALF_KERNEL;ky++)
				{	
					#pragma unroll
					for (int kx=-HALF_KERNEL;kx<=HALF_KERNEL;kx++)
					{						
						match_cost += abs(LEFT_IM((glIdy+ky),(glIdx+kx),WIDTH) - right_patch[(kx+idx_col)%512][ky+HALF_KERNEL]);
						//match_cost += abs(LEFT_IM((glIdy+ky),(glIdx+kx),WIDTH) - RIGHT_IM((glIdy+ky),(idx_col+kx),WIDTH));
						
						/*if ( (glIdx == (WIDTH-HALF_KERNEL-1)) && (glIdy==10) && (idx_disp<=2) )
							printf("[%d] - [%d/%d] (%d - %d)\n", (kx+idx_col)%512, (glIdy+ky),(ky+HALF_KERNEL), LEFT_IM((glIdy+ky),(glIdx+kx),WIDTH), right_patch[(kx+idx_col)%512][ky+HALF_KERNEL]);
							//printf("[%d] - [%d] (%d - %d)\n", (kx+idx_col)%512, (ky+HALF_KERNEL), LEFT_IM((glIdy+ky),(glIdx+kx),WIDTH), LEFT_IM((glIdy+ky),(idx_col+kx),WIDTH));*/
					}
					
					/*if ( (glIdx == (WIDTH-HALF_KERNEL-1)) && (glIdy==10) && (idx_disp<=2) )
						printf("---------\n");*/
				}
				
				}else{
					printf("ERROR\n");
				}
				
				disp_block[idx_disp++] = match_cost;
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
				
			DISP_IM(glIdy,glIdx,WIDTH) = disp; //right_patch[glIdx][6];
		}
	}

}