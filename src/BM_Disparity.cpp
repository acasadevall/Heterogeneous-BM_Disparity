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
#include "BM_Disparity.h"
#include <stdlib.h>

BM_Disparity::BM_Disparity(unsigned int w, unsigned int h)
{
    m_width = w;
    m_height = h;
    m_max_disp = 16;
    m_kernel_size = 1;
    m_half_kernel_size = m_kernel_size/2;
}

BM_Disparity::BM_Disparity(unsigned int w, unsigned int h, unsigned int max_d, unsigned int kernel_size)
{
    m_width = w;
    m_height = h;
    m_max_disp = max_d;
    m_kernel_size = kernel_size;
    m_half_kernel_size = m_kernel_size/2;
}

BM_Disparity::~BM_Disparity()
{

};

unsigned char* BM_Disparity::computeBM_Dispartity(unsigned char* left_image, unsigned char* right_image) {

    // static vector for the dispartity block
    // each line will contain a disparity with a maximum idx_disp

    int disp_block[m_max_disp];
    unsigned int* disp_image = new unsigned int[m_width*m_height];
    unsigned char* disp_image_norm = new unsigned char[m_width*m_height];

    //init zeros
    for (int k=0; k<m_width*m_height; k++)
        disp_image[k] = 0;

    int max_value = 0;

    for (int i=0; i<m_height; i++)
    {
        if ( (i < m_half_kernel_size) || (i >= (m_height - m_half_kernel_size)) )
            continue;

        for (int j=0; j<m_width; j++)
        {
            if ( (j < m_half_kernel_size) || (j >= (m_width - m_half_kernel_size)) )
                continue;

            // Take a point on the left, search the correspondence one in the right image and shift this to the left
            unsigned int idx_disp = 0;

            for (int idx_col = j; (idx_disp < m_max_disp) && (idx_col >= (int) m_half_kernel_size); idx_col--)
            {
                // SAD Match Cost between Patches
                unsigned int match_cost = 0;
                for (int ki=i-m_half_kernel_size;ki<=(i+m_half_kernel_size);ki++)
                {
                    for (int kj=idx_col-m_half_kernel_size;kj<=(idx_col+m_half_kernel_size);kj++)
                    {
                        match_cost += abs(left_image[ki*m_width + kj + (j-idx_col)] - right_image[ki*m_width + kj]);
                    }
                }

                disp_block[idx_disp++] = match_cost; //MatchCost(left_block, right_block);
            }

            // get the minimum cost for each disp_block and maximum cost for the whole image
            int min = disp_block[0];
            unsigned int disp = 0;
            for (unsigned int k=0; k<idx_disp; k++)
            {
                if ( disp_block[k] < min )
                {
                    min = disp_block[k];
                    disp = k;
                }
            }

            if (disp > max_value)
                max_value = disp;

            disp_image[i*m_width + j] = disp;
        }
    }

    // norm image
    for (int k = 0; k < m_width * m_height; k++)
        disp_image_norm[k] = static_cast<unsigned char>(disp_image[k] * 255 / max_value);

    return disp_image_norm;
}

unsigned int BM_Disparity::MatchCost(unsigned char *a, unsigned char *b) {

    //TODO static_assert(sizeof(a) == sizeof(b), "Mismatch dimensions between blocks!");

    unsigned int match_cost = 0;
    for (int k=0; k<(m_kernel_size*m_kernel_size); k++) {
        match_cost += abs(a[k] - b[k]);
    }

    return match_cost;
}

unsigned char* BM_Disparity::GetKernelImage(unsigned char* image, unsigned int i, unsigned int j)
{
    /*uint8_t kernel_block[m_kernel_size];

    for (int k=0; k<m_kernel_size; k++)
        kernel_block[k] = image[(i-m_helf_kernel_size)*m_width + (j-m_helf_kernel_size)];*/

    return 0;
}