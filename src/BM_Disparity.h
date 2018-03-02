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

#ifndef DISPARITYMAP_BM_DISPARITY_H
#define DISPARITYMAP_BM_DISPARITY_H

class BM_Disparity {

public:
    BM_Disparity(unsigned int w, unsigned int h);
    BM_Disparity(unsigned int w, unsigned int h, unsigned int max_d, unsigned int kernel_size);
    ~BM_Disparity();

    unsigned char* computeBM_Dispartity(unsigned char* left_image, unsigned char* right_image);

private:
    unsigned int m_width;
    unsigned int m_height;
    unsigned int m_max_disp;
    unsigned int m_kernel_size;
    unsigned int m_half_kernel_size;

    unsigned char* GetKernelImage(unsigned char* image, unsigned int i, unsigned int j);
    unsigned int MatchCost(unsigned char* a, unsigned char* b);
};


#endif //DISPARITYMAP_BM_DISPARITY_H
