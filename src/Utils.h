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

#include "CL/opencl.h"
#include <opencv2/opencv.hpp>

#define STRING_BUFFER_LEN 1024

#ifndef DISPARITYMAP_UTILS_H
#define DISPARITYMAP_UTILS_H

unsigned char* matToUint8(cv::Mat image)
{
    int total_size = (int) (image.total() * image.elemSize());
    unsigned char* uint8_image = new unsigned char[total_size];
    memcpy(uint8_image, image.data, total_size*sizeof(unsigned char));

    /*std::cout << "> Pixel Test ... ";
    for (int k=0;k<image.cols*image.rows;k++)
    {
        assert(static_cast<unsigned >(uint8_image[k]) == static_cast<unsigned >(image.at<uchar>(k/image.cols, k%image.cols)));
    }
    std::cout << "Passed [OK]" << std::endl;*/

    return uint8_image;
}

#endif //DISPARITYMAP_UTILS_H
