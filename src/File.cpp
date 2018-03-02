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

#include <sys/stat.h>
#include <dirent.h>
#include <algorithm>
#include <string.h>
#include <iostream>
#include <cstring>
#include <stdio.h>
#include <iterator> // std::ostream_iterator
#include <sstream> // std::istringstream
#include <cctype> // std::isdigit
#include "File.h"

bool compareNat(const std::string& a, const std::string& b)
{
    if (a.empty())
        return true;
    if (b.empty())
        return false;
    if (std::isdigit(a[0]) && !std::isdigit(b[0]))
        return true;
    if (!std::isdigit(a[0]) && std::isdigit(b[0]))
        return false;
    if (!std::isdigit(a[0]) && !std::isdigit(b[0]))
    {
        if (std::toupper(a[0]) == std::toupper(b[0]))
            return compareNat(a.substr(1), b.substr(1));
        return (std::toupper(a[0]) < std::toupper(b[0]));
    }

    // Both strings begin with digit --> parse both numbers
    std::istringstream issa(a);
    std::istringstream issb(b);
    int ia, ib;
    issa >> ia;
    issb >> ib;
    if (ia != ib)
        return ia < ib;

    // Numbers are the same --> remove numbers and recurse
    std::string anew, bnew;
    std::getline(issa, anew);
    std::getline(issb, bnew);
    return (compareNat(anew, bnew));
}

File::File(const char* path)
{
    m_path = path;
}

File::File(std::string& path)
{
    m_path = path;
}


File::~File()
{
}

bool File::exists()
{
    struct stat res;
    int ret = stat(m_path.c_str(), &res);

    return !ret;
}

void File::mkdirs()
{
    File* parent = getParentFile();

    if (parent != NULL)
    {
        parent->mkdirs();
        delete parent;
    }

    std::string cmd = "mkdir ";
    cmd.append(m_path);

    printf("[EXE]: %s\n", cmd.c_str());

    system(cmd.c_str());
}

File* File::getParentFile()
{
    int pos = m_path.find_last_of("/");

    if (pos == std::string::npos)
        return NULL;

    std::string parent = m_path.substr(0, pos);

    return new File(parent.c_str());
}

std::vector<std::string> File::getListFiles() {

    DIR *dir;
    dirent* ep;
    errno = 0;

    std::vector<std::string> list_files;

    dir = opendir(m_path.c_str());
    if (dir != NULL)
    {
        //while (true)
        while ((ep = readdir(dir)) != NULL)
        {
            if (!strcmp(ep->d_name, "."))
                continue;

            if (!strcmp(ep->d_name, ".."))
                continue;

            list_files.push_back(std::string(ep->d_name));
        }
        closedir(dir);

        std::sort(list_files.begin(), list_files.end(), compareNat);
    }
    else {
        std::cerr << "Invalid directory: " << m_path << std::endl;
        exit(EXIT_FAILURE);
    }

    return list_files;
}

void File::showListFiles(std::vector<std::string> list)
{
    for(int i=0; i< (int) list.size(); ++i)
        std::cout << list[i] << std::endl;
}

