#pragma once

#ifndef INPUT_HPP
#define INPUT_HPP

#include "defs.hpp"
#include <fstream>
#include <string>
#include <stdexcept>

uint_vector split(const std::string& s, char delimiter) {
    std::stringstream ss(s);
    std::string item;
    uint_vector elements;
    while (std::getline(ss, item, delimiter)) {
        elements.push_back(atoi(item.c_str()));
    }
    return elements;
}

sets readSets(const std::string& filename, unsigned int& universe)
{
    sets collection;
    std::ifstream infile;
    std::string line;
    infile.open(filename.c_str());

    if (!infile) {
        throw std::invalid_argument("Wrong input file!");
    }

    unsigned int id = 0; // set ids must start from 0
    while (!infile.eof()) {
        std::getline(infile, line);
        if (line.empty()) continue;

        uint_vector elements = split(line, ' ');
        collection.emplace_back(set(id++, elements));

        if (elements.back() > universe) {
            universe = elements.back();
        }
    }
    infile.close();
    universe++;
    return collection;
}


#endif // INPUT_HPP