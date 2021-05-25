#pragma once

#ifndef DEFS_HPP
#define DEFS_HPP

#include <vector>
#include <algorithm>

typedef std::vector<unsigned int> uint_vector;
typedef std::vector<uint_vector> inverted_index;
typedef std::vector<std::pair<unsigned int, unsigned int>> result_set;
typedef std::pair<unsigned int, unsigned int> degree_pair;
typedef std::pair<unsigned long long, unsigned long long> cdf_pair;

struct set {
    unsigned int id;
    uint_vector elements;
    set(unsigned int id, uint_vector elements) : id(id), elements(std::move(elements)) {}
};

typedef std::vector<set> sets;
typedef unsigned long long ull;
#endif // DEFS_HPP