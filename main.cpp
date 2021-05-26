#include <cmath>
#include <numeric>
#include <vector>
#include <string>
#include <set>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <sstream>
#include <omp.h>
#define EIGEN_USE_MKL_ALL
#include "Eigen/Dense"

#include <cxxopts.hpp>
#include <fmt/core.h>
//#include <fmt/ranges.h>
#include "timer.hpp"
#include "input.hpp"

std::string formatBytes(ull bytes)
{
    ull gb = 1073741824;
    int mb = 1048576;
    int kb = 1024;

    std::stringstream stream;
    stream << std::fixed << std::setprecision(3);

    if (bytes >= gb) {
        stream << ((double) bytes / (double) gb) << " GB";
    } else if (bytes >= mb) {
        stream << ((double) bytes / (double) mb) << " MB";
    } else if (bytes >= kb) {
        stream << ((double) bytes / (double) kb) << " KB";
    } else {
        stream << bytes << " bytes";
    }

    return stream.str();
}


degree_pair optimize(sets& collection, double eps, unsigned int iterations,
                     ull relationSize, ull fullJoinSize, unsigned int relationFactor,
                     std::vector<degree_pair>& degreeSet, std::vector<degree_pair>& degreeElement,
                     std::vector<cdf_pair>& setCDF, std::vector<cdf_pair>& elementCDF)
{
    if (fullJoinSize <= relationFactor * relationSize) {
        return std::make_pair(0, INT32_MAX);
    }

    double outputSizeEstimate = (sqrt(fullJoinSize) / relationSize) * fullJoinSize;

    unsigned int delta1 = collection.size();
    unsigned int delta2 = relationSize * (delta1 / outputSizeEstimate);

    ull tHeavy = 0;
    ull prevHeavy = 0;
    ull tLight = fullJoinSize / 1000;
    ull prevLight = LONG_LONG_MAX;
    ull prevDelta1 = 0;
    ull prevDelta2 = 0;

    unsigned int heavySets = 0;
    unsigned int heavyElements = 0;
    unsigned int indexSet = 0;
    unsigned int indexElement = 0;

    unsigned long i = 0;
    while (i < iterations) {
        prevLight = tLight;
        prevHeavy = tHeavy;
        prevDelta1 = delta1;
        prevDelta2 = delta2;

        delta1 = delta1 * eps;
        delta2 = relationSize * delta1 / outputSizeEstimate;
        auto x = std::lower_bound(degreeSet.begin(), degreeSet.end(), std::make_pair(delta2, (unsigned int) 0));
        indexSet = x - degreeSet.begin();
        heavySets = degreeSet.size() - indexSet;
        x = std::lower_bound(degreeElement.begin(), degreeElement.end(), std::make_pair(delta1, (unsigned int) 0));
        indexElement = x - degreeElement.begin();
        heavyElements = degreeElement.size() - indexElement;

        tLight = setCDF[indexSet].second / 1000.0 + elementCDF[indexElement].second / 1000.0;
        tHeavy = heavySets * heavyElements * heavySets / 10000.0;

        if ((prevLight + prevHeavy) <= (tLight + tHeavy) && prevHeavy > 0) {
            return std::make_pair(prevDelta1, prevDelta2);
        }
        i++;
    }

    return std::make_pair(prevDelta1, prevDelta2);
}


int main(int argc, char** argv)
{
    try {
        cxxopts::Options options(argv[0], "Fast SSJ/SCJ Project Query Evaluation using Matrix Multiplication");

        bool scj = false;
        unsigned int c = 1;
        int threads = 1;
        double eps = 0.95;
        unsigned int rf = 20;
        unsigned int iterations = INT32_MAX; // substitute for while (true)
        unsigned int deltaSet;
        unsigned int deltaElement;
        bool overrideOptimizer = false;

        options.add_options()
                ("input", "Input dataset file", cxxopts::value<std::string>())
                ("c", "Overlap/Common elements (for set similarity join)", cxxopts::value<unsigned int>(c))
                ("scj", "Set containment join (ignores c)", cxxopts::value<bool>(scj))
                ("iter", "Number of iterations for the cost-based optimizer", cxxopts::value<unsigned int>(iterations))
                ("rf", "The relation factor for the cost-based optimizer", cxxopts::value<unsigned int>(rf))
                ("eps", "The epsilon factor for the cost-based optimizer", cxxopts::value<double>(eps))
                ("threads", "Number of parallel threads", cxxopts::value<int>(threads))
                ("delta-set", "Delta Set (Set cutoff degree)", cxxopts::value<unsigned int>(deltaSet))
                ("delta-element", "Delta Element (Element cutoff degree)", cxxopts::value<unsigned int>(deltaElement))
                ("help", "Print help");

        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            fmt::print("{}\n", options.help());
            return 0;
        }

        if (!result.count("input")) {
            fmt::print("ERROR: No input dataset given! Exiting...\n");
            return 1;
        }

        if (result.count("delta-set") || result.count("delta-element")){ // if either is given
            if (result.count("delta-set") && result.count("delta-element")) { // we must ensure that both deltas are given
                overrideOptimizer = true;
            } else { // otherwise, show error and exit
                fmt::print("ERROR: Both delta-set and delta-element are required! Exiting...\n");
                return 1;
            }
        }

        timer t;

        std::string inputPath = result["input"].as<std::string>();
        unsigned int universe = 0;

        timer::Interval* readInput = t.add("Read input");
        sets collection = readSets(inputPath, universe);
        timer::finish(readInput);

        inverted_index index(universe);
        std::vector<degree_pair> degreeSet;
        std::vector<cdf_pair> setCDF;
        std::vector<degree_pair> degreeElement;
        std::vector<cdf_pair> elementCDF;
        ull relationSize = 0; // aka total number of elements for input relation
        ull fullJoinSize = 0;

        timer::Interval* constructIndex = t.add("Construct Index");
        for (auto& s : collection) {
            if (!scj && s.elements.size() < c) continue;
            for (auto& el : s.elements) {
                index[el].push_back(s.id);
            }
            relationSize += s.elements.size();
            degreeSet.push_back(std::make_pair(s.elements.size(), s.id));
        }
        timer::finish(constructIndex);

        timer::Interval* constructArrays = t.add("Construct arrays");
        unsigned int invListSize = 0;
        for (auto& s : collection) {
            for (auto& el : s.elements) {
                invListSize += index[el].size();
            }
            setCDF.push_back(std::make_pair(s.elements.size(), invListSize));
            invListSize = 0;
        }

        unsigned int element = 0;
        for (auto& invList : index) {
            elementCDF.push_back(std::make_pair(invList.size(), invList.size())); // this only for the self join scenario
            degreeElement.push_back(std::make_pair(invList.size(), element++));
            fullJoinSize += invList.size() * invList.size();
        }
        timer::finish(constructArrays);

        timer::Interval* sortArrays = t.add("Sort arrays");
        sort(degreeSet.begin(), degreeSet.end(), [] (const degree_pair& a, degree_pair& b) {
            return a.first < b.first;
        });
        sort(degreeElement.begin(), degreeElement.end(), [] (const degree_pair& a, degree_pair& b) {
            return a.first < b.first;
        });

        sort(setCDF.begin(), setCDF.end(), [] (const cdf_pair& a, const cdf_pair& b) {
            return a.first < b.first;
        });
        sort(elementCDF.begin(), elementCDF.end(), [] (const cdf_pair& a, const cdf_pair& b) {
            return a.first < b.first;
        });
        timer::finish(sortArrays);


        timer::Interval* constructCDF = t.add("Construct CDF");
        for (int i = 0 ; i < elementCDF.size(); i++) {
            if (i == 0)
                elementCDF.at(i).second = elementCDF.at(i).second * elementCDF.at(i).second;
            if (i > 0)
                elementCDF.at(i).second = elementCDF.at(i - 1).second + elementCDF.at(i).second * elementCDF.at(i).second;
        }

        for (int i = 0 ; i < setCDF.size(); i++) {
            if (i > 0)
                setCDF.at(i).second += setCDF.at(i - 1).second;
        }
        timer::finish(constructCDF);

        if (!overrideOptimizer) {
            timer::Interval* costBasedOptimize = t.add("Cost-based optimize");
            degree_pair opt = optimize(collection, eps, iterations, relationSize, fullJoinSize, rf,
                                       degreeSet, degreeElement, setCDF, elementCDF);
            deltaSet = opt.second;
            deltaElement = opt.first;
            timer::finish(costBasedOptimize);
        }

        // find bounds for light/heavy sets
        // x is the cut-off point to distinguish light from heavy sets
        auto x = std::lower_bound(degreeSet.begin(), degreeSet.end(), std::make_pair(deltaSet, (unsigned int) 0));
        // extra verbosity for ease
        unsigned int lightSetHigh = x - degreeSet.begin();
        unsigned int lightSets = lightSetHigh;
        unsigned int heavySetLow = lightSetHigh;
        unsigned int heavySetHigh = degreeSet.size();
        unsigned int heavySets = heavySetHigh - heavySetLow;

        // find the number of light/heavy elements respectively
        x = std::lower_bound(degreeElement.begin(), degreeElement.end(), std::make_pair(deltaElement, (unsigned int) 0));
        unsigned int lightElements = x - degreeElement.begin();
        unsigned int heavyElements = degreeElement.size() - lightElements;

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "│{9: ^{2}}|{10: ^{2}}│\n"
                "│{11: ^{2}}|{12: ^{2}}│\n"
                "│{13: ^{2}}|{14: ^{2}}│\n"
                "│{15: ^{2}}|{16: ^{2}}│\n"
                "│{17: ^{2}}|{18: ^{2}}│\n"
                "│{19: ^{2}}|{20: ^{2}}│\n"
                "│{21: ^{2}}|{22: ^{2}}│\n"
                "│{23: ^{2}}|{24: ^{2}}│\n"
                "│{25: ^{2}}|{26: ^{2}}│\n"
                "│{27: ^{2}}|{28: ^{2}}│\n"
                "│{29: ^{2}}|{30: ^{2}}│\n"
                "│{31: ^{2}}|{32: ^{2}}│\n"
                "│{33: ^{2}}|{34: ^{2}}│\n"
                "└{35:─^{1}}┘\n", "Info", 51, 25,
                "Type of join", scj ? "Set containment join" : "Set similarity join",
                "Number of threads", threads,
                "Overlap (c)", scj ? "-" : std::to_string(c),
                "Number of sets", collection.size(),
                "Universe size", universe,
                "Relation size", relationSize,
                "Full Join Size", fullJoinSize,
                "Split", overrideOptimizer ? "Manually" : "Cost-based",
                "Delta set", deltaSet,
                "Delta element", deltaElement,
                "Light sets (count)", lightSets,
                "Light sets (range)", lightSets > 0 ? "[0 - " + std::to_string(lightSetHigh > 0 ? lightSetHigh - 1 : 0) + "]" : "-",
                "Heavy sets (count)", heavySets,
                "Heavy sets (range)", heavySets > 0 ? "[" + std::to_string(heavySetLow) + " - " + std::to_string(heavySetHigh) + "]" : "-",
                "Light elements (count)", lightElements,
                "Heavy elements (count)", heavyElements,
                ""
        );

        ull indexBytes = sizeof(unsigned int) * relationSize;
        ull mmBytes = (sizeof(float) * heavySets * heavyElements * 2) + (sizeof(float) * heavySets * heavySets);

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "└{9:─^{1}}┘\n", "Memory Requirements", 51, 25,
                "Raw dataset", formatBytes(indexBytes),
                "Index", formatBytes(indexBytes),
                "Matrix multiplication", formatBytes(mmBytes), "");

        uint_vector counts(threads, 0);
        omp_set_num_threads(threads);

        timer::Interval* indexBasedLightJoin = t.add("Index-based join (light)");
        #pragma omp parallel
        {
            int threadNumber = omp_get_thread_num();
            uint_vector joinVector(collection.size());

            // calculate thread bounds
            unsigned int lower = lightSets * threadNumber / threads;
            unsigned int upper = lightSets * (threadNumber + 1) / threads;

            // debug
            // fmt::print("Light sets | Thread {}: [ {} - {} )\n", threadNumber, lower, upper);

            unsigned int counter = 0;

            for (unsigned int i = lower; i < upper; ++i) {
                auto& probe = collection[i].elements;
                unsigned int offset = i + 1;

                std::fill(joinVector.begin() + offset, joinVector.end(), 0);
                for (auto& el : probe) {
                    auto& list = index[el];
                    for (auto& set : list) {
                        if (set > i) {
                            joinVector[set]++;
                        }
                    }
                }

                if (scj) {
                    c = probe.size();
                }

                counter += std::count_if(joinVector.begin() + offset, joinVector.end(), [c](unsigned int intersection) {
                    return intersection >= c;
                });
            }

            counts[threadNumber] = counter;
        }
        timer::finish(indexBasedLightJoin);

        timer::Interval* matrixMultiplication = t.add("Matrix multiplication");
        std::unordered_map<unsigned int, unsigned int> heavyElementMap(heavyElements);
        unsigned int columnIndex = 0;
        for (unsigned int i = lightElements; i < degreeElement.size(); ++i) {
            heavyElementMap[degreeElement[i].second] = columnIndex++;
        }

        Eigen::MatrixXf A;

        A.resize(heavySets, heavyElements);
        A.setZero();

        unsigned int idx = 0; // used to determine column index
        for (unsigned int i = heavySetLow; i < heavySetHigh; ++i) {
            for (auto& el : collection[i].elements) {
                if (index[el].size() >= deltaElement) {
                    A.coeffRef(idx, heavyElementMap[el]) = 1.0f;
                }
            }
            idx++;
        }

        Eigen::MatrixXf B;
        B.noalias() = A * A.transpose(); // self-join
        timer::finish(matrixMultiplication);

        timer::Interval* indexBasedHeavyJoin = t.add("Index-based join (heavy)");
        #pragma omp parallel
        {
            int threadNumber = omp_get_thread_num();
            uint_vector joinVector(heavySets);

            // calculate thread bounds
            unsigned int lower = heavySetLow + (heavySets * threadNumber / threads);
            unsigned int upper = heavySetLow + (heavySets * (threadNumber + 1) / threads);

            // debug
            // fmt::print("Heavy sets | Thread {}: [ {} - {} )\n", threadNumber, lower, upper);

            unsigned int counter = 0;

            for (unsigned int i = lower; i < upper; ++i) {
                auto& probe = collection[i].elements;

                unsigned int rowIndex = i - heavySetLow;
                unsigned int colStart = rowIndex + 1;

                std::fill(joinVector.begin() + colStart, joinVector.end(), 0);

                for (auto& el : probe) {
                    auto& list = index[el];
                    if (list.size() < deltaElement) { // for the light tokens of the heavy set, use index
                        for (auto& set : list) {
                            if (set > i) {
                                joinVector[set - heavySetLow]++;
                            }
                        }
                    }
                }

                // add intersections from matrix multiplication
                for (unsigned int colIndex = colStart; colIndex < heavyElements + 1; ++colIndex) {
                    joinVector[colIndex] += (unsigned int) B.coeff(rowIndex, colIndex);
                }

                if (scj) {
                    c = probe.size();
                }

                counter += std::count_if(joinVector.begin() + colStart, joinVector.end(), [c](unsigned int intersection) {
                    return intersection >= c;
                });
            }

            counts[threadNumber] += counter;
        }

        timer::finish(indexBasedHeavyJoin);

        t.print();

        // debug
        // fmt::print("{}\n", counts);

        fmt::print("┌{0:─^{1}}┐\n"
                   "|{2: ^{1}}|\n"
                   "└{3:─^{1}}┘\n", "Result", 51, std::accumulate(counts.begin(), counts.end(), (unsigned int) 0), "");


        return 0;
    } catch (const cxxopts::OptionException& e) {
        fmt::print("Error parsing options: {}\n", e.what());
        return 1;
    }
}
