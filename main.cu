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
#include <cublas_v2.h>
#include <cxxopts.hpp>
#include <thrust/device_ptr.h>
#include <fmt/core.h>
//#include <fmt/ranges.h>
#include "timer.hpp"
#include "input.hpp"

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


std::string formatBytes(size_t bytes)
{
    size_t gb = 1073741824;
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


void transpose(float* out, const float* in, int rows, int cols) {
    unsigned int c = 0;
    for (int i = 0; i < rows; ++i){
        for (int j = 0; j < cols; ++j) {
            out[c++] = in[(j * rows + i)];
        }
    }
}

void copyTile(float* out, const float* in, unsigned int rows, unsigned int cols, unsigned int offset)
{
    size_t c = 0;
    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            out[j] = in[c++];
        }
        out += offset;
    }
}

degree_pair optimize(sets& collection, double eps, unsigned int iterations,
                     size_t relationSize, size_t fullJoinSize, unsigned int relationFactor,
                     std::vector<degree_pair>& degreeSet, std::vector<degree_pair>& degreeElement,
                     std::vector<cdf_pair>& setCDF, std::vector<cdf_pair>& elementCDF)
{
    if (fullJoinSize <= relationFactor * relationSize) {
        return std::make_pair(0, INT32_MAX);
    }

    double outputSizeEstimate = (sqrt(fullJoinSize) / relationSize) * fullJoinSize;

    unsigned int delta1 = collection.size();
    unsigned int delta2 = relationSize * (delta1 / outputSizeEstimate);

    size_t tHeavy = 0;
    size_t prevHeavy = 0;
    size_t tLight = fullJoinSize / 1000;
    size_t prevLight = LONG_LONG_MAX;
    size_t prevDelta1 = 0;
    size_t prevDelta2 = 0;

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

        host_timer hostTimer;
        device_timer deviceTimer;

        std::string inputPath = result["input"].as<std::string>();
        unsigned int universe = 0;

        host_timer::Interval* readInput = hostTimer.add("Read input");
        sets collection = readSets(inputPath, universe);
        host_timer::finish(readInput);

        inverted_index index(universe);
        std::vector<degree_pair> degreeSet;
        std::vector<cdf_pair> setCDF;
        std::vector<degree_pair> degreeElement;
        std::vector<cdf_pair> elementCDF;
        size_t relationSize = 0; // aka total number of elements for input relation
        size_t fullJoinSize = 0;

        host_timer::Interval* constructIndex = hostTimer.add("Construct Index");
        for (auto& s : collection) {
            if (!scj && s.elements.size() < c) continue;
            for (auto& el : s.elements) {
                index[el].push_back(s.id);
            }
            relationSize += s.elements.size();
            degreeSet.push_back(std::make_pair(s.elements.size(), s.id));
        }
        host_timer::finish(constructIndex);

        host_timer::Interval* constructArrays = hostTimer.add("Construct arrays");
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
        host_timer::finish(constructArrays);

        host_timer::Interval* sortArrays = hostTimer.add("Sort arrays");
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
        host_timer::finish(sortArrays);


        host_timer::Interval* constructCDF = hostTimer.add("Construct CDF");
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
        host_timer::finish(constructCDF);

        if (!overrideOptimizer) {
            host_timer::Interval* costBasedOptimize = hostTimer.add("Cost-based optimize");
            degree_pair opt = optimize(collection, eps, iterations, relationSize, fullJoinSize, rf,
                                       degreeSet, degreeElement, setCDF, elementCDF);
            deltaSet = opt.second;
            deltaElement = opt.first;
            host_timer::finish(costBasedOptimize);
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

        size_t indexBytes = sizeof(unsigned int) * relationSize;

        size_t mmBytes = (sizeof(float) * heavySets * heavyElements * 2) + (sizeof(float) * heavySets * heavySets);
        size_t freeDeviceMemory, totalDeviceMemory;

        cudaMemGetInfo(&freeDeviceMemory, &totalDeviceMemory);

        // subtract 500MB from free GPU memory
        freeDeviceMemory -= 500 * 1048576;

        fmt::print(
                "┌{0:─^{1}}┐\n"
                "│{3: ^{2}}|{4: ^{2}}│\n"
                "│{5: ^{2}}|{6: ^{2}}│\n"
                "│{7: ^{2}}|{8: ^{2}}│\n"
                "│{9: ^{2}}|{10: ^{2}}│\n"
                "└{11:─^{1}}┘\n", "Memory Requirements", 51, 25,
                "Raw dataset", formatBytes(indexBytes),
                "Index", formatBytes(indexBytes),
                "Free GPU memory", formatBytes(freeDeviceMemory),
                "Matrix multiplication", formatBytes(mmBytes), "");

        uint_vector counts(threads, 0);
        omp_set_num_threads(threads);

        host_timer::Interval* indexBasedLightJoin = hostTimer.add("Index-based join (light)");
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
        host_timer::finish(indexBasedLightJoin);

        if (heavySets > 0) {
            host_timer::Interval* matrixMultiplication = hostTimer.add("Matrix multiplication");
            std::unordered_map<unsigned int, unsigned int> heavyElementMap(heavyElements);
            unsigned int columnIndex = 0;
            for (unsigned int i = lightElements; i < degreeElement.size(); ++i) {
                heavyElementMap[degreeElement[i].second] = columnIndex++;
            }

            float* hostRawInput;
            float* hostInput;
            float* hostInvInput;
            float* hostOutput;

            cudaCheck(cudaMallocHost((void**)&hostRawInput, heavySets * heavyElements * sizeof(float)))
            cudaCheck(cudaMallocHost((void**)&hostInput, heavySets * heavyElements * sizeof(float)))
            cudaCheck(cudaMallocHost((void**)&hostInvInput, heavySets * heavyElements * sizeof(float)))
            cudaCheck(cudaMallocHost((void**)&hostOutput, heavySets * heavySets * sizeof(float)))

            unsigned int idx = 0; // used to determine column index
            for (unsigned int i = heavySetLow; i < heavySetHigh; ++i) {
                for (auto& el : collection[i].elements) {
                    if (index[el].size() >= deltaElement) {
                        hostRawInput[idx * heavyElements + heavyElementMap[el]] = 1.0f;
                    }
                }
                idx++;
            }

            cublasHandle_t handle;
            cublasCreate_v2(&handle);

            float* devInput;
            float* devInvInput;
            float* devOutput;

            // allocate GPU memory
            device_timer::EventPair* allocInput = deviceTimer.add("Allocate memory");
            cudaCheck(cudaMalloc((void**) &devInput, heavySets * heavyElements * sizeof(float)))
            cudaCheck(cudaMalloc((void**) &devInvInput, heavySets * heavyElements * sizeof(float)))
            device_timer::finish(allocInput);

            // check if tiling is required, this is determined by the available GPU memory
            // if the complete output join matrix fits in GPU memory we call MM only once,
            // otherwise, we need tiling to build the output join matrix incrementally
            // in addition, tiling also affects the way we store and access the input matrices
            if (mmBytes > freeDeviceMemory) { // tiling to produce the join matrix

                // subtract the required input matrices sizes (in bytes)
                // in order to find the max number that a tile can support
                freeDeviceMemory -= (sizeof(float) * heavySets * heavyElements * 2);

                unsigned int maxCells = freeDeviceMemory / sizeof(float);
                unsigned int tileSets = std::sqrt((float) maxCells);
                unsigned int tileCells = tileSets * tileSets;

                unsigned int tileElements = tileSets >= heavyElements ? heavyElements : (double) heavyElements / (double) tileSets;
                unsigned int tilesX = std::ceil((double) heavySets / (double) tileSets);
                unsigned int tilesY = std::ceil((double) heavyElements / (double) tileElements);
                std::vector<tile> tiles;

                c = 0;
                for (unsigned int i = 0; i < tilesX; ++i) {
                    for (unsigned int j = 0; j < tilesY; ++j) {
                        unsigned int offset = (i * tileSets * heavyElements) + tileElements * j;

                        unsigned int row = 0;
                        unsigned int col = 0;
                        unsigned int start = c;
                        while (row < tileSets && offset < heavySets * heavyElements) {
                            unsigned int cols = (j + 1) * tileElements > heavyElements ? (j + 1) * tileElements - heavyElements - 1
                                                                                       : tileElements;
                            for (col = 0; col < cols; ++col) {
                                hostInput[c++] = hostRawInput[offset + col];
                            }
                            row++;
                            offset += heavyElements;
                        }
                        transpose(hostInvInput + start, hostInput + start, col, row);
                        tiles.push_back(std::make_tuple(row, col, c - (row * col)));
                    }
                }

                // redundant since we use cudaMemcpy2D
                //float* hostBlock;
                //cudaCheck(cudaMallocHost((void**)&hostBlock, tileCells * sizeof(float)))


                device_timer::EventPair* transferInput = deviceTimer.add("Transfer data");
                cudaCheck(cudaMemcpy(devInput, hostInput, heavySets * heavyElements * sizeof(float), cudaMemcpyHostToDevice))
                cudaCheck(cudaMemcpy(devInvInput, hostInvInput, heavySets * heavyElements * sizeof(float), cudaMemcpyHostToDevice))
                device_timer::finish(transferInput);

                device_timer::EventPair* allocOutput = deviceTimer.add("Allocate memory");
                cudaCheck(cudaMalloc((void**) &devOutput, tileCells * sizeof(float)))
                device_timer::finish(allocOutput);

                float alpha = 1.0;
                float beta = 1.0;

                for (unsigned int i = 0; i < tilesX; ++i) {
                    for (unsigned int j = 0; j < tilesX; ++j) {

                        unsigned int cRows = 0;
                        unsigned int cCols = 0;

                        if ((i * tilesY) > (j * tilesY)) continue;

                        for (unsigned int k = 0; k < tilesY; ++k) {

                            tile& tileA = tiles[(i * tilesY) + k]; // sets x elements
                            tile& tileB = tiles[(j * tilesY) + k]; // elements x sets (transposed)

                            unsigned int aRows = std::get<0>(tileA);
                            unsigned int aCols = std::get<1>(tileA);
                            unsigned int aOffset = std::get<2>(tileA);

                            unsigned int bRows = std::get<1>(tileB);
                            unsigned int bCols = std::get<0>(tileB);
                            unsigned int bOffset = std::get<2>(tileB);

                            cRows = aRows;
                            cCols = bCols;

                            device_timer::EventPair* mm = deviceTimer.add("Matrix multiplication");
                            auto status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        bCols, aRows, aCols,
                                        &alpha, devInvInput + bOffset, bCols,
                                        devInput + aOffset, aCols,
                                        &beta, devOutput, cCols);
                            device_timer::finish(mm);
                        }
                        device_timer::EventPair* transferOutput = deviceTimer.add("Transfer data");
                        cudaCheck(cudaMemcpy2D(hostOutput + ((i * tileSets * heavySets) + tileSets * j),
                                               heavySets * sizeof(float),
                                               devOutput,
                                               cCols * sizeof(float), cCols * sizeof(float), cRows, cudaMemcpyDeviceToHost))
                        // cudaCheck(cudaMemcpy(hostBlock, devOutput, cRows * cCols * sizeof(float), cudaMemcpyDeviceToHost))
                        device_timer::finish(transferOutput);

                        // redundant since we use cudaMemcpy2D
                        // copyTile(hostOutput + ((i * tileSets * heavySets) + tileSets * j), hostBlock, cRows, cCols, heavySets);

                        device_timer::EventPair* clearMem = deviceTimer.add("Clear memory");
                        cudaMemset(devOutput, 0, cRows * cCols * sizeof(float));
                        device_timer::finish(clearMem);
                    }
                }
            } else { // single MM to produce the join matrix
                transpose(hostInvInput, hostRawInput, heavyElements, heavySets);

                device_timer::EventPair* transferInput = deviceTimer.add("Transfer data");
                cudaCheck(cudaMemcpy(devInput, hostRawInput, heavySets * heavyElements * sizeof(float), cudaMemcpyHostToDevice))
                cudaCheck(cudaMemcpy(devInvInput, hostInvInput, heavyElements * heavySets * sizeof(float), cudaMemcpyHostToDevice))
                device_timer::finish(transferInput);

                float alpha = 1.0;
                float beta = 0.0;

                device_timer::EventPair* allocOutput = deviceTimer.add("Allocate memory");
                cudaCheck(cudaMalloc((void**) &devOutput, heavySets * heavySets * sizeof(float)))
                device_timer::finish(allocOutput);

                device_timer::EventPair* mm = deviceTimer.add("Matrix multiplication");
                // https://peterwittek.com/cublas-matrix-c-style.html
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                            heavySets, heavySets, heavyElements,
                            &alpha, devInvInput, heavySets,
                            devInput, heavyElements,
                            &beta, devOutput, heavySets);
                device_timer::finish(mm);

                device_timer::EventPair* transferOutput = deviceTimer.add("Transfer data");
                cudaCheck(cudaMemcpy(hostOutput, devOutput, heavySets * heavySets * sizeof(float), cudaMemcpyDeviceToHost))
                device_timer::finish(transferOutput);
            }

            device_timer::EventPair* freeMem = deviceTimer.add("Free memory");
            cudaFree(devInput);
            cudaFree(devInvInput);
            cudaFree(devOutput);
            device_timer::finish(freeMem);

            cublasDestroy_v2(handle);

            host_timer::finish(matrixMultiplication);

            host_timer::Interval* indexBasedHeavyJoin = hostTimer.add("Index-based join (heavy)");
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
                    for (unsigned int colIndex = colStart; colIndex < heavySets; ++colIndex) {
                        joinVector[colIndex] += (unsigned int) hostOutput[rowIndex * heavySets + colIndex];
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
            host_timer::finish(indexBasedHeavyJoin);
        }

        hostTimer.print();
        deviceTimer.print();

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
