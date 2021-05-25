cmake_minimum_required(VERSION 3.19)
project(matrix_multi)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -Ofast -march=native -mfma -fopenmp -mavx -fomit-frame-pointer -ffp-contract=fast -flto -DARMA_NO_DEBUG")

# for debugging
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mfma -mavx -fopenmp -fomit-frame-pointer -ffp-contract=fast -flto")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include(FetchContent)

# CXX options library
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG        v2.2.1
)
FetchContent_MakeAvailable(cxxopts)
# Adds cxxopts

# Formatting library
FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG        7.1.2
)
FetchContent_MakeAvailable(fmt)

add_executable(mmjoin main.cpp)
target_include_directories(mmjoin PUBLIC include)
target_include_directories(mmjoin PUBLIC ${cxxopts_SOURCE_DIR}/include)
target_include_directories(mmjoin PUBLIC /opt/intel/mkl/include)

# use libmkl_gnu_thread instead of libmkl_intel_thread
# iomp5 also breaks behavior, prefer just -fopenmp
target_link_libraries(mmjoin
        /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.a
        /opt/intel/mkl/lib/intel64/libmkl_gnu_thread.a
        /opt/intel/mkl/lib/intel64/libmkl_core.a
        pthread m dl)

target_link_libraries(mmjoin fmt::fmt Eigen3::Eigen)