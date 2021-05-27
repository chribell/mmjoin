cmake_minimum_required(VERSION 3.19)
project(mmjoin CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# set host compiler to system's g++, CUDA's integrated doesn't work well with fmt
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Ofast -march=native -fopenmp")

# for debugging
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -mfma -mavx -fopenmp -fomit-frame-pointer -ffp-contract=fast -flto")

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


add_executable(mmjoin main.cu)
target_compile_features(mmjoin PRIVATE cxx_std_14)
set_target_properties(
        mmjoin
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(mmjoin PROPERTIES CUDA_ARCHITECTURES "61")

target_include_directories(mmjoin PUBLIC include)
target_include_directories(mmjoin PUBLIC ${cxxopts_SOURCE_DIR}/include)
target_include_directories(mmjoin PUBLIC ${fmt_SOURCE_DIR}/include)

target_link_libraries(mmjoin fmt::fmt -lcublas)
