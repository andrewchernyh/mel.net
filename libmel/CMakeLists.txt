# CMakeList.txt : CMake project for mel.c, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

add_subdirectory(eigen)

# Add source to this project's executable.
include_directories(eigen)
project (mel.c VERSION 1.0.0 DESCRIPTION "Mel feature extractor")


add_library (mel.c  SHARED "melFeatureExtractor.cpp" "utils.h" "utils.cpp" "libmel.cpp"  "windows_entrypoint.c")
add_executable (mel.test  "mel.c.cpp" "melFeatureExtractor.cpp" "utils.h" "utils.cpp" "libmel.cpp")

# TODO: Add tests and install targets if needed.
