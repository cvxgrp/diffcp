cmake_minimum_required (VERSION 3.2)

project (diffcp)

include_directories(external/eigen)
include_directories(include)

set(diffcp_src
    src/cones.cpp
    src/deriv.cpp
    src/exp.c
    src/linop.cpp
    src/lsqr.cpp
)

add_executable (example example/example.cpp src)
