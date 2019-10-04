cmake_minimum_required (VERSION 3.2)

project (diffcp)

include_directories(external/eigen)
include_directories(include)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DIFFCP_SRC
    "src/cones.cpp"
    "src/deriv.cpp"
    "src/exp.c"
    "src/linop.cpp"
    "src/lsqr.cpp"
)

add_executable (run_example example/example.cpp ${DIFFCP_SRC})
