cmake_minimum_required(VERSION 3.20)

project(cpp_gnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cpp_gnn
    src/Graph.cpp
    src/Linear.cpp
    src/ReLU.cpp
    src/MLP.cpp
)

target_include_directories(cpp_gnn
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

add_executable(test_relu
    examples/test_relu.cpp
)

target_link_libraries(test_relu
    PRIVATE
        cpp_gnn
)

add_executable(test_mlp
    examples/test_mlp.cpp
)

target_link_libraries(test_mlp
    PRIVATE
        cpp_gnn
)