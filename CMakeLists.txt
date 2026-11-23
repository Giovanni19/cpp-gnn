cmake_minimum_required(VERSION 3.20)

project(cpp_gnn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(cpp_gnn
    src/Graph.cpp
    src/Linear.cpp
    src/ReLU.cpp
    src/MLP.cpp
    src/VectorUtils.cpp
    src/MessagePassing.cpp
    src/Loss.cpp
)

target_include_directories(cpp_gnn
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

foreach(example test_relu test_mlp test_node test_message_passing test_seed test_train test_train_gnn test_two_layers karate_club)
    add_executable(${example} examples/${example}.cpp)
    target_link_libraries(${example} PRIVATE cpp_gnn)
endforeach()

enable_testing()
foreach(test test_seed test_train test_train_gnn test_two_layers)
    add_test(NAME ${test} COMMAND ${test})
endforeach()
add_test(NAME karate_club COMMAND karate_club ${PROJECT_SOURCE_DIR}/data/karate_club.txt)
