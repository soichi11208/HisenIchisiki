cmake_minimum_required(VERSION 3.18)
project(hisen_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# libtorch: pip の torch を流用する場合
#   cmake -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)') ..
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O2")

# --- core ライブラリ (対局エンジン本体のみ) ---
add_library(hisen_core
  src/engine/board.cpp
  src/engine/kiki.cpp
  src/engine/movegen.cpp
  src/engine/blocking.cpp
  src/engine/mate.cpp
  src/graph/builder.cpp
  src/model/gatv2.cpp
  src/model/hisen.cpp
  src/mcts/search.cpp
)
target_link_libraries(hisen_core PUBLIC ${TORCH_LIBRARIES})
find_package(Threads REQUIRED)
target_link_libraries(hisen_core PUBLIC Threads::Threads)

# --- 対局バイナリ (USI) ---
add_executable(hisen-usi src/usi/main.cpp)
target_link_libraries(hisen-usi PRIVATE hisen_core)

# --- テスト (gtest は FetchContent) ---
option(HISEN_BUILD_TESTS "Build tests" ON)
if(HISEN_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  foreach(t IN ITEMS test_kiki test_blocking test_graph test_mcts)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE hisen_core gtest_main)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endif()
