# CMake generated Testfile for 
# Source directory: /mnt/ssd3/shogi/Hisen-Engine
# Build directory: /mnt/ssd3/shogi/Hisen-Engine/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_kiki "/mnt/ssd3/shogi/Hisen-Engine/build/test_kiki")
set_tests_properties(test_kiki PROPERTIES  _BACKTRACE_TRIPLES "/mnt/ssd3/shogi/Hisen-Engine/CMakeLists.txt;49;add_test;/mnt/ssd3/shogi/Hisen-Engine/CMakeLists.txt;0;")
add_test(test_blocking "/mnt/ssd3/shogi/Hisen-Engine/build/test_blocking")
set_tests_properties(test_blocking PROPERTIES  _BACKTRACE_TRIPLES "/mnt/ssd3/shogi/Hisen-Engine/CMakeLists.txt;49;add_test;/mnt/ssd3/shogi/Hisen-Engine/CMakeLists.txt;0;")
add_test(test_graph "/mnt/ssd3/shogi/Hisen-Engine/build/test_graph")
set_tests_properties(test_graph PROPERTIES  _BACKTRACE_TRIPLES "/mnt/ssd3/shogi/Hisen-Engine/CMakeLists.txt;49;add_test;/mnt/ssd3/shogi/Hisen-Engine/CMakeLists.txt;0;")
add_test(test_mcts "/mnt/ssd3/shogi/Hisen-Engine/build/test_mcts")
set_tests_properties(test_mcts PROPERTIES  _BACKTRACE_TRIPLES "/mnt/ssd3/shogi/Hisen-Engine/CMakeLists.txt;49;add_test;/mnt/ssd3/shogi/Hisen-Engine/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
