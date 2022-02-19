cmake_minimum_required(VERSION 3.17)

project(eigen_ext C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PACKAGE_TESTS "Build the tests" ON)

set(eigen_ext_INSTALL_default ON)
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
   set(eigen_ext_INSTALL_default OFF)
endif()
set(eigen_ext_INSTALL ${cppkg_INSTALL_default} CACHE BOOL "Generate installation target")
set(eigen_ext_INSTALL_BINDIR "bin" CACHE STRING "Installation directory for executables")
set(eigen_ext_INSTALL_LIBDIR "lib" CACHE STRING "Installation directory for libraries")
set(eigen_ext_INSTALL_INCLUDEDIR "include" CACHE STRING "Installation directory for headers")
set(eigen_ext_INSTALL_CMAKEDIR "lib/cmake/${PACKAGE_NAME}" CACHE STRING "Installation directory for cmake config files")

#### third_parties ####

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#### local libraries ####

include_directories(
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/.>
    $<INSTALL_INTERFACE:.>
)

add_library(eigen_ext INTERFACE)
target_link_libraries(eigen_ext INTERFACE ${CONAN_LIBS_EIGEN})

#### install ####
if(eigen_ext_INSTALL)
    foreach(_HDR
        EigenExt/sparse_shape.hpp
        EigenExt/sparse_tensor.hpp
        EigenExt/sparse_tensor_assign.hpp
        EigenExt/sparse_tensor_broadcast.hpp
        EigenExt/sparse_tensor_buffer.hpp
        EigenExt/sparse_tensor_cast.hpp
        EigenExt/sparse_tensor_chip.hpp
        EigenExt/sparse_tensor_contract.hpp
        EigenExt/sparse_tensor_convolve.hpp
        EigenExt/sparse_tensor_cwise_binary.hpp
        EigenExt/sparse_tensor_cwise_unary.hpp
        EigenExt/sparse_tensor_dst.hpp
        EigenExt/sparse_tensor_pad.hpp
        EigenExt/sparse_tensor_reduce.hpp
        EigenExt/sparse_tensor_reverse.hpp
        EigenExt/sparse_tensor_scatter.hpp
        EigenExt/sparse_tensor_select.hpp
        EigenExt/sparse_tensor_shuffle.hpp
        EigenExt/sparse_tensor_slice.hpp
        EigenExt/sparse_tensor_src.hpp
        EigenExt/sparse_tensor_stride.hpp
        EigenExt/sparse_tensor_util.hpp
    )
        get_filename_component(_PATH ${_HDR} PATH)
        install(FILES ${_HDR} DESTINATION "${eigen_ext_INSTALL_INCLUDEDIR}/${_PATH}")
    endforeach()

    set(TargetFile ${PROJECT_NAME}Targets)
    install(TARGETS
        eigen_ext
        EXPORT ${TargetFile}
        RUNTIME DESTINATION ${eigen_ext_INSTALL_BINDIR}
        LIBRARY DESTINATION ${eigen_ext_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${eigen_ext_INSTALL_LIBDIR}
    )
    install(EXPORT ${TargetFile}
        DESTINATION ${eigen_ext_INSTALL_CMAKEDIR}
        NAMESPACE ${PROJECT_NAME}::
    )
endif()

#### tests ####

if(PACKAGE_TESTS)
    enable_testing()
endif()

set(EIGEN_EXT_TEST eigen_ext_test)
add_executable(${EIGEN_EXT_TEST}
	EigenExt/test/main.cpp
    EigenExt/test/test_sparse_assign.cpp
    EigenExt/test/test_sparse_broadcast.cpp
    EigenExt/test/test_sparse_buffer.cpp
    EigenExt/test/test_sparse_chip.cpp
    EigenExt/test/test_sparse_contract.cpp
    EigenExt/test/test_sparse_convolve.cpp
    EigenExt/test/test_sparse_cwise_binary.cpp
    EigenExt/test/test_sparse_cwise_unary.cpp
    EigenExt/test/test_sparse_dst.cpp
    EigenExt/test/test_sparse_pad.cpp
    EigenExt/test/test_sparse_reduce.cpp
    EigenExt/test/test_sparse_reverse.cpp
    EigenExt/test/test_sparse_scatter.cpp
    EigenExt/test/test_sparse_select.cpp
    EigenExt/test/test_sparse_shuffle.cpp
    EigenExt/test/test_sparse_slice.cpp
    EigenExt/test/test_sparse_src.cpp
    EigenExt/test/test_sparse_stride.cpp
)
target_link_libraries(${EIGEN_EXT_TEST} ${CONAN_LIBS_CISAB} ${CONAN_LIBS_VERUM} ${CONAN_LIBS_GTEST} eigen_ext)
add_test(NAME ${EIGEN_EXT_TEST} COMMAND ${EIGEN_EXT_TEST})
