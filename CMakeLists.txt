cmake_minimum_required(VERSION 3.5)
project(DSN-MD)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(MPI REQUIRED)

add_library("MD_Engine" STATIC
        "MD_Engine/pmd.cpp"
        "MD_Engine/pmd.hpp"
        )

add_library("Pingu" STATIC
        "Source/pingu.cpp"
        "Source/pingu.hpp"
        )
target_link_libraries(Pingu "${TORCH_LIBRARIES}")

include_directories(${MPI_INCLUDE_PATH})

add_executable(pingu_example Example/scratch_pad.cpp Example/scratch_pad.hpp)
target_link_libraries(pingu_example "${TORCH_LIBRARIES}")
target_link_libraries(pingu_example MD_Engine)
target_link_libraries(pingu_example Pingu)
set_property(TARGET pingu_example PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET example-app
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            $<TARGET_FILE_DIR:example-app>)
endif (MSVC)

target_link_libraries(pingu_example ${MPI_LIBRARIES})