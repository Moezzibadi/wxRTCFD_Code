cmake_minimum_required(VERSION 3.5)

project(RTCFD_Code VERSION 0.01 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------- wxWidgets ----------------
find_package(wxWidgets REQUIRED gl core base propgrid OPTIONAL_COMPONENTS net)
include(${wxWidgets_USE_FILE})

# ---------------- OpenMP ----------------
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message(STATUS "OpenMP found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# ---------------- LibTorch ----------------
# IMPORTANT: set CMAKE_PREFIX_PATH when configuring
#   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
find_package(Torch REQUIRED)
add_definitions(-DUSE_LIBTORCH)

include_directories("${PROJECT_SOURCE_DIR}/include")

set(PROJECT_SOURCES
    include/GUIFrame.h
    include/settings.h
    include/wxRTCFD_Code_App.h
    include/wxRTCFD_Code_Main.h
    src/GUIFrame.cpp
    src/WxWizFrame.fbp
    src/settings.cpp
    src/wxRTCFD_Code_App.cpp
    src/wxRTCFD_Code_Main.cpp
    src/fluid.cpp
    include/fluid.h
    src/region.cpp
    include/region.h
    src/draw.cpp
    include/draw.h
)

add_executable(RTCFD_Code ${PROJECT_SOURCES})

# # ---------------- Linking ----------------
# target_link_libraries(RTCFD_Code
#     PRIVATE
#         ${wxWidgets_LIBRARIES}
#         OpenMP::OpenMP_CXX
#         Torch::Torch
#         -O3 -fopenmp -ltbb
# )
# ---------------- Linking ----------------
target_link_libraries(RTCFD_Code
    PRIVATE
        ${wxWidgets_LIBRARIES}
        OpenMP::OpenMP_CXX
        ${TORCH_LIBRARIES}
        tbb
)

# Make sure torch headers are visible
target_include_directories(RTCFD_Code PRIVATE ${TORCH_INCLUDE_DIRS})

# Optimization flags
target_compile_options(RTCFD_Code PRIVATE -O3)


# ---------------- Properties ----------------
set_target_properties(RTCFD_Code PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER sofiane.khelladi.page
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

install(TARGETS RTCFD_Code
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# ---------------- Trained Network ----------------
file(COPY ${CMAKE_SOURCE_DIR}/ConstInvel3Res122_5288To122210.pt DESTINATION ${CMAKE_BINARY_DIR})


# From build folder, run:
# cd ..; rm -r build; mkdir build; cd build;
# cmake .. -DCMAKE_BUILD_TYPE=Release \
#   -DCMAKE_C_COMPILER=/usr/local/opt/llvm/bin/clang \
#   -DCMAKE_CXX_COMPILER=/usr/local/opt/llvm/bin/clang++ \
#   -DCMAKE_PREFIX_PATH=/Users/mohammadmoezzibadi/libtorch
# make -j