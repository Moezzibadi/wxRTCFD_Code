cmake_minimum_required(VERSION 3.5)

project(RTCFD_Code VERSION 0.01 LANGUAGES CXX OBJCXX) # Enable Objective-C++ for Metal

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------- wxWidgets ----------------
find_package(wxWidgets REQUIRED gl core base propgrid OPTIONAL_COMPONENTS net)
include(${wxWidgets_USE_FILE})
add_definitions(-DUSE_WX)

# ---------------- OpenMP ----------------
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message(STATUS "OpenMP found")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# ---------------- LibTorch ----------------
find_package(Torch REQUIRED)
add_definitions(-DUSE_LIBTORCH)

# ---------------- Metal (macOS) ----------------
if(APPLE)
    find_library(METAL_LIBRARY Metal)
    find_library(FOUNDATION_LIBRARY Foundation)
    find_library(QUARTZCORE_LIBRARY QuartzCore)
    message(STATUS "Metal Frameworks found")
endif()
include_directories("${PROJECT_SOURCE_DIR}/include")

set(PROJECT_SOURCES
    include/GUIFrame.h
    include/settings.h
    include/wxRTCFD_Code_App.h
    include/wxRTCFD_Code_Main.h
    src/GUIFrame.cpp
    src/settings.cpp
    src/wxRTCFD_Code_App.cpp
    src/wxRTCFD_Code_Main.cpp
    src/fluid.mm
    include/fluid.h
    src/region.cpp
    include/region.h
    src/draw.cpp
    include/draw.h
)

add_executable(RTCFD_Code ${PROJECT_SOURCES})

# ---------------- Linking ----------------
target_link_libraries(RTCFD_Code
    PRIVATE
        ${wxWidgets_LIBRARIES}
        OpenMP::OpenMP_CXX
        ${TORCH_LIBRARIES}
        ${METAL_LIBRARY}
        ${FOUNDATION_LIBRARY}
        ${QUARTZCORE_LIBRARY}
        tbb
)

target_include_directories(RTCFD_Code PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_options(RTCFD_Code PRIVATE -O3)

set_target_properties(RTCFD_Code PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER sofiane.khelladi.page
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

install(TARGETS RTCFD_Code
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

file(COPY ${CMAKE_SOURCE_DIR}/net-best_traced.pt DESTINATION ${CMAKE_BINARY_DIR})