project(dvs_bootstrapping)
cmake_minimum_required(VERSION 3.0.2)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

find_package(Boost REQUIRED COMPONENTS system)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

SET(CMAKE_BUILD_TYPE RelWithDebInfo) # Release, RelWithDebInfo
ADD_DEFINITIONS(
    -std=c++11
    -Wno-ignored-qualifiers
    -w
)

################################################################################

set(HEADERS_ef
    include/dvs_bootstrapping/Bootstrapper.hpp
    include/dvs_bootstrapping/EventsFramesBootstrapper.hpp
    include/motion_correction.hpp
)
set(SOURCES_ef
    src/Bootstrapper.cpp
    src/EventsFramesBootstrapper.cpp
    src/motion_correction.cpp
)

cs_add_library(${PROJECT_NAME}_ef ${SOURCES_ef} ${HEADERS_ef})

################################################################################

cs_add_executable(dvs_bootstrapping_ef_ros src/dvs_bootstrapping_ef_ros.cpp)
target_link_libraries(dvs_bootstrapping_ef_ros 
  ${PROJECT_NAME}_ef
  ${OpenCV_LIBS}
  ${Boost_INCLUDE_DIRS}
)

################################################################################
cs_install()
cs_export()