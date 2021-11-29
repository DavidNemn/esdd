project(dvs_mapping)
cmake_minimum_required(VERSION 2.8.3)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

SET(CMAKE_BUILD_TYPE RelWithDebInfo) # Release, RelWithDebInfo
ADD_DEFINITIONS(
    -std=c++11
    -fopenmp
    -w
    -Wfatal-errors
)

set(HEADERS
  include/dvs_depth_from_defocus/depth_defocus_node.hpp
  include/dvs_depth_from_defocus/depth_vector.hpp
)
set(SOURCES
  src/depth_defocus_node.cpp
)
# 把HEADERS、SOURCES添加到dvs_mapping库
cs_add_library(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# 编译出可执行的dvs_mapping_ros
cs_add_executable(dvs_mapping_ros src/dvs_mapping_ros.cpp)
target_link_libraries(dvs_mapping_ros
  ${PROJECT_NAME}
)

cs_install()
cs_export()

