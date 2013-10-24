cmake_minimum_required(VERSION 2.4)

project(ekf)

include_directories(/usr/include/eigen3 include/)
add_definitions(-std=c++0x)

#the EKF library
add_library(ekf STATIC src/EKF/core.cpp src/EKF/EKF.cpp)

target_link_libraries(ekf boost_date_time)

#install the library
install(TARGETS ekf ARCHIVE DESTINATION lib)
install(DIRECTORY include/EKF DESTINATION include)

## THE TEST ##

add_definitions(-DNDEBUG)
add_executable(ekf_test src/main.cpp src/EKF/core.cpp src/EKF/EKF.cpp)
target_link_libraries(ekf_test boost_date_time)

# install(TARGETS ekf RUNTIME DESTINATION bin)
