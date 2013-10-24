project(ekf)

add_executable(ekf main.cpp)

install(TARGETS ekf RUNTIME DESTINATION bin)
