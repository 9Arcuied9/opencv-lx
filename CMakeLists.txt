cmake_minimum_required(VERSION 3.16)
project(vision_math_demo LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Ceres REQUIRED)

# 四个任务的可执行文件（源码都在 src/<task>/ 下）
add_executable(task1_image    src/task1_image/main.cpp)
add_executable(task2_fit      src/task2_fit/main.cpp)
add_executable(task3_windmill src/task3_windmill/main.cpp)
add_executable(task4_windmill src/task3_windmill/main2.cpp)

foreach(t task1_image task2_fit task3_windmill task4_windmill)
    target_include_directories(${t} PRIVATE ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
    target_link_libraries(${t} PRIVATE ${OpenCV_LIBS})
endforeach()
target_link_libraries(task2_fit PRIVATE Ceres::ceres)