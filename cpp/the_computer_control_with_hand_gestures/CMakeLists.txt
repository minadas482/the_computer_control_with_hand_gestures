
# 1. Minimum CMake Version
cmake_minimum_required(VERSION 3.15)

# 2. Project Name
project(the_computer_control_with_hand_gestures)

# 3. Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 4. Find the vcpkg Engines
find_package(OpenCV REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

# 5. Add the OS Controller Submodule
add_subdirectory(externals/robot-cpp)

# 6. Create the Executable Target
add_executable(the_computer_control_with_hand_gestures 
    main.cpp 
    gesture_cnn.cpp
    hand_tracker.cpp
 "gesture_cnn.h"  "hand_tracker.h" "hand_tracker.cpp" "mouse_controller.h")

# --- THE BULLETPROOF FIX FOR ALL LIBRARIES ---
target_include_directories(the_computer_control_with_hand_gestures PRIVATE
    ${OpenCV_INCLUDE_DIRS}                                          
    "C:/Users/johnc/vcpkg/installed/x64-windows/include/opencv4"    
    "C:/Users/johnc/vcpkg/installed/x64-windows/include/onnxruntime" 
    "${CMAKE_SOURCE_DIR}/externals/robot-cpp" 
    "${CMAKE_SOURCE_DIR}/externals/robot-cpp/src"  # <--- Add this line!
)

# 7. Link all external binaries
target_link_libraries(the_computer_control_with_hand_gestures PRIVATE 
    ${OpenCV_LIBS}
    "C:/Users/johnc/vcpkg/installed/x64-windows/lib/onnxruntime.lib"  # <--- THE BULLETPROOF FIX
    RobotCPP
)
