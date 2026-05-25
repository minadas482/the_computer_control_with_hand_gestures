
# AYNIYH: System-Level Hand Gesture Control

![C++](https://img.shields.io/badge/C++-17-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.14-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)
![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)

**AYNIYH** is a high-performance, contactless mouse controller that uses computer vision to track hand gestures via a standard RGB webcam. 

While most academic hand-tracking prototypes are written in Python, they often suffer from input lag and frame drops due to the Global Interpreter Lock (GIL) and garbage collection overhead. AYNIYH solves this "latency gap" by moving the entire inference pipeline to **C++**. 

## The Performance Advantage
We extract 3D hand landmarks using MediaPipe, normalize the coordinates, and feed them into a lightweight 1D Convolutional Neural Network (CNN) deployed via the **ONNX Runtime engine**. 

By running natively in C++, the system drops average neural network inference latency from **0.23 ms (Python)** to **0.14 ms (C++)**—an almost 40% improvement that ensures jitter-free OS control and leaves your CPU free for heavy workloads.

### Latency Comparison (10,000 Frame Telemetry)
| Execution Environment | Mean Latency | Median Latency | Standard Deviation |
| :--- | :--- | :--- | :--- |
| **Python (Interpreter Baseline)** | 0.23 ms | 0.22 ms | 0.04 ms |
| **C++ (AYNIYH Compiled)** | **0.14 ms** | **0.10 ms** | 0.20 ms |

## Key Features
* **Zero Python Overhead:** Written completely in C++17 for native, OS-level execution.
* **Ultra-Low Latency:** Averages 0.14 ms inference time, ensuring cursor movements never stutter.
* **Lightweight 1D CNN:** Instead of running heavy 2D image classifiers, it runs spatial mathematics on 63 coordinate points (21 joints × 3 axes).
* **Hardware Accessible:** Works on standard consumer webcams—no expensive depth-sensors or infrared cameras required.
* **Built-in Stability:** Includes a 5-frame voting debouncer and teleportation guards to prevent accidental clicks or erratic mouse jumps caused by webcam noise.

## Tech Stack
* **C++17** - Core execution and memory management
* **ONNX Runtime** - Neural network inference engine
* **OpenCV** - Video capture and matrix math
* **MediaPipe** - Hand landmark extraction

## Prerequisites
To build and run this project locally, you will need:
* Windows OS
* Visual Studio 2022 (with Desktop development with C++ workload)
* CMake (3.15 or higher)
* A standard webcam

## Installation & Build Instructions

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/minadas482/the_computer_control_with_hand_gestures.git](https://github.com/minadas482/the_computer_control_with_hand_gestures.git)
   cd the_computer_control_with_hand_gestures
   ```

2. **Configure with CMake:**
   Create a build directory and generate the Visual Studio project files.
   ```bash
   mkdir build
   cd build
   cmake ..
   ```

3. **Build the project:**
   Open the generated `.sln` file in Visual Studio, or build directly from the command line:
   ```bash
   cmake --build . --config Release
   ```

4. **Run the application:**
   Ensure your webcam is connected, then run the generated `.exe` file in the `Release` folder.
