
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <deque>
#include <map>
#include <cmath>
#include "gesture_cnn.h"
#include "hand_tracker.h"
#include "mouse_controller.h"

int main() {
    try {
        std::wstring path = L"C:/Users/johnc/Downloads/The Projects/The Computer Control With Hand Gestures/cpp/the_computer_control_with_hand_gestures/";
        GestureCNN brain(path + L"gesture_cnn.onnx");
        HandTracker tracker(path + L"palm_detection.onnx", path + L"hand_landmark.onnx");

        cv::VideoCapture cap(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 60);

        if (!cap.isOpened()) {
            std::cerr << "Cannot open webcam!" << std::endl;
            return -1;
        }

        cv::Mat frame;
        auto lastActionTime = std::chrono::steady_clock::now();
        int lostFramesCount = 0;

        // --- STABILITY BUFFERS ---
        std::deque<int> gestureHistory;
        const int HISTORY_SIZE = 5;
        float prevX = -1.0f, prevY = -1.0f;

        // --- TELEMETRY INITIALIZATION ---
        std::vector<double> inference_times;
        int frame_count = 0;
        std::cout << "Starting C++ Telemetry Data Collection (10000 frames)..." << std::endl;

        while (true) {
            cap >> frame;
            if (frame.empty()) break;

            cv::flip(frame, frame, 1);

            std::vector<float> data = tracker.extractLandmarks(frame);

            if (data.size() >= 65) {
                lostFramesCount = 0;

                float currX = data[63];
                float currY = data[64];

                // 1. TELEPORTATION GUARD
                if (prevX != -1.0f) {
                    float distance = std::sqrt(std::pow(currX - prevX, 2) + std::pow(currY - prevY, 2));
                    if (distance > 0.15f) {
                        MouseController::resetTracking();
                        prevX = currX; prevY = currY;
                        cv::imshow("AYNIYH High-Performance Tracker", frame);
                        if (cv::waitKey(1) == 27) break;
                        continue;
                    }
                }
                prevX = currX; prevY = currY;

                // 2. GESTURE INFERENCE VIA ONNX RUNTIME (WITH TELEMETRY)

                // --- START TELEMETRY TIMER ---
                auto start_time = std::chrono::high_resolution_clock::now();
                int raw_gesture = brain.predict(data);
                auto end_time = std::chrono::high_resolution_clock::now();
                // --- END TELEMETRY TIMER ---

                // Record Latency
                std::chrono::duration<double, std::milli> elapsed = end_time - start_time;
                inference_times.push_back(elapsed.count());
                frame_count++;

                if (raw_gesture != -1 && frame_count % 500 == 0) {
                    std::cout << "[" << frame_count << "/10000] Inference Executed. Latency: " << elapsed.count() << "ms" << std::endl;
                }

                // Export conditions
                if (frame_count >= 10000) {
                    std::ofstream outFile("cpp_latency.csv");
                    outFile << "Latency_ms\n";
                    for (double t : inference_times) {
                        outFile << t << "\n";
                    }
                    outFile.close();
                    std::cout << "C++ Benchmark Complete. Data saved to 'cpp_latency.csv'" << std::endl;
                    break;
                }

                gestureHistory.push_back(raw_gesture);
                if (gestureHistory.size() > HISTORY_SIZE) gestureHistory.pop_front();

                int gesture = -1;
                std::map<int, int> voteCount;
                for (int g : gestureHistory) voteCount[g]++;

                int maxVotes = 0;
                for (auto const& [g, count] : voteCount) {
                    if (count > maxVotes) {
                        maxVotes = count;
                        gesture = g;
                    }
                }

                if (maxVotes < 3) gesture = -1;

                // --- TRACKPAD CLUTCHING LOGIC ---
                if (gesture == 0 || gesture == 1) MouseController::resetTracking();
                else MouseController::moveCursor(currX, currY);

                auto now = std::chrono::steady_clock::now();
                auto msSinceLastAction = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastActionTime).count();

                switch (gesture) {
                case 2: case 3: case 6: case 7: MouseController::releaseLeft(); break;
                case 0: if (msSinceLastAction > 400) { MouseController::leftClick(); lastActionTime = now; } break;
                case 1: if (msSinceLastAction > 500) { MouseController::rightClick(); lastActionTime = now; } break;
                case 10: MouseController::holdLeftDown(); break;
                case 4: if (msSinceLastAction > 100) { MouseController::scroll(120); lastActionTime = now; } break;
                case 5: if (msSinceLastAction > 100) { MouseController::scroll(-120); lastActionTime = now; } break;
                case 8: if (msSinceLastAction > 150) { MouseController::volumeUp(); lastActionTime = now; } break;
                case 9: if (msSinceLastAction > 150) { MouseController::volumeDown(); lastActionTime = now; } break;
                case 11: if (msSinceLastAction > 1000) { MouseController::muteVolume(); lastActionTime = now; } break;
                default: MouseController::releaseLeft(); break;
                }
            }
            else {
                lostFramesCount++;
                if (lostFramesCount > 5) {
                    MouseController::resetTracking();
                    MouseController::releaseLeft();
                    prevX = -1.0f;
                    gestureHistory.clear();
                }
            }

            cv::imshow("AYNIYH High-Performance Tracker", frame);
            if (cv::waitKey(1) == 27) break;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
    }
    return 0;
}
