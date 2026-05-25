
#include "hand_tracker.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <array>

HandTracker::HandTracker(const std::wstring& palmModelPath, const std::wstring& landmarkModelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "HandTracker"),
    sessionOptions(), palmSession(nullptr), landmarkSession(nullptr),
    memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) // FIXED: memoryInfo initialization (fixes C2512)
{
    // Performance optimization
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    palmSession = Ort::Session(env, palmModelPath.c_str(), sessionOptions);
    landmarkSession = Ort::Session(env, landmarkModelPath.c_str(), sessionOptions);
}

std::vector<float> HandTracker::preprocessImage(const cv::Mat& image, int width, int height) {
    cv::Mat resized, floatImg;
    cv::resize(image, resized, cv::Size(width, height));
    resized.convertTo(floatImg, CV_32F, 1.0 / 255.0);
    std::vector<float> tensorValues(width * height * 3);
    std::vector<cv::Mat> channels(3);
    cv::split(floatImg, channels);
    int size = width * height;
    memcpy(tensorValues.data(), channels[2].data, size * sizeof(float));
    memcpy(tensorValues.data() + size, channels[1].data, size * sizeof(float));
    memcpy(tensorValues.data() + size * 2, channels[0].data, size * sizeof(float));
    return tensorValues;
}

std::vector<float> HandTracker::extractLandmarks(cv::Mat& frame) {
    int frameW = frame.cols, frameH = frame.rows;

    // 1. RE-CENTERED BOX: Set to 85% size
    int boxSize = std::min(frameW, frameH) * 0.85;
    int offsetX = (frameW - boxSize) / 2;
    int offsetY = (frameH - boxSize) / 2;

    currentCropBox = cv::Rect(offsetX, offsetY, boxSize, boxSize);

    // Draw the green box
    cv::rectangle(frame, currentCropBox, cv::Scalar(0, 255, 0), 2);

    cv::Mat cropped = frame(currentCropBox);
    std::vector<float> input = preprocessImage(cropped, 224, 224);
    std::vector<int64_t> shape = { 1, 3, 224, 224 };
    Ort::Value inputT = Ort::Value::CreateTensor<float>(memoryInfo, input.data(), input.size(), shape.data(), shape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    auto inName = landmarkSession.GetInputNameAllocated(0, allocator);
    const char* inNames[] = { inName.get() };
    std::vector<const char*> outNames = { "xyz_x21", "hand_score", "lefthand_0_or_righthand_1" };
    auto outT = landmarkSession.Run(Ort::RunOptions{ nullptr }, inNames, &inputT, 1, outNames.data(), outNames.size());

    // 2. BALANCED CONFIDENCE FILTER: 70% threshold. 

    float handScore = 1.0f / (1.0f + std::exp(-outT[1].GetTensorMutableData<float>()[0]));
    if (handScore < 0.70f) {
        return {};
    }

    float* raw = outT[0].GetTensorMutableData<float>();

    // Zero-latency Stack Memory extraction
    std::array<float, 21> tempX, tempY, tempZ;
    for (int i = 0; i < 21; i++) {
        tempX[i] = raw[i * 3];
        tempY[i] = raw[i * 3 + 1];
        tempZ[i] = raw[i * 3 + 2];
    }

    // Make relative to Wrist (Index 0 becomes 0,0,0)
    float wristX = tempX[0], wristY = tempY[0], wristZ = tempZ[0];
    std::array<float, 21> relX, relY, relZ;

    for (int i = 0; i < 21; i++) {
        relX[i] = tempX[i] - wristX;
        relY[i] = tempY[i] - wristY;
        relZ[i] = tempZ[i] - wristZ;
    }

    std::vector<float> landmarks;
    landmarks.reserve(65);
    for (int i = 0; i < 21; i++) {
        landmarks.push_back(relX[i]);
        landmarks.push_back(relY[i]);
        landmarks.push_back(relZ[i]);
    }

    // Attach mouse position (Raw 0.0-1.0 scale mapped to landmark 9)
    landmarks.push_back(tempX[9] / 224.0f);
    landmarks.push_back(tempY[9] / 224.0f);

    return landmarks;
}
