
#include "gesture_cnn.h"
#include <iostream>
#include <cmath>

GestureCNN::GestureCNN(const std::wstring& modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "GestureCNN"),
    sessionOptions(),
    session(nullptr),
    memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)),
    lastID(-1)
{
    sessionOptions.SetIntraOpNumThreads(4);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = Ort::Session(env, modelPath.c_str(), sessionOptions);
}

int GestureCNN::predict(const std::vector<float>& inputData) {
    if (inputData.size() < 63) return -1;

    std::vector<int64_t> inputShape = { 1, 21, 3 };
    std::vector<float> brainData(inputData.begin(), inputData.begin() + 63);

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, brainData.data(), brainData.size(), inputShape.data(), inputShape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNamePtr = session.GetInputNameAllocated(0, allocator);
    auto outputNamePtr = session.GetOutputNameAllocated(0, allocator);
    const char* inputNames[] = { inputNamePtr.get() };
    const char* outputNames[] = { outputNamePtr.get() };

    auto outputTensors = session.Run(Ort::RunOptions{ nullptr }, inputNames, &inputTensor, 1, outputNames, 1);

    // Per train_cnn.py: Dense(5, activation='softmax')
    // The model outputs exactly 5 true probabilities. No manual math needed!
    float* probabilities = outputTensors.front().GetTensorMutableData<float>();

    // --- FIND THE BEST GESTURE (Out of 5) ---
    int bestID = -1;
    float maxP = 0.0f;

    for (int i = 0; i < 5; i++) {
        if (probabilities[i] > maxP) {
            maxP = probabilities[i];
            bestID = i;
        }
    }

    if (maxP > 0.60f) {
        if (bestID != lastID) {
            std::cout << "[BRAIN] GESTURE ID " << bestID << " DETECTED (" << (int)(maxP * 100) << "% Confidence)" << std::endl;
            lastID = bestID;
        }
        return bestID;
    }

    return -1;
}
