
#pragma once
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>

class GestureCNN {
public:
    GestureCNN(const std::wstring& modelPath);
    int predict(const std::vector<float>& inputData);
private:
    std::string inputName;
    std::string outputName;
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session session;
    Ort::MemoryInfo memoryInfo;
    int lastID;
};
