
#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class HandTracker {
public:
    HandTracker(const std::wstring& palmModelPath, const std::wstring& landmarkModelPath);
    std::vector<float> extractLandmarks(cv::Mat& frame);

private:
    std::string inputName;
    std::string outputName0;
    std::string outputName1;
    std::string outputName2;
    std::vector<float> preprocessImage(const cv::Mat& image, int width, int height);

    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    Ort::Session palmSession;
    Ort::Session landmarkSession;
    Ort::MemoryInfo memoryInfo;
    cv::Rect currentCropBox;
};
