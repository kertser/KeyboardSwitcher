#include "Languages.h"

#include <onnxruntime_cxx_api.h>
#include <windows.h>
#include <fstream>
#include <algorithm>
#include <numeric>

// nlohmann/json single-header
#include <nlohmann/json.hpp>

// ============================================================
// Keyboard layout strings
// ============================================================
namespace Layouts {

    // Russian layout (lowercase + uppercase rows)
    const std::wstring russian_layout =
        L"\u0451\u0439\u0446\u0443\u043a\u0435\u043d\u0433\u0448\u0449\u0437\u0445\u044a"
        L"\u0444\u044b\u0432\u0430\u043f\u0440\u043e\u043b\u0434\u0436\u044d"
        L"\\\u044f\u0447\u0441\u043c\u0438\u0442\u044c\u0431\u044e."
        L"\u0401\u0419\u0426\u0423\u041a\u0415\u041d\u0413\u0428\u0429\u0417\u0425\u042a"
        L"\u0424\u042b\u0412\u0410\u041f\u0420\u041e\u041b\u0414\u0416\u042d"
        L"/\u042f\u0427\u0421\u041c\u0418\u0422\u042c\u0411\u042e,";

    // English layout
    const std::wstring english_layout =
        L"`qwertyuiop[]asdfghjkl;'\\zxcvbnm,./"
        L"~QWERTYUIOP{}ASDFGHJKL:\"|ZXCVBNM<>?";

    // Hebrew layout
    const std::wstring hebrew_layout =
        L";/'\u05e7\u05e8\u05d0\u05d8\u05d5\u05df\u05dd\u05e4]"
        L"[\u05e9\u05d3\u05d2\u05db\u05e2\u05d9\u05d7\u05dc\u05da\u05e3,"
        L"\\\u05d6\u05e1\u05d1\u05d4\u05e0\u05de\u05e6\u05ea\u05e5.";
}

// ============================================================
// Layout conversion functions
// ============================================================
std::unordered_map<wchar_t, wchar_t> CreateConversionMap(
    const std::wstring& sourceLayout, const std::wstring& targetLayout)
{
    std::unordered_map<wchar_t, wchar_t> map;
    size_t len = (std::min)(sourceLayout.size(), targetLayout.size());
    for (size_t i = 0; i < len; ++i) {
        map[sourceLayout[i]] = targetLayout[i];
    }
    return map;
}

std::wstring ConvertText(const std::wstring& text,
    const std::unordered_map<wchar_t, wchar_t>& conversionMap)
{
    std::wstring result;
    result.reserve(text.size());
    for (wchar_t ch : text) {
        auto it = conversionMap.find(ch);
        if (it != conversionMap.end()) {
            result.push_back(it->second);
        } else {
            result.push_back(ch);
        }
    }
    return result;
}

std::wstring ConvertTextBidirectional(const std::wstring& text,
    const std::wstring& fromLayout, const std::wstring& toLayout)
{
    std::wstring input = text;
    // If converting to Hebrew, lowercase the input first
    if (&toLayout == &Layouts::hebrew_layout || toLayout == Layouts::hebrew_layout) {
        for (auto& ch : input) {
            ch = towlower(ch);
        }
    }
    auto map = CreateConversionMap(fromLayout, toLayout);
    return ConvertText(input, map);
}

// ============================================================
// LanguageDetector implementation (ONNX Runtime)
// ============================================================

// UTF-8 <-> UTF-16 helpers
static std::wstring Utf8ToWide(const std::string& utf8) {
    if (utf8.empty()) return {};
    int needed = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), (int)utf8.size(), nullptr, 0);
    std::wstring wide(needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), (int)utf8.size(), wide.data(), needed);
    return wide;
}

static std::string WideToUtf8(const std::wstring& wide) {
    if (wide.empty()) return {};
    int needed = WideCharToMultiByte(CP_UTF8, 0, wide.data(), (int)wide.size(), nullptr, 0, nullptr, nullptr);
    std::string utf8(needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), (int)wide.size(), utf8.data(), needed, nullptr, nullptr);
    return utf8;
}

struct LanguageDetector::Impl {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "KeyboardSwitcher"};
    std::unique_ptr<Ort::Session> session;
    std::unordered_map<wchar_t, int64_t> charToIndex;
};

LanguageDetector::LanguageDetector() : pImpl(std::make_unique<Impl>()) {}
LanguageDetector::~LanguageDetector() = default;

bool LanguageDetector::Load(const std::wstring& modelPath, const std::wstring& dictionaryPath) {
    try {
        // Load ONNX model
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        pImpl->session = std::make_unique<Ort::Session>(pImpl->env, modelPath.c_str(), sessionOptions);

        // Load dictionary.json
        std::ifstream file(dictionaryPath.c_str());
        if (!file.is_open()) return false;

        nlohmann::json j;
        file >> j;

        for (auto& [key, value] : j.items()) {
            // key is a UTF-8 string (single character), value is int
            std::wstring wkey = Utf8ToWide(key);
            if (!wkey.empty()) {
                pImpl->charToIndex[wkey[0]] = value.get<int64_t>();
            }
        }

        return true;
    }
    catch (const std::exception&) {
        return false;
    }
}

std::optional<std::string> LanguageDetector::PredictLanguage(const std::wstring& text) {
    if (!pImpl->session) return std::nullopt;

    try {
        // Tokenize: convert characters to indices
        std::vector<int64_t> inputIndices;
        inputIndices.reserve(MAX_LENGTH);

        for (wchar_t ch : text) {
            auto it = pImpl->charToIndex.find(ch);
            if (it != pImpl->charToIndex.end()) {
                inputIndices.push_back(it->second);
            }
        }

        // Pad to MAX_LENGTH
        while (static_cast<int>(inputIndices.size()) < MAX_LENGTH) {
            inputIndices.push_back(0);
        }

        // Truncate if longer
        if (static_cast<int>(inputIndices.size()) > MAX_LENGTH) {
            inputIndices.resize(MAX_LENGTH);
        }

        // Create input tensor [1, MAX_LENGTH]
        std::array<int64_t, 2> inputShape = {1, MAX_LENGTH};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<int64_t>(
            memoryInfo, inputIndices.data(), inputIndices.size(),
            inputShape.data(), inputShape.size());

        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputNamePtr = pImpl->session->GetInputNameAllocated(0, allocator);
        auto outputNamePtr = pImpl->session->GetOutputNameAllocated(0, allocator);

        const char* inputNames[] = {inputNamePtr.get()};
        const char* outputNames[] = {outputNamePtr.get()};

        // Run inference
        auto outputTensors = pImpl->session->Run(
            Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

        // Get output
        float* outputData = outputTensors[0].GetTensorMutableData<float>();
        auto outputInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
        size_t outputSize = outputInfo.GetElementCount();

        // Argmax
        int predictedClass = 0;
        float maxVal = outputData[0];
        for (size_t i = 1; i < outputSize; ++i) {
            if (outputData[i] > maxVal) {
                maxVal = outputData[i];
                predictedClass = static_cast<int>(i);
            }
        }

        // Map class to language
        switch (predictedClass) {
            case 0: return std::nullopt;  // N/A
            case 1: return "en";
            case 2: return "he";
            case 3: return "ru";
            default: return std::nullopt;
        }
    }
    catch (const std::exception&) {
        return std::nullopt;
    }
}

