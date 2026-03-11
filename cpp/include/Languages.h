#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <memory>

// Result of a single language prediction with confidence scores
struct DetectionResult {
    std::string language;   // "en", "he", "ru", or "" for N/A
    float confidence;       // softmax probability of the top class
    float scores[4];        // raw softmax probabilities [N/A, en, he, ru]
};

class LanguageDetector {
public:
    LanguageDetector();
    ~LanguageDetector();

    // Load the ONNX model and dictionary.json
    bool Load(const std::wstring& modelPath, const std::wstring& dictionaryPath);

    // Predict language from text. Returns "en", "he", "ru", or nullopt
    std::optional<std::string> PredictLanguage(const std::wstring& text);

    // Predict with full confidence information
    std::optional<DetectionResult> PredictLanguageWithConfidence(const std::wstring& text);

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl;
    static constexpr int MAX_LENGTH = 45;
};

// Keyboard layout strings (as wide strings for UTF-16)
namespace Layouts {
    extern const std::wstring russian_layout;
    extern const std::wstring english_layout;
    extern const std::wstring hebrew_layout;

    // Get layout string for a language code
    const std::wstring& GetLayoutForLanguage(const std::string& lang);
}

// Create a conversion map from source to target layout
std::unordered_map<wchar_t, wchar_t> CreateConversionMap(
    const std::wstring& sourceLayout, const std::wstring& targetLayout);

// Convert text using a conversion map
std::wstring ConvertText(const std::wstring& text,
    const std::unordered_map<wchar_t, wchar_t>& conversionMap);

// Convert text bidirectionally between layouts
std::wstring ConvertTextBidirectional(const std::wstring& text,
    const std::wstring& fromLayout, const std::wstring& toLayout);

