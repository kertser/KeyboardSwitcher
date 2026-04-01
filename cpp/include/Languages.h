#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <optional>
#include <set>
#include <memory>

// Result of a single language prediction with confidence scores
struct DetectionResult {
    std::string language;   // "en", "he", "ru", or "" for N/A
    float confidence;       // softmax probability of the top class
    float scores[4];        // raw softmax probabilities [N/A, en, he, ru]
};

// ============================================================
// DetectionHistory – tracks consecutive-keystroke agreement
// ============================================================
class DetectionHistory {
public:
    // Record the result of the latest detection pass.
    void Update(const std::string& lang, float confidence);

    // Returns true when the last N detections (N = requiredCount)
    // all predicted the same language.
    bool IsConsistent(const std::string& currentLang, int requiredCount) const;

    // Reset (call on mouse-click, manual switch, Alt+Tab, etc.)
    void Clear();

private:
    std::string lastLang_;
    int         streak_ = 0;
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

// ============================================================
// Typo-resilient detection wrapper
// ============================================================
// Runs detection across all layout variants, applies:
//   Tier 1 — consecutive-agreement gate
//   Tier 2 — drop-one confidence boosting (borderline zone only)
// Uses per-language-pair parameters: after finding the best candidate
// language, looks up the (currentLang → bestLang) pair params to
// determine the required confidence threshold and other settings.
// Returns a DetectionResult if confident enough, or nullopt.
std::optional<DetectionResult> TypoResilientDetect(
    LanguageDetector& detector,
    const std::vector<std::wstring>& textVariants,
    const std::string& currentLang,
    size_t numChars,
    DetectionHistory& history,
    const std::set<std::string>& excludedLangs = {});

