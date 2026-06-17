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
//                    plus a rolling window of softmax frames for
//                    the Hebrew weak-signal gates.
// ============================================================
class DetectionHistory {
public:
    // Record the result of the latest detection pass.  scores4 (optional)
    // is the full softmax vector [N/A, en, he, ru]; when supplied it feeds
    // the persistent / weak-score gates.
    void Update(const std::string& lang, float confidence,
                const float* scores4 = nullptr);

    // Returns true when the last N detections (N = requiredCount)
    // all predicted the same language.
    bool IsConsistent(const std::string& currentLang, int requiredCount) const;

    // Persistent Moderate Confidence Gate: true when the last `minSteps`
    // frames ALL had top-1 == lang and their average top-1 confidence
    // is >= minAvgConf.
    bool IsPersistent(const std::string& lang, int minSteps,
                      float minAvgConf) const;

    // Cumulative Weak Score Gate: average softmax score for class index
    // `classIdx` over the last `window` frames (0 if not enough frames).
    float WeakScoreAvg(int classIdx, int window) const;

    // Reset (call on mouse-click, manual switch, Alt+Tab, etc.)
    void Clear();

private:
    std::string lastLang_;
    int         streak_ = 0;

    struct Frame {
        std::string lang;       // top-1 language ("" = N/A)
        float       conf = 0.0f;
        float       scores[4] = {0, 0, 0, 0};
    };
    static constexpr size_t MAX_WINDOW = 10;
    std::vector<Frame> frames_;   // newest at back, capped at MAX_WINDOW
};

class LanguageDetector {
public:
    LanguageDetector();
    ~LanguageDetector();

    // Load the ONNX model and dictionary.json
    bool Load(const std::wstring& modelPath, const std::wstring& dictionaryPath);


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
// Hebrew final-form normalisation (for model input only)
// ============================================================
// Replaces sofit (word-final) forms with their base equivalents:
//   ך→כ  ם→מ  ן→נ  ף→פ  ץ→צ
// Call this on text before passing to the model to improve detection
// confidence on words that end with sofit letters.  Never apply to
// text that the user sees — only to detection variant strings.
std::wstring NormalizeHebrewFinals(const std::wstring& text);

// ============================================================
// Cached conversion maps (Iteration 4 — perf)
// ============================================================
// Returns a reference to a statically cached conversion map for the
// six fixed layout pairs.  Falls back to a freshly created map for
// any non-standard pair.
const std::unordered_map<wchar_t, wchar_t>& GetCachedConversionMap(
    const std::wstring& sourceLayout, const std::wstring& targetLayout);

// ============================================================
// Hebrew script coverage helper (ported from 1.3.0)
// ============================================================
// Returns the fraction of alpha characters in `text` that fall in the
// Hebrew Unicode block (U+05D0–U+05EA).  Spaces and non-alpha chars are
// ignored.  Used by the Hebrew Script Coverage gate in TypoResilientDetect.
float ComputeHebrewScriptCoverage(const std::wstring& text);

// ============================================================
// Typo-resilient detection wrapper
// ============================================================
// Runs detection across all layout variants, applies:
//   Tier 1 — consecutive-agreement gate
//   Tier 2 — drop-one confidence boosting (borderline zone only)
// Uses per-language-pair parameters: after finding the best candidate
// language, looks up the (currentLang → bestLang) pair params to
// determine the required confidence threshold and other settings.
//
// excludedLangs — language codes (e.g. "he") to skip as candidates.
//                 Used for case-signal exclusion (Hebrew excluded when
//                 ALL-CAPS / internal capitals detected) and for the
//                 user-rejection fallback path.
//
// isFallback    — when true, the history/consecutive-agreement gate is
//                 skipped.  Set this only on the user-rejection retry
//                 call, NOT on the case-exclusion primary call (the
//                 exclusion must still honour the agreement gate).
//
// Returns a DetectionResult if confident enough, or nullopt.
std::optional<DetectionResult> TypoResilientDetect(
    LanguageDetector& detector,
    const std::vector<std::wstring>& textVariants,
    const std::string& currentLang,
    size_t numChars,
    DetectionHistory& history,
    const std::set<std::string>& excludedLangs = {},
    bool isFallback = false);

