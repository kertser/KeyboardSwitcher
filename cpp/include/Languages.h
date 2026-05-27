#pragma once

#include <string>
#include <vector>
#include <deque>
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
//                    and a sliding confidence window for the
//                    Trend Gate (Iteration 2+)
// ============================================================

// One observation stored in the sliding window.
struct HistoryFrame {
    std::string top1Lang;
    float       top1Conf  = 0.0f;
    std::string top2Lang;   // runner-up language ("" if unknown)
    float       top2Conf  = 0.0f;
    float       margin    = 0.0f;  // top1Conf - top2Conf
    float       scores[4] = {};    // full softmax [N/A, en, he, ru] (0 if unavailable)
};

class DetectionHistory {
public:
    // Record the result of the latest detection pass.
    // scores4: optional pointer to float[4] with full softmax probs — stored in the
    // sliding window and used by the weak-signal cumulative gate.
    void Update(const std::string& lang, float confidence,
                const std::string& top2lang = "", float top2conf = 0.0f,
                const float* scores4 = nullptr);

    // Returns true when the last N detections (N = requiredCount)
    // all predicted the same language.
    bool IsConsistent(const std::string& currentLang, int requiredCount) const;

    // Trend Gate: returns true when the sliding window shows a stable
    // upward trend of confidence for toLang.
    //   windowSize   – how many recent frames to examine
    //   minSteps     – minimum frames where top1Lang == toLang
    //   minSlope     – minimum average per-step confidence increase
    //                  (0.0 = slope check disabled)
    //   minMargin    – minimum top1-top2 margin in the most recent
    //                  toLang frame (0.0 = margin check disabled)
    bool IsTrendStable(const std::string& toLang,
                       int windowSize, int minSteps,
                       float minSlope, float minMargin) const;

    // Persistent Moderate Confidence Gate (Iteration 3):
    // Returns true when ALL of the last minSteps frames had the same
    // top1Lang == toLang AND the average top1Conf across those frames is
    // >= minAvgConf.  Unlike IsTrendStable this does NOT require a rising
    // slope — a flat but persistent signal (e.g. 58% × 5 steps) is enough.
    bool IsPersistentModerateConf(const std::string& toLang,
                                  float minAvgConf, int minSteps) const;

    // Cumulative Weak Score Gate (Iteration 3):
    // Returns the average softmax score for class index classIdx
    // (0=N/A, 1=en, 2=he, 3=ru) over the last windowSize frames.
    // Only frames that have non-zero score data are included.
    // Returns 0 if no scored frames are available.
    float GetAvgClassScore(int classIdx, int windowSize) const;

    // Reset (call on mouse-click, manual switch, Alt+Tab, etc.)
    void Clear();

    int GetStreak() const { return streak_; }

private:
    std::string lastLang_;
    int         streak_ = 0;

    // Sliding window – last MAX_WINDOW observations, oldest first.
    // Bumped from 8 → 10 to support the persistent gate (minSteps up to 6).
    static constexpr int MAX_WINDOW = 10;
    std::deque<HistoryFrame> window_;
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
// Hebrew script coverage helper (Iteration 3)
// ============================================================
// Returns the fraction of alpha characters in `text` that fall in the
// Hebrew Unicode block (U+05D0–U+05EA).  Spaces are ignored.
// Used by the Hebrew Script Coverage gate in TypoResilientDetect.
float ComputeHebrewScriptCoverage(const std::wstring& text);

// ============================================================
// Typo-resilient detection wrapper
// ============================================================
// Runs detection across all layout variants, applies:
//   Tier 1 — consecutive-agreement gate
//   Tier 2 — drop-one confidence boosting (borderline zone only)
//   Tier 3 — Hebrew script coverage gate   (→he pairs)
//             Persistent moderate confidence gate
//             Cumulative weak Hebrew score gate
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
