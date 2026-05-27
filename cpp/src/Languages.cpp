#include "Languages.h"
#include "Config.h"

#include <onnxruntime_cxx_api.h>
#include <windows.h>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <map>
#include <deque>
#include <unordered_map>
#include <cmath>
#include <sstream>

// nlohmann/json single-header
#include <nlohmann/json.hpp>
#include <set>

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

    const std::wstring& GetLayoutForLanguage(const std::string& lang) {
        if (lang == "ru") return russian_layout;
        if (lang == "he") return hebrew_layout;
        return english_layout;
    }
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
// Cached conversion maps (Iteration 4 — perf optimisation)
// ============================================================
// Lazily initialised on first use; thereafter returns a const ref
// to the pre-built map.  Non-standard pairs fall back to a local
// static (one extra allocation, but safe for unusual inputs).
const std::unordered_map<wchar_t, wchar_t>& GetCachedConversionMap(
    const std::wstring& sourceLayout, const std::wstring& targetLayout)
{
    // Key: pair of pointers to the global layout objects.
    // Because all callers use the same three globals the pointer
    // comparison is safe and avoids hashing large strings.
    using PtrPair = std::pair<const std::wstring*, const std::wstring*>;

    static std::map<PtrPair, std::unordered_map<wchar_t, wchar_t>> cache;
    static bool initialised = false;

    if (!initialised) {
        const std::wstring* layouts[] = {
            &Layouts::english_layout,
            &Layouts::russian_layout,
            &Layouts::hebrew_layout
        };
        for (auto* from : layouts)
            for (auto* to : layouts)
                if (from != to)
                    cache[{from, to}] = CreateConversionMap(*from, *to);
        initialised = true;
    }

    PtrPair key{&sourceLayout, &targetLayout};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    // Fallback: non-standard pair — store it so the next call is fast.
    cache[key] = CreateConversionMap(sourceLayout, targetLayout);
    return cache[key];
}

// ============================================================
// Hebrew final-form normalisation (Iteration 2 — B)
// ============================================================
// Maps word-final (sofit) Hebrew letters to their base forms.
// Used exclusively as a model-input pre-processing step so that
// confidence is boosted for text ending with sofit letters.
// The original user text is never modified by this function.
std::wstring NormalizeHebrewFinals(const std::wstring& text) {
    std::wstring result = text;
    for (wchar_t& ch : result) {
        switch (ch) {
            case 0x05DA: ch = 0x05DB; break; // ך → כ  (kaf sofit → kaf)
            case 0x05DD: ch = 0x05DE; break; // ם → מ  (mem sofit → mem)
            case 0x05DF: ch = 0x05E0; break; // ן → נ  (nun sofit → nun)
            case 0x05E3: ch = 0x05E4; break; // ף → פ  (pe  sofit → pe)
            case 0x05E5: ch = 0x05E6; break; // ץ → צ  (tsadi sofit → tsadi)
            default: break;
        }
    }
    return result;
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

        // ── MinKnownCharsForInference gate (Iteration 1 — A) ──────
        // If fewer than the configured minimum characters are recognised
        // by the model vocabulary, the input is too noisy/sparse to
        // produce a reliable prediction.  Return nullopt early.
        if (static_cast<int>(inputIndices.size()) < Config::MinKnownCharsForInference) {
            ++Config::Guards.skipLowKnownChars;
            return std::nullopt;
        }

        // Pad to MAX_LENGTH
        while (static_cast<int>(inputIndices.size()) < MAX_LENGTH) {
            inputIndices.push_back(0);
        }
        if (static_cast<int>(inputIndices.size()) > MAX_LENGTH) {
            inputIndices.resize(MAX_LENGTH);
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

static std::string ClassToLanguage(int cls) {
    switch (cls) {
        case 1: return "en";
        case 2: return "he";
        case 3: return "ru";
        default: return "";
    }
}

// Internal helper: run one ONNX inference pass and return softmax result.
// Separated so PredictLanguageWithConfidence can call it for both the
// original and the Hebrew-finals-normalised variant.
static std::optional<DetectionResult> RunInference(
    Ort::Session& session,
    const std::unordered_map<wchar_t, int64_t>& charToIndex,
    const std::wstring& text,
    int maxLength)
{
    std::vector<int64_t> inputIndices;
    inputIndices.reserve(maxLength);

    for (wchar_t ch : text) {
        auto it = charToIndex.find(ch);
        if (it != charToIndex.end())
            inputIndices.push_back(it->second);
    }

    // knownChars gate – checked by caller, but guard here too
    if (static_cast<int>(inputIndices.size()) < Config::MinKnownCharsForInference)
        return std::nullopt;

    while (static_cast<int>(inputIndices.size()) < maxLength)
        inputIndices.push_back(0);
    if (static_cast<int>(inputIndices.size()) > maxLength)
        inputIndices.resize(maxLength);

    std::array<int64_t, 2> inputShape = {1, static_cast<int64_t>(maxLength)};
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<int64_t>(
        memoryInfo, inputIndices.data(), inputIndices.size(),
        inputShape.data(), inputShape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    auto inputNamePtr  = session.GetInputNameAllocated(0, allocator);
    auto outputNamePtr = session.GetOutputNameAllocated(0, allocator);

    const char* inputNames[]  = {inputNamePtr.get()};
    const char* outputNames[] = {outputNamePtr.get()};

    auto outputTensors = session.Run(
        Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

    float* outputData  = outputTensors[0].GetTensorMutableData<float>();
    auto   outputInfo  = outputTensors[0].GetTensorTypeAndShapeInfo();
    size_t outputSize  = outputInfo.GetElementCount();
    if (outputSize < 4) return std::nullopt;

    // Softmax
    float maxLogit = *std::max_element(outputData, outputData + outputSize);
    float sumExp   = 0.0f;
    DetectionResult result = {};
    for (size_t i = 0; i < 4; ++i) {
        result.scores[i] = std::exp(outputData[i] - maxLogit);
        sumExp += result.scores[i];
    }
    for (size_t i = 0; i < 4; ++i)
        result.scores[i] /= sumExp;

    int   bestClass = 0;
    float bestProb  = 0.0f;
    for (int i = 1; i < 4; ++i) {
        if (result.scores[i] > bestProb) {
            bestProb  = result.scores[i];
            bestClass = i;
        }
    }

    if (result.scores[0] > bestProb) return std::nullopt;

    result.language   = ClassToLanguage(bestClass);
    result.confidence = bestProb;
    if (result.language.empty()) return std::nullopt;
    return result;
}

std::optional<DetectionResult> LanguageDetector::PredictLanguageWithConfidence(const std::wstring& text) {
    if (!pImpl->session) return std::nullopt;

    try {
        // Count known chars for the gate (fast pass — no allocation)
        int knownChars = 0;
        for (wchar_t ch : text) {
            if (pImpl->charToIndex.count(ch)) ++knownChars;
        }

        // ── MinKnownCharsForInference gate (Iteration 1 — A) ──────
        if (knownChars < Config::MinKnownCharsForInference) {
            ++Config::Guards.skipLowKnownChars;
            return std::nullopt;
        }

        // ── Primary inference on original text ─────────────────────
        auto result = RunInference(*pImpl->session, pImpl->charToIndex, text, MAX_LENGTH);
        if (!result.has_value()) return std::nullopt;

        // ── Hebrew final-form normalisation boost (Iteration 2 — B) ─
        // If the text contains any Hebrew characters (including sofit
        // forms), run a second inference on the normalised text and keep
        // whichever confidence is higher.  This handles words that end
        // with ך, ם, ן, ף, ץ — the model was trained mostly on
        // base (non-sofit) forms, so normalisation often yields a
        // higher confidence for the Hebrew class.
        bool hasHebrew = false;
        for (wchar_t c : text) {
            if (c >= 0x05D0 && c <= 0x05EA) { hasHebrew = true; break; }
        }
        if (hasHebrew) {
            std::wstring normed = NormalizeHebrewFinals(text);
            if (normed != text) {
                auto normResult = RunInference(
                    *pImpl->session, pImpl->charToIndex, normed, MAX_LENGTH);
                if (normResult.has_value() &&
                    normResult->language == result->language &&
                    normResult->confidence > result->confidence) {
                    // Boost: use the higher confidence but keep the
                    // original predicted language (they agree).
                    result->confidence = normResult->confidence;
                    // Copy scores from the better run for full visibility
                    std::copy(std::begin(normResult->scores),
                              std::end(normResult->scores),
                              std::begin(result->scores));
                }
            }
        }

        return result;
    }
    catch (const std::exception&) {
        return std::nullopt;
    }
}

// ============================================================
// Hebrew script coverage helper (Iteration 3)
// ============================================================
float ComputeHebrewScriptCoverage(const std::wstring& text) {
    int total = 0, hebrew = 0;
    for (wchar_t c : text) {
        if (c == L' ' || c == L'\t') continue;
        if (c >= 0x05D0 && c <= 0x05EA) {
            ++hebrew;
            ++total;
        } else if (std::iswalpha(c)) {
            ++total;
        }
    }
    return (total > 0) ? (static_cast<float>(hebrew) / static_cast<float>(total)) : 0.0f;
}

// ============================================================
// DetectionHistory
// ============================================================
void DetectionHistory::Update(const std::string& lang, float confidence,
                               const std::string& top2lang, float top2conf,
                               const float* scores4)
{
    if (lang == lastLang_) {
        ++streak_;
    } else {
        lastLang_ = lang;
        streak_ = 1;
    }

    // Maintain sliding window for trend gate
    HistoryFrame frame;
    frame.top1Lang = lang;
    frame.top1Conf = confidence;
    frame.top2Lang = top2lang;
    frame.top2Conf = top2conf;
    frame.margin   = confidence - top2conf;
    if (scores4) {
        for (int i = 0; i < 4; ++i) frame.scores[i] = scores4[i];
    }

    window_.push_back(frame);
    while (static_cast<int>(window_.size()) > MAX_WINDOW)
        window_.pop_front();
}

bool DetectionHistory::IsConsistent(const std::string& currentLang, int requiredCount) const {
    return currentLang == lastLang_ && streak_ >= requiredCount;
}

bool DetectionHistory::IsTrendStable(const std::string& toLang,
                                      int windowSize, int minSteps,
                                      float minSlope, float minMargin) const
{
    if (windowSize <= 0 || window_.empty() || minSteps <= 0) return false;

    int available = static_cast<int>(window_.size());
    int start     = std::max(0, available - windowSize);

    // Collect confidence values for frames where top1Lang == toLang
    std::vector<float> confs;
    confs.reserve(windowSize);
    float lastMargin = -1.0f;
    for (int i = start; i < available; ++i) {
        if (window_[i].top1Lang == toLang) {
            confs.push_back(window_[i].top1Conf);
            lastMargin = window_[i].margin;
        }
    }

    if (static_cast<int>(confs.size()) < minSteps) return false;

    // Slope check (optional — disabled when minSlope <= 0)
    if (minSlope > 0.0f && confs.size() >= 2) {
        float slope = (confs.back() - confs.front())
                    / static_cast<float>(confs.size() - 1);
        if (slope < minSlope) return false;
    }

    // Margin check in the most recent toLang frame (optional)
    if (minMargin > 0.0f && lastMargin >= 0.0f && lastMargin < minMargin)
        return false;

    return true;
}

void DetectionHistory::Clear() {
    lastLang_.clear();
    streak_ = 0;
    window_.clear();
}

// ── Persistent Moderate Confidence Gate (Iteration 3) ─────────────────────
// Returns true when ALL of the last minSteps frames had top1Lang == toLang
// AND the average confidence in those frames was >= minAvgConf.
// A flat, persistent signal (e.g. 58% × 5 keystrokes) triggers even without
// the growth requirement of IsTrendStable.
bool DetectionHistory::IsPersistentModerateConf(const std::string& toLang,
                                                float minAvgConf,
                                                int   minSteps) const
{
    if (minSteps <= 0 || minAvgConf <= 0.0f || window_.empty()) return false;

    int available = static_cast<int>(window_.size());
    if (available < minSteps) return false;

    float sum = 0.0f;
    int   count = 0;
    // Walk the tail of the window (most recent minSteps frames)
    for (int i = available - minSteps; i < available; ++i) {
        if (window_[i].top1Lang != toLang) return false;  // streak broken
        sum += window_[i].top1Conf;
        ++count;
    }
    return (count == minSteps) && (sum / static_cast<float>(count) >= minAvgConf);
}

// ── Cumulative Weak Score Gate (Iteration 3) ──────────────────────────────
// Returns the average softmax score for class `classIdx` over the last
// `windowSize` frames that have non-zero score data.  Returns 0 if none.
float DetectionHistory::GetAvgClassScore(int classIdx, int windowSize) const
{
    if (classIdx < 0 || classIdx >= 4 || windowSize <= 0 || window_.empty())
        return 0.0f;

    int available = static_cast<int>(window_.size());
    int start     = std::max(0, available - windowSize);

    float sum   = 0.0f;
    int   count = 0;
    for (int i = start; i < available; ++i) {
        // Only include frames where score data is present
        bool hasData = false;
        for (int k = 0; k < 4; ++k) if (window_[i].scores[k] != 0.0f) { hasData = true; break; }
        if (!hasData) continue;
        sum += window_[i].scores[classIdx];
        ++count;
    }
    return (count > 0) ? (sum / static_cast<float>(count)) : 0.0f;
}

// ============================================================
// Typo-resilient detection
// ============================================================

// Helper: map class index to language string (used in TypoResilientDetect
// for extracting runner-up from scores[]).
static const char* kClassLang[4] = { "", "en", "he", "ru" };

std::optional<DetectionResult> TypoResilientDetect(
    LanguageDetector& detector,
    const std::vector<std::wstring>& textVariants,
    const std::string& currentLang,
    size_t numChars,
    DetectionHistory& history,
    const std::set<std::string>& excludedLangs)
{
    const bool isFallback = !excludedLangs.empty();

    // Pre-fetch params for the →he pair (needed by the Hebrew script gate
    // which runs inside Pass 1 before bestLang is known).
    const auto& heParams = Config::GetParamsForPair(currentLang, "he");

    // ─── Pass 1: find best language across all variants ──────────
    std::string bestLang;
    float       bestConf = 0.0f;
    std::wstring bestVariant;
    DetectionResult bestFullResult = {};
    std::unordered_map<std::string, int> langVotes;

    // Tier 3 — Hebrew Script Coverage Gate:
    // Track the best "virtual Hebrew confidence" seen across variants.
    float       heScriptVirtualConf = 0.0f;
    std::wstring heScriptVariant;

    for (const auto& variant : textVariants) {
        auto result = detector.PredictLanguageWithConfidence(variant);

        // ── Hebrew script coverage check (parallel to ONNX) ──────────
        if (Config::EnableHebrewScriptGate &&
            heParams.HebrewScriptVirtualConf > 0.0f &&
            currentLang != "he" &&
            !excludedLangs.count("he"))
        {
            float coverage = ComputeHebrewScriptCoverage(variant);
            if (coverage >= heParams.HebrewScriptCoverageThreshold) {
                // The variant looks like Hebrew text.
                // Only apply if ONNX did NOT strongly identify this same
                // variant as a non-Hebrew language (false-positive guard).
                bool onnxContradicts = result.has_value() &&
                                       result->language != "he" &&
                                       result->confidence > 0.80f;
                if (!onnxContradicts) {
                    float vc = coverage * heParams.HebrewScriptVirtualConf;
                    if (vc > heScriptVirtualConf) {
                        heScriptVirtualConf = vc;
                        heScriptVariant     = variant;
                    }
                }
            }
        }

        if (!result.has_value()) continue;
        if (excludedLangs.count(result->language)) continue;

        langVotes[result->language]++;
        if (result->confidence > bestConf) {
            bestConf       = result->confidence;
            bestLang       = result->language;
            bestVariant    = variant;
            bestFullResult = *result;
        }
    }

    // ── Apply Hebrew script gate if it beats ONNX ─────────────────
    bool heScriptGateFired = false;
    if (!isFallback &&
        heScriptVirtualConf > bestConf &&
        !heScriptVariant.empty())
    {
        bestLang    = "he";
        bestConf    = heScriptVirtualConf;
        bestVariant = heScriptVariant;
        bestFullResult = {};  // no real ONNX scores for this path
        heScriptGateFired = true;

        char buf[192];
        std::snprintf(buf, sizeof(buf),
            "KS:HEBREW_SCRIPT_GATE pair=%s->he virtualConf=%.3f coverage_threshold=%.2f\n",
            currentLang.c_str(), heScriptVirtualConf,
            heParams.HebrewScriptCoverageThreshold);
        OutputDebugStringA(buf);
    }

    if (bestLang.empty()) {
        if (!isFallback) history.Update("", 0.0f);
        return std::nullopt;
    }

    // ─── Extract runner-up from the winning variant's softmax scores ─
    // scores[0]=N/A, [1]=en, [2]=he, [3]=ru
    std::string runnerUpLang;
    float       runnerUpConf = 0.0f;
    for (int i = 1; i < 4; ++i) {
        const std::string cls = kClassLang[i];
        if (cls == bestLang || excludedLangs.count(cls)) continue;
        if (bestFullResult.scores[i] > runnerUpConf) {
            runnerUpConf = bestFullResult.scores[i];
            runnerUpLang = cls;
        }
    }
    float margin = bestConf - runnerUpConf;
    int   variantAgreement = langVotes.count(bestLang) ? langVotes.at(bestLang) : 0;

    // ─── Look up per-pair switching parameters ────────────────────
    const auto& params = Config::GetParamsForPair(currentLang, bestLang);

    // Phrase mode: if the detection text contains a space, we have at least
    // two words — apply PhraseConfScale to lower the required threshold.
    bool isPhrase = false;
    for (const auto& v : textVariants) {
        if (v.find(L' ') != std::wstring::npos) { isPhrase = true; break; }
    }
    float requiredConfidence = params.GetRequiredConfidence(numChars, isPhrase);

    // Per-pair min-chars check
    if (static_cast<int>(numChars) < params.EarlyDetectionMinChars) {
        if (!isFallback) history.Update(bestLang, bestConf, runnerUpLang, runnerUpConf);
        return std::nullopt;
    }

    // ─── Short-input extra confidence ─────────────────────────────
    // For alphaCount in [EarlyDetectionMinChars, +2] we require a higher
    // bar to suppress the main source of FP on short words.
    // (Not applied in phrase mode — PhraseConfScale already adjusts threshold.)
    bool isShortInput = (static_cast<int>(numChars) <=
                         params.EarlyDetectionMinChars + 2);
    if (isShortInput && !isPhrase && params.ShortInputExtraConf > 0.0f) {
        requiredConfidence = std::min(0.9999f,
                                      requiredConfidence + params.ShortInputExtraConf);
    }

    // ─── Tier 2: Drop-one boosting (borderline zone) ──────────────
    if (Config::EnableTypoResilience &&
        bestConf < requiredConfidence &&
        bestConf >= requiredConfidence * params.BorderlineZoneFactor &&
        bestVariant.size() > 2)
    {
        for (size_t i = 0; i < bestVariant.size(); ++i) {
            std::wstring dropped = bestVariant.substr(0, i)
                                 + bestVariant.substr(i + 1);
            auto res = detector.PredictLanguageWithConfidence(dropped);
            if (res.has_value() &&
                res->language == bestLang &&
                res->confidence > bestConf)
            {
                bestConf = res->confidence;
                // Update margin with potentially higher confidence
                margin   = bestConf - runnerUpConf;
            }
        }
    }

    // ─── History & consecutive-agreement gate (Tier 1) ────────────
    if (!isFallback) {
        // Pass full softmax scores when they are available
        const float* scoresPtr = (bestFullResult.scores[0] != 0.0f ||
                                  bestFullResult.scores[1] != 0.0f ||
                                  bestFullResult.scores[2] != 0.0f ||
                                  bestFullResult.scores[3] != 0.0f)
                                     ? bestFullResult.scores
                                     : nullptr;
        history.Update(bestLang, bestConf, runnerUpLang, runnerUpConf, scoresPtr);

        if (Config::EnableTypoResilience) {
            bool agreementOk = history.IsConsistent(bestLang,
                                                     params.ConsecutiveAgreementCount);

            // Trend gate as OR-alternative to agreement gate
            bool trendOk = false;
            if (Config::EnableTrendGate && params.TrendWindowSize > 0) {
                trendOk = history.IsTrendStable(
                    bestLang,
                    params.TrendWindowSize,
                    params.MinStableSteps,
                    params.MinTrendSlope,
                    0.0f);   // margin checked separately below
            }

            // ── Tier 3-A: Persistent Moderate Confidence Gate ─────────
            bool persistentOk = false;
            if (Config::EnablePersistentConfGate &&
                params.PersistentMinSteps > 0 &&
                params.PersistentMinAvgConf > 0.0f)
            {
                persistentOk = history.IsPersistentModerateConf(
                    bestLang,
                    params.PersistentMinAvgConf,
                    params.PersistentMinSteps);
            }

            // ── Tier 3-B: Cumulative Weak Score Gate ──────────────────
            bool weakScoreOk = false;
            if (Config::EnableWeakScoreGate &&
                params.WeakScoreClassIdx >= 0 &&
                params.WeakScoreWindow > 0 &&
                params.WeakScoreMinAvg > 0.0f)
            {
                const std::string weakLang = (params.WeakScoreClassIdx == 1) ? "en" :
                                             (params.WeakScoreClassIdx == 2) ? "he" :
                                             (params.WeakScoreClassIdx == 3) ? "ru" : "";
                if (weakLang == bestLang) {
                    float avg = history.GetAvgClassScore(params.WeakScoreClassIdx,
                                                         params.WeakScoreWindow);
                    if (avg >= params.WeakScoreMinAvg) {
                        weakScoreOk = true;
                        char buf[192];
                        std::snprintf(buf, sizeof(buf),
                            "KS:WEAK_SCORE_GATE pair=%s->%s avgScore=%.3f "
                            "threshold=%.3f window=%d\n",
                            currentLang.c_str(), bestLang.c_str(),
                            avg, params.WeakScoreMinAvg, params.WeakScoreWindow);
                        OutputDebugStringA(buf);
                    }
                }
            }

            // ── Tier 3-C: Hebrew Script Gate as agreement bypass ───────
            // When heScriptGateFired, the virtual confidence already changed
            // bestLang to "he".  Allow it to pass the gate after the normal
            // ConsecutiveAgreementCount steps have confirmed "he".
            // (heScriptGateFired does NOT unconditionally skip gates — the
            //  agreement step count is still required to prevent single-shot
            //  false positives.  The gate simply changes what bestLang is.)

            // Debug: log gate states
            {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "KS:GATE pair=%s->%s conf=%.3f margin=%.3f "
                    "agree=%d trend=%d persist=%d weak=%d script=%d streak=%d varVotes=%d/%d alpha=%zu\n",
                    currentLang.c_str(), bestLang.c_str(),
                    bestConf, margin,
                    (int)agreementOk, (int)trendOk,
                    (int)persistentOk, (int)weakScoreOk,
                    (int)heScriptGateFired,
                    history.GetStreak(), variantAgreement,
                    (int)textVariants.size(), numChars);
                OutputDebugStringA(buf);
            }

            if (!agreementOk && !trendOk && !persistentOk && !weakScoreOk) {
                return std::nullopt;
            }

            // If only trend gate passed (not agreement), log it
            if (!agreementOk && trendOk && !persistentOk && !weakScoreOk) {
                OutputDebugStringA("KS:TREND_GATE_PASSED (agreement not met)\n");
                if (!Config::EnableTrendGateBlock) {
                    return std::nullopt;
                }
            }

            if (!agreementOk && !trendOk && persistentOk) {
                OutputDebugStringA("KS:PERSISTENT_GATE_PASSED (agreement/trend not met)\n");
            }
        }
    }

    // ─── Margin gate ──────────────────────────────────────────────
    // Low margin means the model is split between two languages → risky.
    if (!isFallback && params.MinTop1Top2Margin > 0.0f &&
        margin < params.MinTop1Top2Margin)
    {
        char buf[192];
        std::snprintf(buf, sizeof(buf),
            "KS:BLOCKED[margin] pair=%s->%s conf=%.3f margin=%.3f req=%.3f\n",
            currentLang.c_str(), bestLang.c_str(),
            bestConf, margin, params.MinTop1Top2Margin);
        OutputDebugStringA(buf);
        return std::nullopt;
    }

    // ─── Variant consensus gate ───────────────────────────────────
    // For borderline confidence, require a minimum number of layout
    // variants to agree on the target language.
    if (Config::EnableVariantConsensus && !isFallback &&
        params.VariantAgreementCount > 0 &&
        bestConf < params.ConfidenceAtMinChars * 0.99f)
    {
        if (variantAgreement < params.VariantAgreementCount) {
            char buf[192];
            std::snprintf(buf, sizeof(buf),
                "KS:BLOCKED[variant_consensus] pair=%s->%s "
                "votes=%d/%d required=%d conf=%.3f\n",
                currentLang.c_str(), bestLang.c_str(),
                variantAgreement, (int)textVariants.size(),
                params.VariantAgreementCount, bestConf);
            OutputDebugStringA(buf);
            return std::nullopt;
        }
    }

    // ─── Final confidence threshold ───────────────────────────────
    if (bestConf >= requiredConfidence) {
        DetectionResult result = bestFullResult;
        result.language   = bestLang;
        result.confidence = bestConf;
        return result;
    }

    return std::nullopt;
}
