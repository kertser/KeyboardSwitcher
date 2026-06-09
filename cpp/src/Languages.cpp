#include "Languages.h"
#include "Config.h"

#include <onnxruntime_cxx_api.h>
#include <windows.h>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <map>
#include <cwctype>

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
// DetectionHistory
// ============================================================
void DetectionHistory::Update(const std::string& lang, float /*confidence*/) {
    if (lang == lastLang_) {
        ++streak_;
    } else {
        lastLang_ = lang;
        streak_ = 1;
    }
}

bool DetectionHistory::IsConsistent(const std::string& currentLang, int requiredCount) const {
    return currentLang == lastLang_ && streak_ >= requiredCount;
}

void DetectionHistory::Clear() {
    lastLang_.clear();
    streak_ = 0;
}

// ============================================================
// Hebrew script coverage helper (ported from 1.3.0)
// ============================================================
float ComputeHebrewScriptCoverage(const std::wstring& text) {
    int alpha = 0, hebrew = 0;
    for (wchar_t c : text) {
        if (c == L' ') continue;
        if (iswalpha(c)) {
            ++alpha;
            if (c >= 0x05D0 && c <= 0x05EA) ++hebrew;
        }
    }
    if (alpha == 0) return 0.0f;
    return static_cast<float>(hebrew) / static_cast<float>(alpha);
}

// Map softmax class index → language string (scores order: 0=N/A,1=en,2=he,3=ru)
static const char* kClassLang[4] = { "", "en", "he", "ru" };

// ============================================================
// Typo-resilient detection
// ============================================================
std::optional<DetectionResult> TypoResilientDetect(
    LanguageDetector& detector,
    const std::vector<std::wstring>& textVariants,
    const std::string& currentLang,
    size_t numChars,
    DetectionHistory& history,
    const std::set<std::string>& excludedLangs,
    bool isFallback)
{
    // isFallback is now an explicit parameter.
    // excludedLangs may be non-empty even on a primary (non-fallback) call
    // (e.g. case-signal Hebrew exclusion) — history gate still applies.

    // Pre-fetch →he params for the Hebrew script gate (runs during Pass 1,
    // before bestLang is known).
    const auto& heParams = Config::GetParamsForPair(currentLang, "he");

    // --- Pass 1: find best language across all variants ---
    std::string bestLang;
    float       bestConf = 0.0f;
    std::wstring bestVariant;
    DetectionResult bestFullResult = {};   // keeps softmax scores for margin gate

    // Hebrew Script Coverage gate: best virtual-Hebrew confidence across variants.
    float        heScriptVirtualConf = 0.0f;
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
                // Variant looks like Hebrew. Only apply if ONNX did NOT
                // strongly identify this variant as a non-Hebrew language.
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

        if (result.has_value() && result->confidence > bestConf) {
            // Skip languages the caller has excluded (user-rejected / case-signal)
            if (excludedLangs.count(result->language)) continue;
            bestConf       = result->confidence;
            bestLang       = result->language;
            bestVariant    = variant;
            bestFullResult = *result;
        }
    }

    // --- Apply Hebrew script gate if it beats ONNX ---
    if (!isFallback &&
        heScriptVirtualConf > bestConf &&
        !heScriptVariant.empty())
    {
        bestLang       = "he";
        bestConf       = heScriptVirtualConf;
        bestVariant    = heScriptVariant;
        bestFullResult = {};   // no real ONNX scores for this path
    }

    if (bestLang.empty()) {
        if (!isFallback) history.Update("", 0.0f);
        return std::nullopt;
    }

    // --- Extract runner-up from the winning variant's softmax scores ---
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

    // --- Look up per-pair switching parameters ---
    const auto& params = Config::GetParamsForPair(currentLang, bestLang);

    // Phrase mode: a space in any variant means ≥ 2 words → relax threshold.
    bool isPhrase = false;
    for (const auto& v : textVariants) {
        if (v.find(L' ') != std::wstring::npos) { isPhrase = true; break; }
    }
    float requiredConfidence = params.GetRequiredConfidence(numChars, isPhrase);

    // Per-pair min-chars check: the pair may require more characters
    // than the global minimum that was used as the early-out in the caller.
    if (static_cast<int>(numChars) < params.EarlyDetectionMinChars) {
        if (!isFallback) history.Update(bestLang, bestConf);
        return std::nullopt;
    }

    // --- Short-input extra confidence (false-positive guard) ---
    // For numChars in [EarlyDetectionMinChars, +2] raise the bar to suppress
    // FP on very short words.  Not applied in phrase mode (PhraseConfScale
    // already adjusts the threshold there).
    bool isShortInput = (static_cast<int>(numChars) <= params.EarlyDetectionMinChars + 2);
    if (isShortInput && !isPhrase && params.ShortInputExtraConf > 0.0f) {
        requiredConfidence = std::min(0.9999f,
                                      requiredConfidence + params.ShortInputExtraConf);
    }

    // --- Tier 2: Drop-one boosting (borderline zone) ---
    // If the confidence is close to but below the threshold, a single typo
    // character may be dragging it down.  Try removing each character once
    // and see if confidence jumps above the threshold.
    //
    // ── Iteration cap ───────────────────────────────────────────────────
    // Each dropped-char call is a full ONNX inference.  Running O(N) calls
    // inside the low-level keyboard hook risks exceeding LowLevelHooksTimeout
    // (~300 ms), which causes Windows to deliver the triggering key to the
    // app before the hook returns — the root cause of the "stray leading
    // character" bug.  We cap iterations at MAX_DROP_ONE_ITERS to keep the
    // total inference budget predictable and well under the OS timeout.
    static constexpr int MAX_DROP_ONE_ITERS = 6;
    if (Config::EnableTypoResilience &&
        bestConf < requiredConfidence &&
        bestConf >= requiredConfidence * params.BorderlineZoneFactor &&
        bestVariant.size() > 2)
    {
        int dropIters = 0;
        for (size_t i = 0; i < bestVariant.size(); ++i) {
            if (dropIters >= MAX_DROP_ONE_ITERS) break;
            ++dropIters;
            std::wstring dropped = bestVariant.substr(0, i)
                                 + bestVariant.substr(i + 1);
            auto res = detector.PredictLanguageWithConfidence(dropped);
            if (res.has_value() &&
                res->language == bestLang &&
                res->confidence > bestConf)
            {
                bestConf = res->confidence;
                margin   = bestConf - runnerUpConf;   // keep margin in sync
            }
        }
    }

    // --- History & consecutive-agreement gate ---
    // When this is a fallback call (excludedLangs non-empty), skip
    // history update and the agreement gate — the primary detection
    // already proved consistency, and the user explicitly rejected
    // the top choice.  We only require the confidence threshold.
    if (!isFallback) {
        // Update the history with whatever language won this round.
        history.Update(bestLang, bestConf);

        // --- Tier 1: Consecutive-agreement gate ---
        if (Config::EnableTypoResilience) {
            if (!history.IsConsistent(bestLang, params.ConsecutiveAgreementCount)) {
                return std::nullopt;
            }
        }
    }

    // --- Margin gate (false-positive guard) ---
    // A low top1/top2 gap means the model is split between two languages.
    // Skip on fallback, and skip when the Hebrew script gate fired (no real
    // ONNX scores → margin is meaningless, indicated by all-zero scores).
    {
        bool noScores = (bestFullResult.scores[0] == 0.0f &&
                         bestFullResult.scores[1] == 0.0f &&
                         bestFullResult.scores[2] == 0.0f &&
                         bestFullResult.scores[3] == 0.0f);
        if (!isFallback && !noScores && params.MinTop1Top2Margin > 0.0f &&
            margin < params.MinTop1Top2Margin) {
            return std::nullopt;
        }
    }

    if (bestConf >= requiredConfidence) {
        DetectionResult result = bestFullResult;
        result.language = bestLang;
        result.confidence = bestConf;
        return result;
    }

    return std::nullopt;
}
