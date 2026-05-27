#include "Config.h"
#include <cstdio>
#include <string>

namespace Config {

    std::atomic<bool> EnableSwitcher{true};
    std::atomic<bool> SEARCH{false};
    std::atomic<bool> SaveWindowState{true};
    std::atomic<bool> alt_pressed{false};
    std::string LastSetting = "en";

    // ================================================================
    // SwitchingParams – per-pair adaptive confidence curve
    // ================================================================
    float SwitchingParams::GetRequiredConfidence(size_t numChars, bool isPhrase) const {
        int n = static_cast<int>(numChars);
        float base;
        if (n < EarlyDetectionMinChars)
            return 1.1f;                       // impossible → no detection
        if (n >= FullConfidenceChars)
            base = ConfidenceAtMaxChars;        // floor
        else {
            // Linear interpolation
            float t = static_cast<float>(n - EarlyDetectionMinChars)
                    / static_cast<float>(FullConfidenceChars - EarlyDetectionMinChars);
            base = ConfidenceAtMinChars + t * (ConfidenceAtMaxChars - ConfidenceAtMinChars);
        }
        // Phrase mode: lower the threshold when text is a multi-word phrase.
        // The margin gate (MinTop1Top2Margin) already ensures signal quality,
        // so reducing the absolute threshold here is safe.
        if (isPhrase && PhraseConfScale < 1.0f)
            base *= PhraseConfScale;
        return base;
    }

    // ── Global default parameters ──────────────────────────────────
    // Field order: EarlyMin, FullConf, ConfAtMin, ConfAtMax,
    //              ConsecAgree, BorderlineFactor,
    //              MinMargin, VariantAgree,
    //              TrendWindow, MinSteps, MinSlope,
    //              ShortExtraConf, PhraseConfScale,
    //              HebrewScriptVC, HebrewScriptThresh,
    //              PersistMinAvg, PersistMinSteps,
    //              WeakScoreClassIdx, WeakScoreMinAvg, WeakScoreWindow
    SwitchingParams DefaultParams = {
        /* EarlyDetectionMinChars    */ 3,
        /* FullConfidenceChars       */ 15,
        /* ConfidenceAtMinChars      */ 0.99f,
        /* ConfidenceAtMaxChars      */ 0.70f,
        /* ConsecutiveAgreementCount */ 2,
        /* BorderlineZoneFactor      */ 0.85f,
        /* MinTop1Top2Margin         */ 0.05f,
        /* VariantAgreementCount     */ 0,
        /* TrendWindowSize           */ 4,
        /* MinStableSteps            */ 2,
        /* MinTrendSlope             */ 0.0f,
        /* ShortInputExtraConf       */ 0.02f,
        /* PhraseConfScale           */ 1.0f,
        /* HebrewScriptVirtualConf   */ 0.0f,   // disabled for non-→he pairs
        /* HebrewScriptCovThreshold  */ 0.90f,
        /* PersistentMinAvgConf      */ 0.0f,   // disabled by default
        /* PersistentMinSteps        */ 0,
        /* WeakScoreClassIdx         */ -1,     // disabled by default
        /* WeakScoreMinAvg           */ 0.0f,
        /* WeakScoreWindow           */ 0,
    };

    // ── Per-pair overrides ─────────────────────────────────────────
    // Column order matches SwitchingParams field declaration order.
    //
    // Notes on VariantAgreementCount for →he pairs:
    //   Hebrew detection typically produces exactly ONE valid variant.
    //   Requiring VariantAgree>0 for →he would block virtually all correct
    //   detections.  Keep it at 0 (disabled) for →he.
    //
    // Notes on HebrewScriptVirtualConf:
    //   Only meaningful for →he pairs.  The script gate assigns a virtual
    //   "he" confidence equal to (coverage × HebrewScriptVirtualConf) when a
    //   variant is ≥ HebrewScriptCoverageThreshold Hebrew Unicode chars AND
    //   ONNX did not strongly claim a non-Hebrew language for that variant.
    //   Virtual conf must beat the bestConf from ONNX to override.
    //   0.78 means a 100%-Hebrew variant gets virtual conf 0.78 — enough to
    //   beat EN at 56% (כך רציתי) but not enough to beat RU at 99.8%.
    //
    // Notes on WeakScoreGate for →he:
    //   classIdx=2 tracks the "he" softmax score even when "he" is not top-1.
    //   WeakScoreMinAvg=0.28 over 6 frames fires when Hebrew receives a weak
    //   but persistent signal that is unmistakably above the random baseline.
    std::map<LangPair, SwitchingParams> PairOverrides = {
        // ── English ↔ Russian ───────────────────────────────────────────
        //  EMin FConf CAt0  CAt1  Agr BLF   Mrg  VA  TWin MnSt Slp  SXC  PCS
        //  HeSVC HeSCT  PerAv PrSt  WSC   WSA   WSW
        { {"en", "ru"}, {
            3,    15, 0.99f, 0.70f,  2, 0.85f, 0.05f, 0,   4,   2, 0.0f, 0.02f, 1.00f,
            0.0f, 0.90f, 0.0f, 0,  -1, 0.0f, 0 } },
        { {"ru", "en"}, {
            3,    15, 0.99f, 0.70f,  2, 0.85f, 0.05f, 0,   4,   2, 0.0f, 0.02f, 1.00f,
            0.0f, 0.90f, 0.0f, 0,  -1, 0.0f, 0 } },

        // ── English → Hebrew ────────────────────────────────────────────
        // PhraseConfScale=0.72  (threshold reduced by 28% for multi-word phrases)
        // HebrewScriptVC=0.78   (100%-Hebrew variant → virtual conf 0.78)
        // PersistentGate=0.52/5 (5 consecutive steps at avg≥52% → fire)
        // WeakScoreGate: track Hebrew class (idx=2); avg≥0.28 over 7 frames
        { {"en", "he"}, {
            3,    15, 0.97f, 0.65f,  2, 0.88f, 0.10f, 0,   4,   2, 0.0f, 0.0f,  0.72f,
            0.78f, 0.90f, 0.52f, 5,   2, 0.28f, 7 } },

        // ── Hebrew → English ────────────────────────────────────────────
        { {"he", "en"}, {
            3,    15, 0.99f, 0.70f,  2, 0.85f, 0.05f, 0,   4,   2, 0.0f, 0.02f, 1.00f,
            0.0f, 0.90f, 0.0f, 0,  -1, 0.0f, 0 } },

        // ── Russian → Hebrew ────────────────────────────────────────────
        // Same Iteration-3 gates as en→he.
        { {"ru", "he"}, {
            3,    15, 0.97f, 0.65f,  2, 0.88f, 0.10f, 0,   4,   2, 0.0f, 0.0f,  0.72f,
            0.78f, 0.90f, 0.52f, 5,   2, 0.28f, 7 } },

        // ── Hebrew → Russian ────────────────────────────────────────────
        { {"he", "ru"}, {
            3,    15, 0.99f, 0.70f,  2, 0.80f, 0.05f, 0,   4,   2, 0.0f, 0.02f, 1.00f,
            0.0f, 0.90f, 0.0f, 0,  -1, 0.0f, 0 } },
    };

    const SwitchingParams& GetParamsForPair(const std::string& fromLang,
                                            const std::string& toLang) {
        auto it = PairOverrides.find({fromLang, toLang});
        if (it != PairOverrides.end())
            return it->second;
        return DefaultParams;
    }

    // ── Legacy global aliases (point into DefaultParams) ───────────
    int&   EarlyDetectionMinChars   = DefaultParams.EarlyDetectionMinChars;
    int&   FullConfidenceChars      = DefaultParams.FullConfidenceChars;
    float& ConfidenceAtMinChars     = DefaultParams.ConfidenceAtMinChars;
    float& ConfidenceAtMaxChars     = DefaultParams.ConfidenceAtMaxChars;
    int&   ConsecutiveAgreementCount = DefaultParams.ConsecutiveAgreementCount;
    float& BorderlineZoneFactor     = DefaultParams.BorderlineZoneFactor;

    float GetRequiredConfidence(size_t numChars) {
        return DefaultParams.GetRequiredConfidence(numChars);
    }

    // ── Typo resilience master toggle ──
    bool  EnableTypoResilience = true;

    // ── Trend Gate / Variant Consensus feature flags ────────────────
    bool  EnableTrendGate       = true;  // compute & use trend gate
    bool  EnableTrendGateBlock  = true;  // actually block (false = log-only)
    bool  EnableVariantConsensus = true; // variant-agreement gate

    // ── Iteration 3 feature flags ───────────────────────────────────
    bool  EnableHebrewScriptGate    = true;  // Hebrew Unicode coverage gate
    bool  EnablePersistentConfGate  = true;  // flat persistent confidence gate
    bool  EnableWeakScoreGate       = true;  // cumulative weak class-score gate
    bool  AddOriginalTextAsVariant  = true;  // prepend raw input to variants

    // ── New parameters (Iteration 1) ───────────────────────────────
    // Minimum known-char count for ONNX inference.  2 keeps very short
    // but legitimate inputs detectable while filtering pure-symbol noise.
    int  MinKnownCharsForInference = 2;

    // Protect English layout in file-open/save dialogs against False
    // auto-switches to Hebrew/Russian.  Default: enabled.
    bool DisableAutoSwitchFromEnglishInFileDialogs = true;

    // ── Skip-reason counters ────────────────────────────────────────
    SkipCounters Guards;

    void SkipCounters::Reset() noexcept {
        skipEmptyAfterTokenize  = 0;
        skipLowKnownChars       = 0;
        skipLowAlpha            = 0;
        skipUrlOrPath           = 0;
        skipFileDialogEnProtection = 0;
        correctionsApplied      = 0;
    }

    std::string SkipCounters::Summary() const {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "guards: emptyTok=%u lowKnown=%u lowAlpha=%u urlPath=%u fileDialog=%u corrections=%u",
            skipEmptyAfterTokenize.load(),
            skipLowKnownChars.load(),
            skipLowAlpha.load(),
            skipUrlOrPath.load(),
            skipFileDialogEnProtection.load(),
            correctionsApplied.load());
        return buf;
    }

    // Language codes: HKL values matching the Python project
    const std::unordered_map<std::string, HKL> LANGUAGE_CODES = {
        {"en", reinterpret_cast<HKL>(static_cast<uintptr_t>(0x04090409))},
        {"ru", reinterpret_cast<HKL>(static_cast<uintptr_t>(0x04190419))},
        {"he", reinterpret_cast<HKL>(static_cast<uintptr_t>(0xF03D040D))}
    };

    // LANGID -> language string
    const std::unordered_map<LANGID, std::string> LANGUAGE_ID = {
        {1033, "en"},  // English (United States)
        {1049, "ru"},  // Russian
        {1037, "he"}   // Hebrew (Israel)
    };

    std::string GetLanguageFromId(LANGID langId) {
        auto it = LANGUAGE_ID.find(langId);
        if (it != LANGUAGE_ID.end()) {
            return it->second;
        }
        return "en"; // default
    }

    HKL GetHKLFromLanguage(const std::string& lang) {
        auto it = LANGUAGE_CODES.find(lang);
        if (it != LANGUAGE_CODES.end()) {
            return it->second;
        }
        return LANGUAGE_CODES.at("en"); // default
    }

} // namespace Config
