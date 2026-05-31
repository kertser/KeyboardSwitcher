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
    float SwitchingParams::GetRequiredConfidence(size_t numChars) const {
        int n = static_cast<int>(numChars);
        if (n < EarlyDetectionMinChars)
            return 1.1f;                       // impossible → no detection
        if (n >= FullConfidenceChars)
            return ConfidenceAtMaxChars;        // floor

        // Linear interpolation
        float t = static_cast<float>(n - EarlyDetectionMinChars)
                / static_cast<float>(FullConfidenceChars - EarlyDetectionMinChars);
        return ConfidenceAtMinChars + t * (ConfidenceAtMaxChars - ConfidenceAtMinChars);
    }

    // ── Global default parameters ──────────────────────────────────
    // EarlyDetectionMinChars = 4: wait for at least 4 alpha chars before
    // the detection engine fires.  Raising this from 3 eliminates most
    // false positives on very short input (≤3 chars) while still allowing
    // detection on 4-char words (the threshold tightens to ConfAtMinChars=0.99).
    SwitchingParams DefaultParams = {
        /* EarlyDetectionMinChars  */ 4,
        /* FullConfidenceChars     */ 15,
        /* ConfidenceAtMinChars    */ 0.99f,
        /* ConfidenceAtMaxChars    */ 0.70f,
        /* ConsecutiveAgreementCount */ 2,
        /* BorderlineZoneFactor    */ 0.85f,
    };

    // ── Per-pair overrides ─────────────────────────────────────────
    // All pairs use EarlyDetectionMinChars=4 (conservative; reduces FP
    // on short words like "he", "or", "yes").  The confidence floor
    // differs per direction based on observed false-positive rates.
    std::map<LangPair, SwitchingParams> PairOverrides = {
        // ── English ↔ Russian ───
        { {"en", "ru"}, { 4, 15, 0.99f, 0.70f, 2, 0.85f } },
        { {"ru", "en"}, { 4, 15, 0.99f, 0.70f, 2, 0.85f } },

        // ── English ↔ Hebrew ───
        // en→he: EarlyMin=3 (vs 4) adds ~4 pp TP at zero FP cost (sweep-validated).
        //        ConfAtMax=0.60 is sufficient; the 0.75 floor was over-conservative.
        //        Narrower borderline zone (0.88) still applies.
        { {"en", "he"}, { 3, 15, 0.99f, 0.60f, 2, 0.88f } },
        { {"he", "en"}, { 4, 15, 0.99f, 0.70f, 2, 0.85f } },

        // ── Russian ↔ Hebrew ───
        // ru→he: same tuning as en→he (sweep-validated: 165/200 TP, FP=0).
        { {"ru", "he"}, { 3, 15, 0.99f, 0.60f, 2, 0.88f } },
        { {"he", "ru"}, { 4, 15, 0.99f, 0.70f, 2, 0.80f } },
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

    // ── Case-signal Hebrew exclusion (Iteration 5 — A) ────────────
    // Exclude "he" when ≥ CaseExclusionMinCaps alpha chars were typed
    // with Shift intent, OR any internal (non-leading) alpha char was
    // capitalised.  Sentence-initial caps ("Hello") are not counted.
    bool EnableCaseBasedHeExclusion = true;
    int  CaseExclusionMinCaps       = 2;

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
