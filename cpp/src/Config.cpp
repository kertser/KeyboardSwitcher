#include "Config.h"

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
    SwitchingParams DefaultParams = {
        /* EarlyDetectionMinChars  */ 3,
        /* FullConfidenceChars     */ 10,
        /* ConfidenceAtMinChars    */ 0.97f,
        /* ConfidenceAtMaxChars    */ 0.55f,
        /* ConsecutiveAgreementCount */ 2,
        /* BorderlineZoneFactor    */ 0.85f,
    };

    // ── Per-pair overrides ─────────────────────────────────────────
    // Different language pairs have different detection characteristics:
    //
    //  en↔ru : Distinct scripts (Latin vs Cyrillic). The model is very
    //          confident even on short input → standard defaults work.
    //
    //  en↔he : Distinct scripts (Latin vs Hebrew). Hebrew has no upper-
    //          case, so typed-on-English is always lowercase — slightly
    //          easier to confuse with short English. Require a bit more
    //          confidence at short lengths.
    //
    //  ru↔he : Both non-Latin. Physical key positions overlap less, and
    //          the model may need more context. Slightly higher min-chars
    //          and confidence give more room to discriminate.
    //
    // Pairs not listed here automatically fall back to DefaultParams.
    // To keep all pairs at defaults, simply leave PairOverrides empty.
    std::map<LangPair, SwitchingParams> PairOverrides = {
        // ── English ↔ Russian ─── (standard — same as default)
        { {"en", "ru"}, { 3, 10, 0.97f, 0.55f, 2, 0.85f } },
        { {"ru", "en"}, { 3, 10, 0.97f, 0.55f, 2, 0.85f } },

        // ── English ↔ Hebrew ─── (slightly stricter at short lengths)
        { {"en", "he"}, { 3, 10, 0.98f, 0.60f, 2, 0.85f } },
        { {"he", "en"}, { 3, 10, 0.98f, 0.60f, 2, 0.85f } },

        // ── Russian ↔ Hebrew ─── (both non-Latin; need more context)
        { {"ru", "he"}, { 4, 10, 0.98f, 0.60f, 2, 0.80f } },
        { {"he", "ru"}, { 4, 10, 0.98f, 0.60f, 2, 0.80f } },
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

    // Convenience wrapper using the global default
    float GetRequiredConfidence(size_t numChars) {
        return DefaultParams.GetRequiredConfidence(numChars);
    }

    // ── Typo resilience master toggle ──
    bool  EnableTypoResilience = true;

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

