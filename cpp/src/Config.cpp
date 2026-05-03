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
    // Tuned with tune_confidence.py (sample=200, --phrases, consecutive=2)
    // on a test set that includes short Hebrew phrases (2-char first word).
    // Key changes vs previous: FullConfidenceChars 10→15, ConfAtMax 0.55→0.70.
    // Raising the floor reduces false positives for en↔ru without hurting
    // detection speed on Hebrew (model confidence on Hebrew is typically >0.97
    // even at 3 chars, well above the new 0.70 floor).
    SwitchingParams DefaultParams = {
        /* EarlyDetectionMinChars  */ 3,
        /* FullConfidenceChars     */ 15,
        /* ConfidenceAtMinChars    */ 0.99f,
        /* ConfidenceAtMaxChars    */ 0.70f,
        /* ConsecutiveAgreementCount */ 2,
        /* BorderlineZoneFactor    */ 0.85f,
    };

    // ── Per-pair overrides ─────────────────────────────────────────
    // Different language pairs have different detection characteristics:
    //
    //  en↔ru : Distinct scripts (Latin vs Cyrillic). Updated to the tuned
    //          default values (FullConf=15, floor=0.70).
    //
    //  x→he  : Hebrew Unicode (U+05D0–U+05EA) is completely distinct from
    //          both Latin and Cyrillic, so the model is highly confident even
    //          at 3 characters.  EarlyDetectionMinChars=3 enables correction of
    //          common short Hebrew words (כן, לא, גם, כי, אם, אך…) as well as
    //          short phrases whose first word is only 2 chars (מה שלומך etc.).
    //          Note: numChars is alphaCount (not raw cache size), so spaces and
    //          punctuation no longer inflate the count or lower confidence.
    //          The floor is raised to 0.72 (above the tuned 0.70 default) since
    //          Hebrew-script detection is very reliable and we want to prevent
    //          the rare case where a Cyrillic/Latin partial mis-classifies.
    //
    //  he→x  : Same reasoning in reverse — min=3 with a 0.70 floor.
    //
    // Pairs not listed here automatically fall back to DefaultParams.
    std::map<LangPair, SwitchingParams> PairOverrides = {
        // ── English ↔ Russian ─── (tuned defaults)
        { {"en", "ru"}, { 3, 15, 0.99f, 0.70f, 2, 0.85f } },
        { {"ru", "en"}, { 3, 15, 0.99f, 0.70f, 2, 0.85f } },

        // ── English ↔ Hebrew ───
        // en→he: user has English active but is typing Hebrew.
        //   MinChars=3 catches short Hebrew words and short+long phrases.
        //   ConfAtMin=0.99 keeps early FP very low; ConfAtMax=0.72 is the
        //   relaxed floor (slightly above global 0.70 for Hebrew robustness).
        { {"en", "he"}, { 3, 15, 0.99f, 0.72f, 2, 0.88f } },
        // he→en: user has Hebrew active but is typing English.
        { {"he", "en"}, { 3, 15, 0.99f, 0.70f, 2, 0.85f } },

        // ── Russian ↔ Hebrew ───
        // ru→he: Cyrillic and Hebrew are fully disjoint → same confidence as en→he.
        { {"ru", "he"}, { 3, 15, 0.99f, 0.72f, 2, 0.88f } },
        // he→ru: user has Hebrew active but is typing Russian.
        { {"he", "ru"}, { 3, 15, 0.99f, 0.70f, 2, 0.80f } },
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
