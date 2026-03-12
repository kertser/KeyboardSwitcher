#include "Config.h"

namespace Config {

    std::atomic<bool> EnableSwitcher{true};
    std::atomic<bool> SEARCH{false};
    std::atomic<bool> SaveWindowState{true};
    std::atomic<bool> alt_pressed{false};
    std::string LastSetting = "en";

    // ── Adaptive confidence curve parameters ──
    int   EarlyDetectionMinChars = 3;      // earliest detection fires here
    int   FullConfidenceChars    = 10;     // confidence floor kicks in here
    float ConfidenceAtMinChars   = 0.97f;  // near-certainty at few chars
    float ConfidenceAtMaxChars   = 0.55f;  // relaxed after enough chars

    // ── Typo resilience defaults ──
    int   ConsecutiveAgreementCount = 2;
    float BorderlineZoneFactor      = 0.85f;
    bool  EnableTypoResilience      = true;

    float GetRequiredConfidence(size_t numChars) {
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

