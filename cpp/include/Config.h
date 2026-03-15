#pragma once

#include <string>
#include <unordered_map>
#include <map>
#include <atomic>
#include <utility>
#include <windows.h>

namespace Config {

    // Application version — injected by CMake from project(VERSION ...)
    // Change the version only in CMakeLists.txt; it propagates everywhere.
#ifndef APP_VERSION_STRING
#define APP_VERSION_STRING L"0.0.0"   // fallback if built without CMake
#endif
    constexpr const wchar_t* VERSION = APP_VERSION_STRING;

    // Global flags (atomic for thread safety)
    extern std::atomic<bool> EnableSwitcher;
    extern std::atomic<bool> SEARCH;
    extern std::atomic<bool> SaveWindowState;
    extern std::atomic<bool> alt_pressed;
    extern std::string LastSetting;

    // ================================================================
    // Per-language-pair adaptive switching parameters
    // ================================================================
    // Groups all tunable detection parameters. Each (from→to) language
    // pair can have its own set of thresholds. A global default is used
    // as the fallback for any pair that is not explicitly overridden.
    //
    //  chars < EarlyDetectionMinChars  → never detect
    //  chars = EarlyDetectionMinChars  → require ConfidenceAtMinChars
    //  chars = FullConfidenceChars     → require ConfidenceAtMaxChars
    //  chars > FullConfidenceChars     → require ConfidenceAtMaxChars (floor)
    // ================================================================
    struct SwitchingParams {
        int   EarlyDetectionMinChars;   // min chars before detection fires
        int   FullConfidenceChars;      // chars at which confidence floor kicks in
        float ConfidenceAtMinChars;     // near-certainty required at few chars
        float ConfidenceAtMaxChars;     // relaxed confidence after enough chars
        int   ConsecutiveAgreementCount; // consecutive keystrokes must agree
        float BorderlineZoneFactor;     // drop-one boosting triggers in [threshold*factor, threshold]

        // Compute the required softmax confidence for a given text length
        float GetRequiredConfidence(size_t numChars) const;
    };

    // Global default parameters (used as fallback for unlisted pairs)
    extern SwitchingParams DefaultParams;

    // Per-pair overrides keyed by (from_lang, to_lang).
    // Example: PairOverrides[{"en","ru"}] uses tuned params for English→Russian.
    // Pairs not in this map fall back to DefaultParams.
    using LangPair = std::pair<std::string, std::string>;
    extern std::map<LangPair, SwitchingParams> PairOverrides;

    // Look up the switching parameters for a specific (from→to) pair.
    // Returns the pair-specific override if present, otherwise DefaultParams.
    const SwitchingParams& GetParamsForPair(const std::string& fromLang,
                                            const std::string& toLang);

    // ── Legacy global accessors (kept for the tray-menu "Min Chars" control) ──
    // These read/write DefaultParams directly.
    extern int&   EarlyDetectionMinChars;   // alias into DefaultParams
    extern int&   FullConfidenceChars;
    extern float& ConfidenceAtMinChars;
    extern float& ConfidenceAtMaxChars;
    extern int&   ConsecutiveAgreementCount;
    extern float& BorderlineZoneFactor;

    // Convenience: compute required confidence using the global default params
    float GetRequiredConfidence(size_t numChars);

    // Master toggle for typo resilience (applies to all pairs)
    extern bool  EnableTypoResilience;

    // Language codes for ActivateKeyboardLayout / PostMessage
    extern const std::unordered_map<std::string, HKL> LANGUAGE_CODES;

    // Language ID mapping: LANGID -> language code string
    extern const std::unordered_map<LANGID, std::string> LANGUAGE_ID;

    // Get language string from LANGID
    std::string GetLanguageFromId(LANGID langId);

    // Get HKL from language string
    HKL GetHKLFromLanguage(const std::string& lang);

} // namespace Config

