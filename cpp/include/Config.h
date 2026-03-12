#pragma once

#include <string>
#include <unordered_map>
#include <atomic>
#include <windows.h>

namespace Config {

    // Application version
    constexpr const wchar_t* VERSION = L"1.2.0";

    // Global flags (atomic for thread safety)
    extern std::atomic<bool> EnableSwitcher;
    extern std::atomic<bool> SEARCH;
    extern std::atomic<bool> SaveWindowState;
    extern std::atomic<bool> alt_pressed;
    extern std::string LastSetting;

    // ── Adaptive confidence curve ──
    // Detection starts at EarlyDetectionMinChars but requires very high
    // confidence.  As the user types more characters the required confidence
    // drops linearly until FullConfidenceChars, after which the floor applies.
    //
    //  chars < EarlyDetectionMinChars  → never detect
    //  chars = EarlyDetectionMinChars  → require ConfidenceAtMinChars  (0.97)
    //  chars = FullConfidenceChars     → require ConfidenceAtMaxChars  (0.55)
    //  chars > FullConfidenceChars     → require ConfidenceAtMaxChars  (floor)
    extern int   EarlyDetectionMinChars;   // default: 3
    extern int   FullConfidenceChars;      // default: 10
    extern float ConfidenceAtMinChars;     // default: 0.97
    extern float ConfidenceAtMaxChars;     // default: 0.55

    // Compute the required softmax confidence for a given number of typed chars.
    float GetRequiredConfidence(size_t numChars);

    // Language codes for ActivateKeyboardLayout / PostMessage
    extern const std::unordered_map<std::string, HKL> LANGUAGE_CODES;

    // Language ID mapping: LANGID -> language code string
    extern const std::unordered_map<LANGID, std::string> LANGUAGE_ID;

    // Get language string from LANGID
    std::string GetLanguageFromId(LANGID langId);

    // Get HKL from language string
    HKL GetHKLFromLanguage(const std::string& lang);

} // namespace Config

