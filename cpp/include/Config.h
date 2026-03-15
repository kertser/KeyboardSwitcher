#pragma once

#include <string>
#include <unordered_map>
#include <atomic>
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

    // ── Typo resilience ──
    // Consecutive-agreement: require N consecutive keystrokes to predict the
    // same language before triggering a switch.  Prevents a single mis-typed
    // key from causing a false-positive switch.
    extern int   ConsecutiveAgreementCount;     // default: 2
    // Drop-one boosting: if confidence falls in the borderline zone
    // [threshold * BorderlineZoneFactor .. threshold], try removing each
    // character once and re-run the model.  If the best drop-one confidence
    // exceeds the threshold the detection succeeds — prevents a single typo
    // from suppressing a true detection (false-negative).
    extern float BorderlineZoneFactor;          // default: 0.85
    extern bool  EnableTypoResilience;          // master toggle (default: true)

    // Language codes for ActivateKeyboardLayout / PostMessage
    extern const std::unordered_map<std::string, HKL> LANGUAGE_CODES;

    // Language ID mapping: LANGID -> language code string
    extern const std::unordered_map<LANGID, std::string> LANGUAGE_ID;

    // Get language string from LANGID
    std::string GetLanguageFromId(LANGID langId);

    // Get HKL from language string
    HKL GetHKLFromLanguage(const std::string& lang);

} // namespace Config

