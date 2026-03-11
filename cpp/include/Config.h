#pragma once

#include <string>
#include <unordered_map>
#include <atomic>
#include <windows.h>

namespace Config {

    // Global flags (atomic for thread safety)
    extern std::atomic<bool> EnableSwitcher;
    extern std::atomic<bool> SEARCH;
    extern std::atomic<bool> SaveWindowState;
    extern std::atomic<bool> alt_pressed;
    extern std::string LastSetting;

    // Minimum number of characters typed before language detection triggers
    extern std::atomic<int> MinCharsBeforeDetection;

    // Minimum softmax confidence to trigger a switch (0.0 - 1.0)
    extern float MinConfidence;

    // Language codes for ActivateKeyboardLayout / PostMessage
    extern const std::unordered_map<std::string, HKL> LANGUAGE_CODES;

    // Language ID mapping: LANGID -> language code string
    extern const std::unordered_map<LANGID, std::string> LANGUAGE_ID;

    // Get language string from LANGID
    std::string GetLanguageFromId(LANGID langId);

    // Get HKL from language string
    HKL GetHKLFromLanguage(const std::string& lang);

} // namespace Config

