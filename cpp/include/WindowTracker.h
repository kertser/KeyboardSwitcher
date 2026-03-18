#pragma once

#include <string>
#include <unordered_map>
#include <mutex>
#include <windows.h>

class WindowTracker {
public:
    WindowTracker() = default;

    // Get the saved language for a window+tab combination.
    // titleHash distinguishes tabs within the same HWND (e.g., browser tabs).
    // Returns "" if the window/tab has not been visited yet.
    std::string GetLanguage(HWND hwnd, size_t titleHash) const;

    // Set the language for a window+tab combination.
    void SetLanguage(HWND hwnd, size_t titleHash, const std::string& language);

    // Remove entries for closed windows
    void Cleanup();

private:
    mutable std::mutex mutex_;
    // Outer map: HWND → (inner map: titleHash → language)
    std::unordered_map<HWND, std::unordered_map<size_t, std::string>> windowLanguages_;
};
