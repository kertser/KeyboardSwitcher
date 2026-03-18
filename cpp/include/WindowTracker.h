#pragma once

#include <string>
#include <unordered_map>
#include <mutex>
#include <chrono>
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

    // Remove entries for closed windows and expire stale tab entries.
    void Cleanup();

private:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    struct TabEntry {
        std::string      language;
        mutable TimePoint lastAccess;  // updated on read (GetLanguage is const)
    };

    // Max age before a tab entry is expired (seconds)
    static constexpr int MAX_TAB_AGE_SEC = 600;    // 10 minutes
    // Max tab entries per HWND (prevents unbounded growth from title drift)
    static constexpr size_t MAX_TABS_PER_HWND = 20;

    mutable std::mutex mutex_;
    // Outer map: HWND → (inner map: titleHash → TabEntry)
    std::unordered_map<HWND, std::unordered_map<size_t, TabEntry>> windowLanguages_;
};
