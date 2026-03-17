#pragma once

#include <string>
#include <unordered_map>
#include <mutex>
#include <windows.h>

class WindowTracker {
public:
    WindowTracker() = default;

    // Get the language for the currently active window.
    // Returns "" if the window has not been visited yet.
    std::string GetActiveWindowLanguage();

    // Set the language for the currently active window
    void SetActiveWindowLanguage(const std::string& language);

    // Remove entries for closed windows
    void Cleanup();

private:
    mutable std::mutex mutex_;
    std::unordered_map<HWND, std::string> windowLanguages_;
};
