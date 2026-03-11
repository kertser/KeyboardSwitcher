#pragma once

#include <string>
#include <unordered_map>
#include <mutex>
#include <windows.h>

class WindowTracker {
public:
    explicit WindowTracker(const std::string& defaultLanguage);

    // Get the language for the currently active window
    std::string GetActiveWindowLanguage();

    // Set the language for the currently active window
    void SetActiveWindowLanguage(const std::string& language);

    // Remove entries for closed windows
    void Cleanup();

private:
    mutable std::mutex mutex_;
    std::string defaultLanguage_;
    std::unordered_map<HWND, std::string> windowLanguages_;
};
