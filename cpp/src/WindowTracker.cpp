#include "WindowTracker.h"
#include "Config.h"

WindowTracker::WindowTracker(const std::string& defaultLanguage)
    : defaultLanguage_(defaultLanguage)
{
}

std::string WindowTracker::GetActiveWindowLanguage() {
    std::lock_guard<std::mutex> lock(mutex_);
    HWND hwnd = GetForegroundWindow();
    if (hwnd == nullptr) {
        return defaultLanguage_;
    }
    auto it = windowLanguages_.find(hwnd);
    if (it != windowLanguages_.end()) {
        return it->second;
    }
    // Window not tracked yet, assign default
    windowLanguages_[hwnd] = defaultLanguage_;
    return defaultLanguage_;
}

void WindowTracker::SetActiveWindowLanguage(const std::string& language) {
    std::lock_guard<std::mutex> lock(mutex_);
    HWND hwnd = GetForegroundWindow();
    if (hwnd != nullptr) {
        windowLanguages_[hwnd] = language;
    }
}

void WindowTracker::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto it = windowLanguages_.begin(); it != windowLanguages_.end();) {
        if (!IsWindow(it->first)) {
            it = windowLanguages_.erase(it);
        } else {
            ++it;
        }
    }
}

