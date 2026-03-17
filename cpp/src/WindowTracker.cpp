#include "WindowTracker.h"
#include "Config.h"

std::string WindowTracker::GetActiveWindowLanguage() {
    std::lock_guard<std::mutex> lock(mutex_);
    HWND hwnd = GetForegroundWindow();
    if (hwnd == nullptr) {
        return "";
    }
    auto it = windowLanguages_.find(hwnd);
    if (it != windowLanguages_.end()) {
        return it->second;
    }
    // Window not tracked yet — return empty so callers know it's unvisited
    return "";
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

