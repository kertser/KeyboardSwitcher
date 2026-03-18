#include "WindowTracker.h"

std::string WindowTracker::GetLanguage(HWND hwnd, size_t titleHash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto wit = windowLanguages_.find(hwnd);
    if (wit == windowLanguages_.end()) return "";
    auto tit = wit->second.find(titleHash);
    if (tit == wit->second.end()) return "";
    return tit->second;
}

void WindowTracker::SetLanguage(HWND hwnd, size_t titleHash, const std::string& language) {
    std::lock_guard<std::mutex> lock(mutex_);
    windowLanguages_[hwnd][titleHash] = language;
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
