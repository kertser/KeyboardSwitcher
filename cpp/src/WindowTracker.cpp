#include "WindowTracker.h"
#include <algorithm>
#include <vector>

std::string WindowTracker::GetLanguage(HWND hwnd, size_t titleHash) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto wit = windowLanguages_.find(hwnd);
    if (wit == windowLanguages_.end()) return "";
    auto tit = wit->second.find(titleHash);
    if (tit == wit->second.end()) return "";
    // Touch the entry so it stays alive
    tit->second.lastAccess = Clock::now();
    return tit->second.language;
}

void WindowTracker::SetLanguage(HWND hwnd, size_t titleHash, const std::string& language) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& tabs = windowLanguages_[hwnd];
    tabs[titleHash] = { language, Clock::now() };

    // Cap: if too many tab entries for this HWND, evict the oldest
    if (tabs.size() > MAX_TABS_PER_HWND) {
        // Find the entry with the oldest lastAccess
        auto oldest = tabs.begin();
        for (auto it = tabs.begin(); it != tabs.end(); ++it) {
            if (it->second.lastAccess < oldest->second.lastAccess)
                oldest = it;
        }
        if (oldest->first != titleHash)  // don't evict the one we just set
            tabs.erase(oldest);
    }
}

void WindowTracker::Cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto now = Clock::now();
    auto maxAge = std::chrono::seconds(MAX_TAB_AGE_SEC);

    for (auto wit = windowLanguages_.begin(); wit != windowLanguages_.end();) {
        // Remove entire HWND if the window is closed
        if (!IsWindow(wit->first)) {
            wit = windowLanguages_.erase(wit);
            continue;
        }

        // Expire stale tab entries within live windows
        auto& tabs = wit->second;
        for (auto tit = tabs.begin(); tit != tabs.end();) {
            if ((now - tit->second.lastAccess) > maxAge) {
                tit = tabs.erase(tit);
            } else {
                ++tit;
            }
        }

        // If all tabs expired, remove the HWND entry too
        if (tabs.empty()) {
            wit = windowLanguages_.erase(wit);
        } else {
            ++wit;
        }
    }
}
