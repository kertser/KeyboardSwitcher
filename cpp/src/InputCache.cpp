#include "InputCache.h"

void InputCache::PushChar(wchar_t ch) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.push_back(ch);
}

void InputCache::DelChar() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!cache_.empty()) {
        cache_.pop_back();
    }
}

void InputCache::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
}

size_t InputCache::Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_.size();
}

std::wstring InputCache::GetText() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::wstring(cache_.begin(), cache_.end());
}

std::vector<wchar_t> InputCache::GetCache() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return cache_;
}
