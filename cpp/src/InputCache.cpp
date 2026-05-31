#include "InputCache.h"
#include <cwctype>

void InputCache::PushChar(wchar_t ch, bool shiftHeld) {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.push_back(ch);
    shiftState_.push_back(shiftHeld);
}

void InputCache::DelChar() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!cache_.empty()) {
        cache_.pop_back();
        shiftState_.pop_back();
    }
}

void InputCache::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_.clear();
    shiftState_.clear();
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

bool InputCache::WasFirstCharShifted() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shiftState_.empty()) return false;
    return shiftState_[0];
}

std::vector<bool> InputCache::GetShiftStates() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return shiftState_;
}

int InputCache::UpperCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    int count = 0;
    for (size_t i = 0; i < cache_.size(); ++i) {
        // Count a char as "uppercase intent" if the shift state was recorded OR
        // if the character itself is uppercase (belt-and-suspenders for cases
        // where capsOn is zeroed for Hebrew but the OS still produced a Latin
        // uppercase letter).
        if (iswalpha(cache_[i]) && (shiftState_[i] || iswupper(cache_[i])))
            ++count;
    }
    return count;
}

bool InputCache::HasInternalCapital() const {
    std::lock_guard<std::mutex> lock(mutex_);
    // Find the index of the first alphabetic character.
    // Any alpha char AFTER that index typed with shift = internal capital.
    bool foundFirst = false;
    for (size_t i = 0; i < cache_.size(); ++i) {
        if (!iswalpha(cache_[i])) continue;
        if (!foundFirst) {
            foundFirst = true;   // first alpha — skip; sentence capitals are OK
            continue;
        }
        if (shiftState_[i] || iswupper(cache_[i])) return true;  // internal capital found
    }
    return false;
}
