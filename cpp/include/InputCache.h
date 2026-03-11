#pragma once

#include <vector>
#include <string>
#include <mutex>

class InputCache {
public:
    void PushChar(wchar_t ch, bool shiftHeld = false);
    void DelChar();
    void Clear();
    size_t Size() const;
    std::wstring GetText() const;
    std::vector<wchar_t> GetCache() const;

    // Returns true if Shift was held when the first character was typed
    bool WasFirstCharShifted() const;

private:
    mutable std::mutex mutex_;
    std::vector<wchar_t> cache_;
    std::vector<bool> shiftState_;
};

