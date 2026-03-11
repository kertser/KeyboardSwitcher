#pragma once

#include <vector>
#include <string>
#include <mutex>

class InputCache {
public:
    void PushChar(wchar_t ch);
    void DelChar();
    void Clear();
    size_t Size() const;
    std::wstring GetText() const;
    std::vector<wchar_t> GetCache() const;

private:
    mutable std::mutex mutex_;
    std::vector<wchar_t> cache_;
};

