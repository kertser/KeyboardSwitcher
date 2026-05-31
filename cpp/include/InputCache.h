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

    // Returns the full shift-state vector (one bool per cached character).
    // Used by the correction path to restore per-character capitalisation
    // (e.g. "IDF" → shift[0..2]=true → uppercased after layout conversion).
    std::vector<bool> GetShiftStates() const;

    // Number of alphabetic characters typed with Shift/CapsLock intent.
    // Useful for detecting ALL-CAPS abbreviations (FPS, USB, NATO …).
    int UpperCount() const;

    // True when any alphabetic character at position ≥ 1 (relative to the
    // first alpha character) was typed with Shift/CapsLock intent.
    // Catches internal capitals: iPhone, CamelCase, etc.
    bool HasInternalCapital() const;

private:
    mutable std::mutex mutex_;
    std::vector<wchar_t> cache_;
    std::vector<bool> shiftState_;
};
