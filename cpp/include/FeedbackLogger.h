#pragma once

#include <string>
#include <atomic>
#include <set>
#include <map>


// ============================================================
// FeedbackLogger — logs rejected corrections and maintains a
// per-user exception list of corrected texts that should never
// be applied again.
//
// Instead of adjusting global confidence thresholds (which
// affects ALL words for a language pair), we record the specific
// corrected texts that the user rejected.  Before applying a
// correction, the caller checks IsException() to see if the
// proposed corrected text was previously rejected.
//
// Storage: %APPDATA%/KeyboardSwitcher/feedback.jsonl   (events)
//          %APPDATA%/KeyboardSwitcher/user_prefs.json  (exceptions + prefs)
//
// All public methods are called from the main (UI) thread only.
// ============================================================
namespace Feedback {

    // ── Feedback entry ──────────────────────────────────────
    struct Entry {
        std::string type;            // "esc_undo" | "manual_override"
        std::wstring originalText;   // what the user typed (pre-correction)
        std::wstring correctedText;  // what the model replaced it with
        std::string  fromLang;       // keyboard layout before correction
        std::string  toLang;         // language the model switched to
        std::string  actualLang;     // language the user actually wanted
        float        confidence;     // model softmax score at detection time
        int          numChars;       // cache size when detection fired
    };

    // ── Pending rejection (Pattern B state machine) ─────────
    struct PendingRejection {
        std::wstring originalText;
        std::wstring correctedText;
        std::string  fromLang;
        std::string  toLang;
        float        confidence = 0.0f;
        int          numChars   = 0;
        unsigned long tick      = 0;       // GetTickCount() when set
        bool         active     = false;
    };

    // Time window for Pattern B: backspace-past-boundary then manual switch
    constexpr unsigned long REJECTION_WINDOW_MS = 5000;

    // Time window for direct rejection: manual switch away from
    // corrected language shortly after a correction fired.
    constexpr unsigned long SWITCH_AWAY_WINDOW_MS = 10000;

    // ── Public API ──────────────────────────────────────────

    // Master opt-in toggle for JSONL logging.
    // Exception recording is always active regardless of this flag.
    extern std::atomic<bool> LoggingEnabled;

    // Initialise: create %APPDATA% dir, load user_prefs.json.
    void Init();

    // Log a feedback event to feedback.jsonl (if LoggingEnabled).
    void LogEvent(const Entry& entry);

    // Record a rejected correction: adds correctedText to the
    // exception list for toLang and persists immediately.
    void AddException(const std::string& toLang,
                      const std::wstring& correctedText);

    // Check whether the proposed correctedText for the given
    // target language was previously rejected by the user.
    bool IsException(const std::string& toLang,
                     const std::wstring& correctedText);

    // Record a learned correction override: when the user is on
    // keyboard `currentLang` and types `text`, they actually want
    // `targetLang`.  Derived from rejection feedback (the reverse
    // of a false correction).
    void AddOverride(const std::string& currentLang,
                     const std::wstring& text,
                     const std::string& targetLang);

    // Check if there is a learned override for `text` typed on
    // `currentLang`.  Returns the target language, or "" if none.
    std::string GetOverride(const std::string& currentLang,
                            const std::wstring& text);

    // Persist the current state to user_prefs.json.
    void SavePrefs();

    // Reset all exceptions and delete the JSONL file.
    void ResetAll();

    // Toggle logging and persist the preference.
    void SetLoggingEnabled(bool enabled);

    // Get the %APPDATA%/KeyboardSwitcher directory.
    std::wstring GetDataDir();

}  // namespace Feedback

