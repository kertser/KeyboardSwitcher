#pragma once

#include <string>
#include <unordered_map>
#include <map>
#include <atomic>
#include <utility>
#include <windows.h>

namespace Config {

    // Application version — injected by CMake from project(VERSION ...)
    // Change the version only in CMakeLists.txt; it propagates everywhere.
#ifndef APP_VERSION_STRING
#define APP_VERSION_STRING L"0.0.0"   // fallback if built without CMake
#endif
    constexpr const wchar_t* VERSION = APP_VERSION_STRING;

    // Global flags (atomic for thread safety)
    extern std::atomic<bool> EnableSwitcher;
    extern std::atomic<bool> SEARCH;
    extern std::atomic<bool> SaveWindowState;
    extern std::atomic<bool> alt_pressed;
    extern std::string LastSetting;

    // ================================================================
    // Per-language-pair adaptive switching parameters
    // ================================================================
    // Groups all tunable detection parameters. Each (from→to) language
    // pair can have its own set of thresholds. A global default is used
    // as the fallback for any pair that is not explicitly overridden.
    //
    //  chars < EarlyDetectionMinChars  → never detect
    //  chars = EarlyDetectionMinChars  → require ConfidenceAtMinChars
    //  chars = FullConfidenceChars     → require ConfidenceAtMaxChars
    //  chars > FullConfidenceChars     → require ConfidenceAtMaxChars (floor)
    // ================================================================
    struct SwitchingParams {
        int   EarlyDetectionMinChars;   // min chars before detection fires
        int   FullConfidenceChars;      // chars at which confidence floor kicks in
        float ConfidenceAtMinChars;     // near-certainty required at few chars
        float ConfidenceAtMaxChars;     // relaxed confidence after enough chars
        int   ConsecutiveAgreementCount; // consecutive keystrokes must agree
        float BorderlineZoneFactor;     // drop-one boosting triggers in [threshold*factor, threshold]

        // ── Signal-quality gates (ported from 1.3.0) ──────────────────
        // MinTop1Top2Margin: required gap between the top-1 and top-2 softmax
        //   probabilities. 0.0 = disabled. Blocks corrections when the model
        //   is split between two languages (a major false-positive source).
        float MinTop1Top2Margin;

        // ShortInputExtraConf: extra confidence added on top of the adaptive
        //   threshold when numChars is in [EarlyDetectionMinChars, +2].
        //   Tightens the bar on very short input. 0.0 = disabled.
        float ShortInputExtraConf;

        // PhraseConfScale: when the detection text contains a space (multi-word
        //   phrase), multiply the required confidence by this factor.
        //   1.0 = no change; 0.80 = 20 % lower threshold for phrases. Improves
        //   recall on short Hebrew phrases whose individual words are borderline.
        float PhraseConfScale;

        // HebrewScriptVirtualConf: when a layout-conversion variant is
        //   >= HebrewScriptCoverageThreshold Hebrew Unicode chars (U+05D0–U+05EA)
        //   AND the ONNX model did NOT strongly claim a non-Hebrew language for
        //   that variant, assign this virtual confidence to "he". If it beats the
        //   ONNX winner the gate overrides it. 0.0 = disabled (non-→he pairs).
        float HebrewScriptVirtualConf;

        // HebrewScriptCoverageThreshold: minimum Hebrew-character fraction
        //   required to trigger the script gate (0.90 = 90 % of alpha chars).
        float HebrewScriptCoverageThreshold;

        // ── Incumbent-advantage gate (FP guard) ───────────────────────
        // SwitchBiasMargin: extra confidence the best SWITCH candidate must
        //   have over the strongest "stay on the current language" signal
        //   (the max ONNX confidence assigned to currentLang across all
        //   variants — usually the identity variant).  A switch fires only
        //   when bestConf >= incumbentConf + SwitchBiasMargin.  This is the
        //   single biggest structural FP suppressor: a genuine current-language
        //   word produces a strong incumbent signal that an accidental
        //   cross-layout variant cannot beat.  0.0 = disabled.
        float SwitchBiasMargin;

        // ── Hebrew weak-signal gates (FN guard, →he pairs only) ───────
        // These let detection fire on "flat-signal" Hebrew phrases that the
        // model scores consistently but below the adaptive threshold.  They
        // are OR-alternatives to the consecutive-agreement + confidence gate
        // and only apply when bestLang == "he".
        //
        // Persistent Moderate Confidence Gate:
        //   fires when the last PersistentMinSteps history frames ALL had
        //   top-1 == "he" and the average top-1 confidence >= PersistentMinAvgConf.
        float PersistentMinAvgConf;
        int   PersistentMinSteps;
        // Cumulative Weak Score Gate:
        //   fires when the average softmax score for class WeakScoreClassIdx
        //   (2 = Hebrew) over the last WeakScoreWindow frames >= WeakScoreMinAvg,
        //   even when "he" was never the per-frame argmax winner.
        int   WeakScoreClassIdx;
        float WeakScoreMinAvg;
        int   WeakScoreWindow;

        // Compute the required softmax confidence for a given text length.
        // isPhrase: true when the detection text contains a space — PhraseConfScale
        //   is then applied to lower the threshold.
        float GetRequiredConfidence(size_t numChars, bool isPhrase = false) const;
    };

    // Global default parameters (used as fallback for unlisted pairs)
    extern SwitchingParams DefaultParams;

    // Per-pair overrides keyed by (from_lang, to_lang).
    // Example: PairOverrides[{"en","ru"}] uses tuned params for English→Russian.
    // Pairs not in this map fall back to DefaultParams.
    using LangPair = std::pair<std::string, std::string>;
    extern std::map<LangPair, SwitchingParams> PairOverrides;

    // Look up the switching parameters for a specific (from→to) pair.
    // Returns the pair-specific override if present, otherwise DefaultParams.
    const SwitchingParams& GetParamsForPair(const std::string& fromLang,
                                            const std::string& toLang);

    // ── Legacy global accessors (kept for the tray-menu "Min Chars" control) ──
    // These read/write DefaultParams directly.
    extern int&   EarlyDetectionMinChars;   // alias into DefaultParams
    extern int&   FullConfidenceChars;
    extern float& ConfidenceAtMinChars;
    extern float& ConfidenceAtMaxChars;
    extern int&   ConsecutiveAgreementCount;
    extern float& BorderlineZoneFactor;

    // Convenience: compute required confidence using the global default params
    float GetRequiredConfidence(size_t numChars);

    // Master toggle for typo resilience (applies to all pairs)
    extern bool  EnableTypoResilience;

    // ── Hebrew Script Coverage Gate (ported from 1.3.0) ───────────────
    // When true, TypoResilientDetect assigns a virtual "he" confidence to a
    // layout-converted variant that is >= HebrewScriptCoverageThreshold Hebrew
    // Unicode chars AND was not strongly claimed as another language by ONNX.
    // Catches clearly-Hebrew text the model scores ambiguously. Default: true.
    extern bool  EnableHebrewScriptGate;

    // ── Hebrew weak-signal gates (FN guard) ───────────────────────────
    // Persistent Moderate Confidence Gate and Cumulative Weak Score Gate.
    // Both apply only to "he" targets and recover flat-signal Hebrew phrases
    // the adaptive threshold would otherwise miss.  Default: true.
    extern bool  EnablePersistentConfGate;
    extern bool  EnableWeakScoreGate;

    // ================================================================
    // Case-signal Hebrew exclusion (Iteration 5 — A)
    // ================================================================
    // Hebrew has no upper/lowercase. When the cached text contains
    // enough Shift/CapsLock intent (ALL-CAPS abbreviations like "FPS",
    // or internal capitals like "iPhone") it is almost certain the text
    // is NOT Hebrew. These flags enable a lightweight pre-filter that
    // adds "he" to the exclusion set before running inference.
    //
    //  EnableCaseBasedHeExclusion — master toggle (default: true)
    //  CaseExclusionMinCaps       — exclude Hebrew when UpperCount reaches
    //                               this many alpha chars with shift intent
    //                               (default: 2; "OK" → excluded, "Hi" → ok)
    //
    // The exclusion only removes "he" as a CANDIDATE; it does NOT skip
    // inference altogether, and it does NOT affect the history gate.
    // A sentence-initial capital ("Hello") never triggers exclusion because
    // HasInternalCapital() returns false for single-leading-cap words.
    extern bool EnableCaseBasedHeExclusion;
    extern int  CaseExclusionMinCaps;

    // ================================================================
    // New parameters (Iteration 1)
    // ================================================================

    // Minimum number of dictionary-known characters required before
    // running ONNX inference.  Text with fewer known chars is skipped
    // with reason skip_low_known_chars.  Default: 2.
    extern int MinKnownCharsForInference;

    // When true, auto-switching from English to Hebrew/Russian is blocked
    // inside file-open/save dialogs (#32770 + DirectUIHWND/ComboBoxEx32).
    // Manual layout switches by the user are never blocked.  Default: true.
    extern bool DisableAutoSwitchFromEnglishInFileDialogs;

    // ================================================================
    // Skip-reason counters — diagnostic / telemetry
    // ================================================================
    // Each guard branch increments the corresponding counter so that
    // the debug log can report aggregate statistics.
    struct SkipCounters {
        std::atomic<uint32_t> skipEmptyAfterTokenize{0};
        std::atomic<uint32_t> skipLowKnownChars{0};
        std::atomic<uint32_t> skipLowAlpha{0};
        std::atomic<uint32_t> skipUrlOrPath{0};
        std::atomic<uint32_t> skipFileDialogEnProtection{0};
        std::atomic<uint32_t> correctionsApplied{0};

        void Reset() noexcept;
        std::string Summary() const;
    };
    extern SkipCounters Guards;

    // Language codes for ActivateKeyboardLayout / PostMessage
    extern const std::unordered_map<std::string, HKL> LANGUAGE_CODES;

    // Language ID mapping: LANGID -> language code string
    extern const std::unordered_map<LANGID, std::string> LANGUAGE_ID;

    // Get language string from LANGID
    std::string GetLanguageFromId(LANGID langId);

    // Get HKL from language string
    HKL GetHKLFromLanguage(const std::string& lang);

    // ================================================================
    // Adaptive calibration sink
    // ================================================================
    // Writes calibration-derived effective values for a language pair
    // directly into PairOverrides (inserting a full-param entry from the
    // applicable base if the pair is not yet present).  Called only from
    // FeedbackLogger::RecordOutcome and ::ResetCalibration.
    void ApplyAdaptedParams(const std::string& fromLang, const std::string& toLang,
                            float confAtMax, float margin);

} // namespace Config
