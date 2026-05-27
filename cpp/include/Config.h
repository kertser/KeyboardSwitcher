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

        // ── Trend Gate & signal-quality gates (Iteration 2+) ──────────
        // MinTop1Top2Margin: required gap between top-1 and top-2 softmax
        //   probabilities. 0.0 = disabled.  Prevents corrections when the
        //   model is split between two languages.
        float MinTop1Top2Margin;

        // VariantAgreementCount: minimum number of layout-conversion variants
        //   that must agree on the same target language.  0 = disabled.
        //   Applied only when confidence is below ConfidenceAtMinChars to
        //   avoid slowing down high-confidence detections.
        int   VariantAgreementCount;

        // TrendWindowSize / MinStableSteps / MinTrendSlope:
        //   Sliding-window trend gate used as an OR-alternative to the
        //   ConsecutiveAgreementCount gate.  The window stores the last N
        //   detection observations; IsTrendStable() returns true when at
        //   least MinStableSteps of them confirm the target language AND
        //   the confidence slope is >= MinTrendSlope.  0 disables each.
        int   TrendWindowSize;
        int   MinStableSteps;
        float MinTrendSlope;         // per-step minimum slope (0 = ignore slope)

        // ShortInputExtraConf: extra confidence added on top of the adaptive
        //   threshold when alphaCount is in [EarlyDetectionMinChars, +2].
        //   Aggressively guards against FP on very short input.
        float ShortInputExtraConf;

        // PhraseConfScale: when the detection text contains a space (phrase
        //   context, not a single word), multiply the required confidence by
        //   this factor.  1.0 = no change.  0.85 means the threshold is 15%
        //   lower for multi-word input.  This improves recall on short Hebrew
        //   phrases like "תודה רבה" where individual words have borderline
        //   confidence but the combination is unambiguous.
        float PhraseConfScale;

        // ── Hebrew Script Coverage Gate (Iteration 3) ─────────────────
        // HebrewScriptVirtualConf: when a layout-conversion variant contains
        //   >= HebrewScriptCoverageThreshold fraction of Hebrew Unicode chars
        //   (U+05D0–U+05EA) AND the ONNX model did NOT strongly claim a
        //   non-Hebrew language for that same variant, assign this virtual
        //   confidence to "he".  If it beats the current bestConf the gate
        //   overrides the ONNX winner.
        //   0.0 = disabled.  Typical: 0.78 for →he pairs.
        float HebrewScriptVirtualConf;

        // HebrewScriptCoverageThreshold: minimum Hebrew character fraction
        //   required to trigger the script gate.  0.90 = 90 % of alpha chars
        //   must be Hebrew Unicode.  Higher = fewer false positives.
        float HebrewScriptCoverageThreshold;

        // ── Persistent Moderate Confidence Gate (Iteration 3) ──────────
        // PersistentMinAvgConf: the gate fires when ALL of the last
        //   PersistentMinSteps frames had top1Lang == bestLang AND the
        //   average top1Conf was >= this value.  Catches the case where
        //   the model gives a flat ~58 % confidence across many keystrokes
        //   — below the trend gate's growth requirement but clearly not noise.
        //   0.0 = disabled.  Typical: 0.52 for →he pairs.
        float PersistentMinAvgConf;

        // PersistentMinSteps: minimum consecutive frames required for the
        //   persistent gate.  Typical: 5 for →he pairs.
        int   PersistentMinSteps;

        // ── Cumulative Weak Score Gate (Iteration 3) ───────────────────
        // WeakScoreClassIdx: softmax class index to track (2 = Hebrew).
        //   Set to -1 to disable.
        int   WeakScoreClassIdx;

        // WeakScoreMinAvg: minimum average softmax score for the tracked
        //   class over WeakScoreWindow frames (even when that class is NOT
        //   top-1).  Catches persistent weak "he" signal below the
        //   top-1 threshold.  Must be well above random (>= 0.20 suggested).
        float WeakScoreMinAvg;

        // WeakScoreWindow: sliding-window size for the weak-score gate.
        int   WeakScoreWindow;

        // Compute the required softmax confidence for a given text length.
        // isPhrase: true when the detection text contains a space — in that
        //   case PhraseConfScale is applied to lower the threshold.
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

    // ── Trend Gate feature flags ───────────────────────────────────────
    // EnableTrendGate: when true, IsTrendStable() is evaluated and used as
    //   an OR-alternative to the consecutive-agreement gate.  Default: true.
    // EnableTrendGateBlock: when false (log-only mode), the trend gate is
    //   computed and logged but never blocks a correction.  Default: true.
    // EnableVariantConsensus: require VariantAgreementCount variants to
    //   agree before firing on borderline-confidence input.  Default: true.
    extern bool  EnableTrendGate;
    extern bool  EnableTrendGateBlock;
    extern bool  EnableVariantConsensus;

    // ── Iteration 3 feature flags ──────────────────────────────────────
    // EnableHebrewScriptGate: activates the Hebrew Unicode coverage gate
    //   inside TypoResilientDetect.  When a layout-converted variant is
    //   >= HebrewScriptCoverageThreshold Hebrew chars AND the ONNX model
    //   did not strongly claim another language for that variant, a virtual
    //   "he" confidence (HebrewScriptVirtualConf) is assigned.  Default: true.
    extern bool  EnableHebrewScriptGate;

    // EnablePersistentConfGate: activates the "N consecutive frames of the
    //   same language at moderate confidence" gate as a third OR-alternative
    //   to agreement/trend.  Default: true.
    extern bool  EnablePersistentConfGate;

    // EnableWeakScoreGate: activates the cumulative weak softmax-score gate.
    //   Tracks the avg softmax for a specific class (e.g. Hebrew = 2) even
    //   when it is not top-1; fires when the average exceeds WeakScoreMinAvg
    //   over WeakScoreWindow frames.  Default: true.
    extern bool  EnableWeakScoreGate;

    // AddOriginalTextAsVariant: when true, the unmodified detectionText is
    //   prepended to textVariants before ONNX inference.  This lets the model
    //   confirm "this text is already correct English/Russian" and naturally
    //   suppresses false-positive Hebrew corrections on real en/ru words.
    //   Default: true.
    extern bool  AddOriginalTextAsVariant;

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

} // namespace Config
