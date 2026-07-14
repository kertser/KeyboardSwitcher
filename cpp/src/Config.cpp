#include "Config.h"
#include <cstdio>
#include <string>

namespace Config {

    std::atomic<bool> EnableSwitcher{true};
    std::atomic<bool> SEARCH{false};
    std::atomic<bool> SaveWindowState{true};
    std::atomic<bool> alt_pressed{false};
    std::string LastSetting = "en";

    // ================================================================
    // SwitchingParams – per-pair adaptive confidence curve
    // ================================================================
    float SwitchingParams::GetRequiredConfidence(size_t numChars, bool isPhrase) const {
        int n = static_cast<int>(numChars);
        if (n < EarlyDetectionMinChars)
            return 1.1f;                       // impossible → no detection

        float required;
        if (n >= FullConfidenceChars) {
            required = ConfidenceAtMaxChars;    // floor
        } else {
            // Linear interpolation
            float t = static_cast<float>(n - EarlyDetectionMinChars)
                    / static_cast<float>(FullConfidenceChars - EarlyDetectionMinChars);
            required = ConfidenceAtMinChars + t * (ConfidenceAtMaxChars - ConfidenceAtMinChars);
        }

        // Phrase mode: multi-word input is far less ambiguous than a single
        // word, so the threshold is relaxed by PhraseConfScale (≤ 1.0).
        if (isPhrase && PhraseConfScale > 0.0f && PhraseConfScale < 1.0f)
            required *= PhraseConfScale;

        return required;
    }

    // ── Global default parameters ──────────────────────────────────
    // EarlyDetectionMinChars = 4: wait for at least 4 alpha chars before
    // the detection engine fires.  Raising this from 3 eliminates most
    // false positives on very short input (≤3 chars) while still allowing
    // detection on 4-char words (the threshold tightens to ConfAtMinChars=0.99).
    //
    // Field order (11): EarlyMin, FullConf, ConfMin, ConfMax, AgreeCount,
    //   BorderlineFactor, MinTop1Top2Margin, ShortInputExtraConf,
    //   PhraseConfScale, HebrewScriptVirtualConf, HebrewScriptCoverageThreshold,
    //   SwitchBiasMargin, PersistentMinAvgConf, PersistentMinSteps,
    //   WeakScoreClassIdx, WeakScoreMinAvg, WeakScoreWindow.
    SwitchingParams DefaultParams = {
        /* EarlyDetectionMinChars  */ 4,
        /* FullConfidenceChars     */ 15,
        /* ConfidenceAtMinChars    */ 0.99f,
        /* ConfidenceAtMaxChars    */ 0.70f,
        /* ConsecutiveAgreementCount */ 2,
        /* BorderlineZoneFactor    */ 0.85f,
        /* MinTop1Top2Margin       */ 0.05f,
        /* ShortInputExtraConf     */ 0.02f,
        /* PhraseConfScale         */ 1.00f,
        /* HebrewScriptVirtualConf */ 0.00f,   // disabled for non-→he pairs
        /* HebrewScriptCovThreshold*/ 0.90f,
        /* SwitchBiasMargin        */ 0.00f,   // incumbent guard: only →he pairs
        /* PersistentMinAvgConf    */ 0.55f,
        /* PersistentMinSteps      */ 6,       // ≥6 frames → no single-word FP
        /* WeakScoreClassIdx       */ 2,
        /* WeakScoreMinAvg         */ 0.40f,   // conservative: no FP on harness
        /* WeakScoreWindow         */ 7,
    };

    // ── Per-pair overrides ─────────────────────────────────────────
    // Field order matches SwitchingParams (see DefaultParams above):
    //   EMin FConf CAt0  CAt1  Agr BLF    Mrg    SXC    PCS    HeVC   HeCT
    //   SBM    PMAC   PMS WSCI  WSMA   WSW
    //
    // Margin gate (Mrg): 0.05 on robust en↔ru / he→en / he→ru pairs (cheap
    //   FP insurance — measurably free on the single-word harness), and a
    //   stricter 0.10 on →he pairs where EN/HE softmax frequently compete.
    // SwitchBiasMargin (SBM): incumbent-advantage FP guard.  Tuned to 0.02 on
    //   →he only (after the v2.x model retrain the incumbent EN signal on
    //   Hebrew-typed-on-English phrases rose to ~0.93–0.95, so the previous
    //   0.04 blocked ~4 genuine Hebrew phrases — "בוקר טוב", "ברוך הבא",
    //   "הכל בסדר", "בוא נדבר", "מה קורה" — at zero single-word FP benefit;
    //   0.02 restores full phrase recall while keeping an incumbent guard);
    //   0.0 on the robust pairs where it only cost true positives.
    // Persistent / weak-score gates (PMS=6, WSMA=0.40): restored Hebrew
    //   flat-signal recovery, tuned so they add NO single-word false positive
    //   on the offline harness while remaining available for long Hebrew
    //   phrases.  Only ever fire for the "he" target.
    std::map<LangPair, SwitchingParams> PairOverrides = {
        // ── English ↔ Russian ───
        { {"en", "ru"}, { 4, 15, 0.99f, 0.70f, 2, 0.85f, 0.05f, 0.02f, 1.00f, 0.00f, 0.90f,  0.00f, 0.55f, 6, 2, 0.40f, 7 } },
        { {"ru", "en"}, { 4, 15, 0.99f, 0.70f, 2, 0.85f, 0.05f, 0.02f, 1.00f, 0.00f, 0.90f,  0.00f, 0.55f, 6, 2, 0.40f, 7 } },

        // ── English ↔ Hebrew ───
        // en→he: EarlyMin=3 (vs 4) adds ~4 pp TP at zero FP cost (sweep-validated).
        //        ConfAtMax=0.60 floor; narrower borderline zone (0.88).
        //   Signal-quality gates (ported from 1.3.0) — validated +30 pp recall
        //   on multi-word phrases (test_phrases_lite.py), neutral on single words:
        //     Margin=0.10  — stricter top1/top2 gap (Hebrew/EN often compete)
        //     PhraseScale=0.80 — 20 % lower threshold for multi-word phrases
        //     HeScriptVC=0.78  — a 100 %-Hebrew variant gets virtual conf 0.78
        //     SBM=0.02 — incumbent advantage; persistent/weak gates active for →he
        { {"en", "he"}, { 3, 15, 0.99f, 0.60f, 2, 0.88f, 0.10f, 0.00f, 0.80f, 0.78f, 0.90f,  0.02f, 0.55f, 6, 2, 0.40f, 7 } },
        { {"he", "en"}, { 4, 15, 0.99f, 0.70f, 2, 0.85f, 0.05f, 0.02f, 1.00f, 0.00f, 0.90f,  0.00f, 0.55f, 6, 2, 0.40f, 7 } },

        // ── Russian ↔ Hebrew ───
        // ru→he: same tuning + gates as en→he.
        { {"ru", "he"}, { 3, 15, 0.99f, 0.60f, 2, 0.88f, 0.10f, 0.00f, 0.80f, 0.78f, 0.90f,  0.02f, 0.55f, 6, 2, 0.40f, 7 } },
        { {"he", "ru"}, { 4, 15, 0.99f, 0.70f, 2, 0.80f, 0.05f, 0.02f, 1.00f, 0.00f, 0.90f,  0.00f, 0.55f, 6, 2, 0.40f, 7 } },
    };

    const SwitchingParams& GetParamsForPair(const std::string& fromLang,
                                            const std::string& toLang) {
        auto it = PairOverrides.find({fromLang, toLang});
        if (it != PairOverrides.end())
            return it->second;
        return DefaultParams;
    }

    // ── Legacy global aliases (point into DefaultParams) ───────────
    int&   EarlyDetectionMinChars   = DefaultParams.EarlyDetectionMinChars;
    int&   FullConfidenceChars      = DefaultParams.FullConfidenceChars;
    float& ConfidenceAtMinChars     = DefaultParams.ConfidenceAtMinChars;
    float& ConfidenceAtMaxChars     = DefaultParams.ConfidenceAtMaxChars;
    int&   ConsecutiveAgreementCount = DefaultParams.ConsecutiveAgreementCount;
    float& BorderlineZoneFactor     = DefaultParams.BorderlineZoneFactor;

    float GetRequiredConfidence(size_t numChars) {
        return DefaultParams.GetRequiredConfidence(numChars);
    }

    void ApplyAdaptedParams(const std::string& fromLang, const std::string& toLang,
                            float confAtMax, float margin) {
        auto it = PairOverrides.find({fromLang, toLang});
        if (it != PairOverrides.end()) {
            it->second.ConfidenceAtMaxChars = confAtMax;
            it->second.MinTop1Top2Margin    = margin;
        } else {
            // Insert a full-param copy of the applicable base, then override
            // just the two calibrated fields.  This ensures the pair uses the
            // correct per-pair values for all other parameters.
            SwitchingParams p = DefaultParams;
            p.ConfidenceAtMaxChars = confAtMax;
            p.MinTop1Top2Margin    = margin;
            PairOverrides[{fromLang, toLang}] = p;
        }
    }

    // ── Typo resilience master toggle ──
    bool  EnableTypoResilience = true;

    // ── Hebrew Script Coverage Gate (ported from 1.3.0) ──
    bool  EnableHebrewScriptGate = true;

    // ── Hebrew weak-signal gates (FN guard) ──
    bool  EnablePersistentConfGate = true;
    bool  EnableWeakScoreGate      = true;

    // ── Word-aware consensus detection (v1.5.x) ──
    // Per-word scoring fused with the whole-string softmax via geometric-mean
    // consensus.  Only affects multi-word buffers; single words are unchanged.
    // Weighting: base weight = min(wordLen, LenCap); words <= ShortMaxLen chars
    // are multiplied by ShortWeight (they carry little discriminative signal).
    bool  EnableWordAwareDetection = true;
    int   WordAwareMinWords        = 2;
    int   WordAwareShortMaxLen     = 2;
    float WordAwareShortWeight     = 0.35f;
    int   WordAwareLenCap          = 8;

    // ── Case-signal Hebrew exclusion (Iteration 5 — A) ────────────
    // Exclude "he" when ≥ CaseExclusionMinCaps alpha chars were typed
    // with Shift intent, OR any internal (non-leading) alpha char was
    // capitalised.  Sentence-initial caps ("Hello") are not counted.
    bool EnableCaseBasedHeExclusion = true;
    int  CaseExclusionMinCaps       = 2;

    // ── New parameters (Iteration 1) ───────────────────────────────
    // Minimum known-char count for ONNX inference.  2 keeps very short
    // but legitimate inputs detectable while filtering pure-symbol noise.
    int  MinKnownCharsForInference = 2;

    // Protect English layout in file-open/save dialogs against False
    // auto-switches to Hebrew/Russian.  Default: enabled.
    bool DisableAutoSwitchFromEnglishInFileDialogs = true;

    // ── Skip-reason counters ────────────────────────────────────────
    SkipCounters Guards;

    void SkipCounters::Reset() noexcept {
        skipEmptyAfterTokenize  = 0;
        skipLowKnownChars       = 0;
        skipLowAlpha            = 0;
        skipUrlOrPath           = 0;
        skipFileDialogEnProtection = 0;
        correctionsApplied      = 0;
    }

    std::string SkipCounters::Summary() const {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "guards: emptyTok=%u lowKnown=%u lowAlpha=%u urlPath=%u fileDialog=%u corrections=%u",
            skipEmptyAfterTokenize.load(),
            skipLowKnownChars.load(),
            skipLowAlpha.load(),
            skipUrlOrPath.load(),
            skipFileDialogEnProtection.load(),
            correctionsApplied.load());
        return buf;
    }

    // Language codes: HKL values matching the Python project
    const std::unordered_map<std::string, HKL> LANGUAGE_CODES = {
        {"en", reinterpret_cast<HKL>(static_cast<uintptr_t>(0x04090409))},
        {"ru", reinterpret_cast<HKL>(static_cast<uintptr_t>(0x04190419))},
        {"he", reinterpret_cast<HKL>(static_cast<uintptr_t>(0xF03D040D))}
    };

    // LANGID -> language string
    const std::unordered_map<LANGID, std::string> LANGUAGE_ID = {
        {1033, "en"},  // English (United States)
        {1049, "ru"},  // Russian
        {1037, "he"}   // Hebrew (Israel)
    };

    std::string GetLanguageFromId(LANGID langId) {
        auto it = LANGUAGE_ID.find(langId);
        if (it != LANGUAGE_ID.end()) {
            return it->second;
        }
        return "en"; // default
    }

    HKL GetHKLFromLanguage(const std::string& lang) {
        auto it = LANGUAGE_CODES.find(lang);
        if (it != LANGUAGE_CODES.end()) {
            return it->second;
        }
        return LANGUAGE_CODES.at("en"); // default
    }

} // namespace Config
