#pragma once

// ============================================================
// Lexical triage — fast dictionary stage of the hybrid pipeline
// ============================================================
// Loads per-language minimal-DAWG lexicons (triage_lexicons.bin, built by
// model/build_cpp_triage.py) and scores detection-text variants against a
// claimed language with exact + prefix dictionary membership.
//
// The triage is ADVISORY ONLY: the LSTM always runs; the evidence merely
// raises the required confidence when the typed text is fully attested in
// the current language (suppress) or lowers it when the switch candidate
// is fully attested in the target language (boost).  Offline tuning
// (model/sweep_triage_params.py, model/compare_pipelines.py) showed that
// exact+prefix matching alone gives the entire accuracy win; fuzzy /
// trigram / LSH components were evaluated and rejected (see model/triage.py).

#include <cstdint>
#include <string>
#include <vector>

class LexicalTriage {
public:
    // Load the DAWG binary and install the process-wide instance.
    // Returns false (and leaves Instance() == nullptr) on any error;
    // the app then runs the pure-LSTM pipeline.
    static bool Initialize(const std::wstring& binPath);

    // Process-wide instance, or nullptr when not loaded.
    static const LexicalTriage* Instance();

    // Combined lexical evidence in [0,1] that *text* is language *lang*
    // ("en" / "ru" / "he").  Evidence is the fraction of space-separated
    // words attested in the lexicon; the last (still being typed) word
    // gets prefix credit from MIN_PREFIX_LEN chars.  Script mismatch
    // (text not written purely in lang's script) yields 0 by construction.
    float Evidence(const std::wstring& text, const std::string& lang) const;

    // Language whose script the text is written in ("en"/"ru"/"he"),
    // or "" when mixed/none. Mirrors model/triage.py script_lang().
    static std::string ScriptLang(const std::wstring& text);

    // The last partial word needs at least this many chars for prefix credit
    // (mirrors TriageParams.min_prefix_len in model/triage.py).
    static constexpr size_t MIN_PREFIX_LEN = 3;

private:
    struct Dawg {
        std::vector<uint32_t> first;      // stateCount+1 (CSR slice bounds)
        std::vector<uint8_t>  finalBits;  // 1 bit per state
        std::vector<uint16_t> labels;     // transition chars (BMP)
        std::vector<uint32_t> targets;    // transition destinations
        uint32_t wordCount = 0;

        bool Loaded() const { return !first.empty(); }
        // Walk the word; returns the reached state or -1 if it fell off.
        int64_t Walk(const std::wstring& w) const;
        bool Accepts(const std::wstring& w) const;        // exact word
        bool AcceptsPrefix(const std::wstring& w) const;  // prefix of any word
    };

    const Dawg* ForLang(const std::string& lang) const;

    Dawg en_, ru_, he_;
};
