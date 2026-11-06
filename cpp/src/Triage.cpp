#include "Triage.h"
#include "Languages.h"   // NormalizeHebrewFinals

#include <windows.h>     // IsCharAlphaW

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>

// ============================================================
// Singleton
// ============================================================
static std::unique_ptr<LexicalTriage> g_triage;

const LexicalTriage* LexicalTriage::Instance() {
    return g_triage.get();
}

// ============================================================
// DAWG lookups
// ============================================================
int64_t LexicalTriage::Dawg::Walk(const std::wstring& w) const {
    uint32_t s = 0;
    for (wchar_t ch : w) {
        const uint16_t cp = static_cast<uint16_t>(ch);
        const uint32_t lo = first[s], hi = first[s + 1];
        const uint16_t* begin = labels.data() + lo;
        const uint16_t* end   = labels.data() + hi;
        const uint16_t* it = std::lower_bound(begin, end, cp);
        if (it == end || *it != cp) return -1;
        s = targets[lo + static_cast<uint32_t>(it - begin)];
    }
    return static_cast<int64_t>(s);
}

bool LexicalTriage::Dawg::Accepts(const std::wstring& w) const {
    const int64_t s = Walk(w);
    if (s < 0) return false;
    return (finalBits[s >> 3] >> (s & 7)) & 1;
}

bool LexicalTriage::Dawg::AcceptsPrefix(const std::wstring& w) const {
    return Walk(w) >= 0;
}

// ============================================================
// Script detection (mirrors model/triage.py script_lang)
// ============================================================
std::string LexicalTriage::ScriptLang(const std::wstring& text) {
    std::string lang;
    for (wchar_t c : text) {
        const char* cur = nullptr;
        if ((c >= L'a' && c <= L'z') || (c >= L'A' && c <= L'Z')) {
            cur = "en";
        } else if (c >= 0x0400 && c <= 0x04FF) {
            cur = "ru";
        } else if (c >= 0x05D0 && c <= 0x05EA) {
            cur = "he";
        } else if (IsCharAlphaW(c)) {
            return "";   // a letter outside all three scripts -> mixed/none
        } else {
            continue;    // digits / punctuation / spaces are neutral
        }
        if (lang.empty()) lang = cur;
        else if (lang != cur) return "";
    }
    return lang;
}

// ============================================================
// Normalization (mirrors model/triage.py _normalize)
// ============================================================
// Explicit ASCII + Cyrillic lowering: towlower() in the default "C"
// locale does not fold Cyrillic, and the Python side uses str.lower().
static wchar_t TriageLowerChar(wchar_t c) {
    if (c >= L'A' && c <= L'Z') return c + 32;
    if (c >= 0x0410 && c <= 0x042F) return c + 0x20;  // А-Я -> а-я
    if (c == 0x0401) return 0x0451;                   // Ё -> ё
    return c;
}

// ============================================================
// Evidence
// ============================================================
const LexicalTriage::Dawg* LexicalTriage::ForLang(const std::string& lang) const {
    if (lang == "en") return &en_;
    if (lang == "ru") return &ru_;
    if (lang == "he") return &he_;
    return nullptr;
}

float LexicalTriage::Evidence(const std::wstring& text,
                              const std::string& lang) const {
    const Dawg* dawg = ForLang(lang);
    if (!dawg || !dawg->Loaded()) return 0.0f;
    if (ScriptLang(text) != lang) return 0.0f;   // script mismatch: zero

    // Normalize: lowercase (+ Hebrew final forms), then split on spaces.
    std::wstring norm;
    norm.reserve(text.size());
    for (wchar_t c : text) norm.push_back(TriageLowerChar(c));
    if (lang == "he") norm = NormalizeHebrewFinals(norm);

    int total = 0, attested = 0;
    size_t pos = 0;
    const size_t n = norm.size();
    while (pos < n) {
        while (pos < n && norm[pos] == L' ') ++pos;
        if (pos >= n) break;
        size_t end = norm.find(L' ', pos);
        if (end == std::wstring::npos) end = n;
        const std::wstring word = norm.substr(pos, end - pos);
        const bool isLast = (end >= n);
        ++total;
        if (dawg->Accepts(word)) {
            ++attested;
        } else if (isLast && word.size() >= MIN_PREFIX_LEN &&
                   dawg->AcceptsPrefix(word)) {
            // The last word is still being typed: a prefix of a lexicon
            // word counts as exact evidence (mirrors model/triage.py).
            ++attested;
        }
        pos = end;
    }
    if (total == 0) return 0.0f;
    return static_cast<float>(attested) / static_cast<float>(total);
}

// ============================================================
// Binary loading (format: model/build_cpp_triage.py)
// ============================================================
namespace {

struct Reader {
    FILE* f;
    bool ok = true;

    bool Bytes(void* dst, size_t n) {
        if (!ok || fread(dst, 1, n, f) != n) { ok = false; }
        return ok;
    }
    uint32_t U32() {
        uint32_t v = 0;
        Bytes(&v, sizeof(v));
        return v;   // little-endian host (Windows x86/x64/ARM64)
    }
};

} // namespace

bool LexicalTriage::Initialize(const std::wstring& binPath) {
    g_triage.reset();

    FILE* f = _wfopen(binPath.c_str(), L"rb");
    if (!f) return false;
    Reader r{ f };

    char magic[8] = {};
    r.Bytes(magic, 8);
    if (!r.ok || std::memcmp(magic, "KSDAWG01", 8) != 0) {
        fclose(f);
        return false;
    }

    auto triage = std::make_unique<LexicalTriage>();
    const uint32_t langCount = r.U32();
    if (!r.ok || langCount == 0 || langCount > 8) { fclose(f); return false; }

    for (uint32_t i = 0; i < langCount && r.ok; ++i) {
        char code[5] = {};
        r.Bytes(code, 4);
        const uint32_t stateCount = r.U32();
        const uint32_t transCount = r.U32();
        const uint32_t wordCount  = r.U32();
        if (!r.ok || stateCount == 0 || transCount == 0 ||
            stateCount > 50'000'000 || transCount > 100'000'000) {
            r.ok = false;
            break;
        }

        Dawg d;
        d.wordCount = wordCount;
        d.first.resize(stateCount + 1);
        r.Bytes(d.first.data(), sizeof(uint32_t) * (stateCount + 1));
        d.finalBits.resize((stateCount + 7) / 8);
        r.Bytes(d.finalBits.data(), d.finalBits.size());
        d.labels.resize(transCount);
        r.Bytes(d.labels.data(), sizeof(uint16_t) * transCount);
        d.targets.resize(transCount);
        r.Bytes(d.targets.data(), sizeof(uint32_t) * transCount);

        // Structural sanity: CSR bounds and targets must be in range.
        if (r.ok) {
            if (d.first[0] != 0 || d.first[stateCount] != transCount) r.ok = false;
            for (uint32_t s = 0; r.ok && s < stateCount; ++s)
                if (d.first[s] > d.first[s + 1]) r.ok = false;
            for (uint32_t t = 0; r.ok && t < transCount; ++t)
                if (d.targets[t] >= stateCount) r.ok = false;
        }
        if (!r.ok) break;

        if (std::strcmp(code, "en") == 0)      triage->en_ = std::move(d);
        else if (std::strcmp(code, "ru") == 0) triage->ru_ = std::move(d);
        else if (std::strcmp(code, "he") == 0) triage->he_ = std::move(d);
        // unknown language blocks are skipped silently (forward compat)
    }
    fclose(f);

    if (!r.ok || !triage->en_.Loaded() || !triage->ru_.Loaded() ||
        !triage->he_.Loaded()) {
        return false;
    }

    g_triage = std::move(triage);
    return true;
}
