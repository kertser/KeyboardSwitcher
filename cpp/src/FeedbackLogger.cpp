#include "FeedbackLogger.h"
#include "Config.h"

#include <windows.h>
#include <shlobj.h>          // SHGetFolderPathW
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <set>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace Feedback {

// ── State ───────────────────────────────────────────────────
std::atomic<bool> LoggingEnabled{false};

static std::wstring g_dataDir;            // %APPDATA%/KeyboardSwitcher
static std::wstring g_feedbackPath;       // .../feedback.jsonl
static std::wstring g_prefsPath;          // .../user_prefs.json

// Exception list: corrected texts that the user rejected, keyed by
// target language.  E.g. exceptions["he"] = {"סעחר", "ררנוי"} means
// any correction that would produce those Hebrew texts is blocked.
static std::map<std::string, std::set<std::wstring>> g_exceptions;

// Correction overrides: learned reverse corrections from rejections.
// E.g. overrides["he"]["סעחר"] = "ru" means: when the user is on
// Hebrew keyboard and types "סעחר", they actually want Russian.
// Derived from: model wrongly corrected ru→he producing "סעחר",
// user rejected → we learn the reverse (he→ru).
static std::map<std::string, std::map<std::wstring, std::string>> g_overrides;

static constexpr size_t MAX_FEEDBACK_BYTES = 2 * 1024 * 1024;  // 2 MB
static constexpr size_t MAX_EXCEPTIONS_PER_LANG = 500;

// ── Adaptive calibration ─────────────────────────────────────────────
// Per-(from→to) state machine.  All values live on the main hook thread.
struct PairCalibration {
    // EWMA error rates (event-driven, updated on every RecordOutcome call)
    float ewmaFpRate  = 0.0f;   // exponential MA of false-positive rate
    float ewmaFnRate  = 0.0f;   // exponential MA of false-negative rate

    // Events accumulated in the current batch (reset after each adaptation)
    int   batchEvents = 0;

    // Cumulative applied deltas relative to the factory baseline
    float deltaConfAtMax = 0.0f;
    float deltaMargin    = 0.0f;

    // Factory baseline snapshotted once (from Config at first RecordOutcome
    // call, or restored from user_prefs.json on subsequent sessions).
    float baseConfAtMax = 0.0f;
    float baseMargin    = 0.0f;
    bool  baseLoaded    = false;
};

using PairKey = std::pair<std::string, std::string>;
static std::map<PairKey, PairCalibration> g_calibration;

// ── Controller constants ──────────────────────────────────────────────
// EWMA decay factor per event (α=0.2 → ~5 events half-life).
static constexpr float CALIB_EWMA_ALPHA        = 0.20f;
// Minimum events per batch before any adaptation is attempted.
static constexpr int   CALIB_MIN_EVENTS        = 5;
// |pressure| must exceed this before a step is taken (dead zone).
static constexpr float CALIB_HYSTERESIS_BAND   = 0.15f;
// Step size per adaptation for ConfidenceAtMaxChars.
static constexpr float CALIB_STEP_CONF         = 0.01f;
// Step size per adaptation for MinTop1Top2Margin (half the conf step).
static constexpr float CALIB_STEP_MARGIN       = 0.005f;
// Maximum cumulative delta for tightening (raising thresholds).
static constexpr float CALIB_MAX_TIGHTEN_CONF  = 0.10f;
static constexpr float CALIB_MAX_TIGHTEN_MARG  = 0.08f;
// Maximum cumulative delta for loosening (lowering thresholds).
// Deliberately asymmetric: loosening is more conservative than tightening
// because a too-permissive setting produces more visible false positives.
static constexpr float CALIB_MAX_LOOSEN_CONF   = 0.05f;
static constexpr float CALIB_MAX_LOOSEN_MARG   = 0.01f;
// Absolute clamps applied after base+delta to prevent extreme values
// regardless of how far the delta drifts.
static constexpr float CALIB_ABS_MIN_CONF      = 0.50f;
static constexpr float CALIB_ABS_MAX_CONF      = 0.995f;
static constexpr float CALIB_ABS_MIN_MARGIN    = 0.005f;
static constexpr float CALIB_ABS_MAX_MARGIN    = 0.25f;

// ── Helpers ─────────────────────────────────────────────────

// Narrow → wide
static std::wstring s2ws(const std::string& s) {
    if (s.empty()) return {};
    int sz = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    std::wstring out(sz, 0);
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), out.data(), sz);
    return out;
}
// Wide → narrow (UTF-8)
static std::string ws2s(const std::wstring& ws) {
    if (ws.empty()) return {};
    int sz = WideCharToMultiByte(CP_UTF8, 0, ws.data(), (int)ws.size(),
                                 nullptr, 0, nullptr, nullptr);
    std::string out(sz, 0);
    WideCharToMultiByte(CP_UTF8, 0, ws.data(), (int)ws.size(),
                        out.data(), sz, nullptr, nullptr);
    return out;
}

static std::string NowISO8601() {
    auto now = std::chrono::system_clock::now();
    std::time_t tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_s(&tm, &tt);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm);
    return buf;
}

// Rotate feedback.jsonl: keep newest half when file exceeds limit.
static void RotateIfNeeded() {
    WIN32_FILE_ATTRIBUTE_DATA attr{};
    if (!GetFileAttributesExW(g_feedbackPath.c_str(), GetFileExInfoStandard, &attr))
        return;
    ULARGE_INTEGER sz;
    sz.LowPart  = attr.nFileSizeLow;
    sz.HighPart = attr.nFileSizeHigh;
    if (sz.QuadPart < MAX_FEEDBACK_BYTES) return;

    // Read entire file
    std::ifstream in(g_feedbackPath.c_str(), std::ios::binary);
    if (!in) return;
    std::string data((std::istreambuf_iterator<char>(in)),
                      std::istreambuf_iterator<char>());
    in.close();

    // Find the newline closest to the midpoint
    size_t mid = data.size() / 2;
    size_t pos = data.find('\n', mid);
    if (pos == std::string::npos || pos + 1 >= data.size()) return;

    std::string newer = data.substr(pos + 1);
    std::ofstream out(g_feedbackPath.c_str(), std::ios::binary | std::ios::trunc);
    if (out) out << newer;
}

// ── Persistence ─────────────────────────────────────────────

static void LoadPrefs() {
    std::ifstream in(g_prefsPath.c_str());
    if (!in) return;
    try {
        json j = json::parse(in);

        // Version guard — discard exceptions on app update
        // (new model may handle them correctly)
        std::string savedVer = j.value("version", "");
        std::string currentVer = ws2s(Config::VERSION);
        if (savedVer != currentVer) {
            return;  // start fresh
        }

        LoggingEnabled.store(j.value("logging_enabled", false));

        if (j.contains("exceptions") && j["exceptions"].is_object()) {
            for (auto& [lang, arr] : j["exceptions"].items()) {
                if (!arr.is_array()) continue;
                auto& langSet = g_exceptions[lang];
                for (auto& val : arr) {
                    if (val.is_string()) {
                        langSet.insert(s2ws(val.get<std::string>()));
                    }
                }
            }
        }

        if (j.contains("overrides") && j["overrides"].is_object()) {
            for (auto& [lang, obj] : j["overrides"].items()) {
                if (!obj.is_object()) continue;
                auto& langMap = g_overrides[lang];
                for (auto& [text, target] : obj.items()) {
                    if (target.is_string()) {
                        langMap[s2ws(text)] = target.get<std::string>();
                    }
                }
            }
        }

        // ── Calibration state ─────────────────────────────────────────
        if (j.contains("calibration") && j["calibration"].is_object()) {
            for (auto& [key, c] : j["calibration"].items()) {
                auto sep = key.find('>');
                if (sep == std::string::npos) continue;
                std::string from = key.substr(0, sep);
                std::string to   = key.substr(sep + 1);
                if (from.empty() || to.empty()) continue;

                PairCalibration cal;
                cal.ewmaFpRate     = c.value("ewma_fp",        0.0f);
                cal.ewmaFnRate     = c.value("ewma_fn",        0.0f);
                cal.deltaConfAtMax = c.value("delta_conf_max", 0.0f);
                cal.deltaMargin    = c.value("delta_margin",   0.0f);
                cal.baseConfAtMax  = c.value("base_conf_max",  0.0f);
                cal.baseMargin     = c.value("base_margin",    0.0f);
                cal.batchEvents    = c.value("batch_events",   0);
                cal.baseLoaded     = (cal.baseConfAtMax > 0.0f);

                if (cal.baseLoaded) {
                    g_calibration[{from, to}] = cal;
                    // Re-apply effective params so Config reflects the
                    // previously computed calibration deltas immediately.
                    float effConf = std::clamp(cal.baseConfAtMax + cal.deltaConfAtMax,
                                               CALIB_ABS_MIN_CONF, CALIB_ABS_MAX_CONF);
                    float effMarg = std::clamp(cal.baseMargin + cal.deltaMargin,
                                               CALIB_ABS_MIN_MARGIN, CALIB_ABS_MAX_MARGIN);
                    Config::ApplyAdaptedParams(from, to, effConf, effMarg);
                }
            }
        }
    } catch (...) {
        // Corrupt file — ignore
    }
}

void SavePrefs() {
    json j;
    j["version"] = ws2s(Config::VERSION);
    j["logging_enabled"] = LoggingEnabled.load();

    json ex = json::object();
    for (auto& [lang, texts] : g_exceptions) {
        if (texts.empty()) continue;
        json arr = json::array();
        for (auto& t : texts) {
            arr.push_back(ws2s(t));
        }
        ex[lang] = arr;
    }
    j["exceptions"] = ex;

    json ov = json::object();
    for (auto& [lang, textMap] : g_overrides) {
        if (textMap.empty()) continue;
        json obj = json::object();
        for (auto& [text, target] : textMap) {
            obj[ws2s(text)] = target;
        }
        ov[lang] = obj;
    }
    j["overrides"] = ov;

    // ── Calibration state ─────────────────────────────────────────────
    json cal_obj = json::object();
    for (auto& [k, cal] : g_calibration) {
        if (!cal.baseLoaded) continue;
        std::string key = k.first + ">" + k.second;
        json c;
        c["ewma_fp"]        = cal.ewmaFpRate;
        c["ewma_fn"]        = cal.ewmaFnRate;
        c["delta_conf_max"] = cal.deltaConfAtMax;
        c["delta_margin"]   = cal.deltaMargin;
        c["base_conf_max"]  = cal.baseConfAtMax;
        c["base_margin"]    = cal.baseMargin;
        c["batch_events"]   = cal.batchEvents;
        cal_obj[key] = c;
    }
    j["calibration"] = cal_obj;

    std::ofstream out(g_prefsPath.c_str(), std::ios::trunc);
    if (out) out << j.dump(2);
}

// ── Public API ──────────────────────────────────────────────

void Init() {
    // Resolve %APPDATA%/KeyboardSwitcher
    wchar_t appData[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathW(nullptr, CSIDL_APPDATA, nullptr, 0, appData))) {
        g_dataDir = std::wstring(appData) + L"\\KeyboardSwitcher";
    } else {
        // Fallback to exe directory
        wchar_t exePath[MAX_PATH];
        GetModuleFileNameW(nullptr, exePath, MAX_PATH);
        g_dataDir = std::wstring(exePath);
        auto pos = g_dataDir.find_last_of(L"\\/");
        if (pos != std::wstring::npos) g_dataDir = g_dataDir.substr(0, pos);
    }
    CreateDirectoryW(g_dataDir.c_str(), nullptr);

    g_feedbackPath = g_dataDir + L"\\feedback.jsonl";
    g_prefsPath    = g_dataDir + L"\\user_prefs.json";

    LoadPrefs();
}

// ── Adaptive calibration ─────────────────────────────────────────────

void RecordOutcome(const std::string& fromLang, const std::string& toLang,
                   Outcome outcome) {
    if (fromLang.empty() || toLang.empty() || fromLang == toLang) return;

    PairKey key{fromLang, toLang};
    auto& cal = g_calibration[key];

    // Snapshot the factory baseline once (on first use in a fresh session).
    // On subsequent sessions this is restored from user_prefs.json so we
    // never accumulate deltas on top of already-adapted values.
    if (!cal.baseLoaded) {
        const auto& p = Config::GetParamsForPair(fromLang, toLang);
        cal.baseConfAtMax = p.ConfidenceAtMaxChars;
        cal.baseMargin    = p.MinTop1Top2Margin;
        cal.baseLoaded    = true;
    }

    // Update EWMA: signal is 1.0 for the active outcome type, 0.0 otherwise.
    // TruePositive drives both rates toward 0 (the model is doing its job).
    float fpSig = (outcome == Outcome::FalsePositive) ? 1.0f : 0.0f;
    float fnSig = (outcome == Outcome::FalseNegative) ? 1.0f : 0.0f;
    cal.ewmaFpRate = CALIB_EWMA_ALPHA * fpSig + (1.0f - CALIB_EWMA_ALPHA) * cal.ewmaFpRate;
    cal.ewmaFnRate = CALIB_EWMA_ALPHA * fnSig + (1.0f - CALIB_EWMA_ALPHA) * cal.ewmaFnRate;
    ++cal.batchEvents;

    // Wait for a minimum batch before attempting any adaptation.
    if (cal.batchEvents < CALIB_MIN_EVENTS) return;

    // pressure > 0  → too many FPs → tighten thresholds
    // pressure < 0  → too many FNs → loosen thresholds
    float pressure = cal.ewmaFpRate - cal.ewmaFnRate;

    // Dead zone: absorb noise without touching params; reset the batch counter
    // so the next MIN_EVENTS accumulate fresh signal.
    if (std::abs(pressure) < CALIB_HYSTERESIS_BAND) {
        cal.batchEvents = 0;
        return;
    }

    bool tighten = (pressure > 0.0f);
    float dConf   = tighten ?  CALIB_STEP_CONF   : -CALIB_STEP_CONF;
    float dMargin = tighten ?  CALIB_STEP_MARGIN  : -CALIB_STEP_MARGIN;

    // Clamp cumulative delta within asymmetric limits:
    //   tightening (raising thresholds)  → delta in [0, +MAX_TIGHTEN]
    //   loosening  (lowering thresholds) → delta in [-MAX_LOOSEN, 0]
    // The full range is [-MAX_LOOSEN, +MAX_TIGHTEN] to allow reversal.
    cal.deltaConfAtMax = std::clamp(cal.deltaConfAtMax + dConf,
                                    -CALIB_MAX_LOOSEN_CONF, CALIB_MAX_TIGHTEN_CONF);
    cal.deltaMargin    = std::clamp(cal.deltaMargin + dMargin,
                                    -CALIB_MAX_LOOSEN_MARG, CALIB_MAX_TIGHTEN_MARG);

    // Effective value = base + delta, further constrained by absolute limits
    // so a bad base (hypothetical) can never produce an insane threshold.
    float effConf = std::clamp(cal.baseConfAtMax + cal.deltaConfAtMax,
                               CALIB_ABS_MIN_CONF, CALIB_ABS_MAX_CONF);
    float effMarg = std::clamp(cal.baseMargin + cal.deltaMargin,
                               CALIB_ABS_MIN_MARGIN, CALIB_ABS_MAX_MARGIN);

    Config::ApplyAdaptedParams(fromLang, toLang, effConf, effMarg);

    // Safety: if the delta is pinned at the tightening ceiling while FP
    // pressure persists, the pair is intrinsically ambiguous for this user.
    // Decay the EWMA so the controller does not remain locked at the ceiling
    // forever and can recover if the user's typing patterns change.
    if (tighten &&
        (cal.deltaConfAtMax >= CALIB_MAX_TIGHTEN_CONF - 0.001f ||
         cal.deltaMargin    >= CALIB_MAX_TIGHTEN_MARG - 0.001f)) {
        cal.ewmaFpRate *= 0.70f;   // soft reset — one good TP batch will unlock
    }

    cal.batchEvents = 0;  // reset batch; EWMA memory is preserved

    SavePrefs();
}

void ResetCalibration() {
    // Restore factory params for every pair that was ever calibrated.
    for (auto& [k, cal] : g_calibration) {
        if (!cal.baseLoaded) continue;
        Config::ApplyAdaptedParams(k.first, k.second,
                                   cal.baseConfAtMax, cal.baseMargin);
    }
    g_calibration.clear();
    SavePrefs();
}

void LogEvent(const Entry& entry) {
    if (!LoggingEnabled.load()) return;

    RotateIfNeeded();

    json j;
    j["ts"]             = NowISO8601();
    j["ver"]            = ws2s(Config::VERSION);
    j["type"]           = entry.type;
    j["original_text"]  = ws2s(entry.originalText);
    j["corrected_text"] = ws2s(entry.correctedText);
    j["from_lang"]      = entry.fromLang;
    j["to_lang"]        = entry.toLang;
    j["actual_lang"]    = entry.actualLang;
    j["confidence"]     = entry.confidence;
    j["num_chars"]      = entry.numChars;

    std::ofstream out(g_feedbackPath.c_str(), std::ios::app);
    if (out) out << j.dump(-1) << '\n';
}

void AddException(const std::string& toLang,
                  const std::wstring& correctedText) {
    if (correctedText.empty()) return;
    auto& langSet = g_exceptions[toLang];
    // Cap per-language to avoid unbounded growth
    if (langSet.size() >= MAX_EXCEPTIONS_PER_LANG) return;
    langSet.insert(correctedText);
    SavePrefs();
}

bool IsException(const std::string& toLang,
                 const std::wstring& correctedText) {
    auto it = g_exceptions.find(toLang);
    if (it == g_exceptions.end()) return false;
    const auto& exSet = it->second;

    // Exact match — the user rejected this exact text before.
    if (exSet.count(correctedText) > 0) return true;

    // Prefix match — the proposed text is the beginning of a known-bad
    // correction.  The user is still building up the same word on the
    // same physical keys; block early so the model can't fire on a
    // partial prefix before the full exception text is reached.
    // std::set is sorted, so lower_bound finds the first element >= key.
    auto lb = exSet.lower_bound(correctedText);
    if (lb != exSet.end() &&
        lb->size() >= correctedText.size() &&
        lb->compare(0, correctedText.size(), correctedText) == 0) {
        return true;
    }

    return false;
}

void AddOverride(const std::string& currentLang,
                 const std::wstring& text,
                 const std::string& targetLang) {
    if (text.empty() || currentLang == targetLang) return;
    auto& langMap = g_overrides[currentLang];
    if (langMap.size() >= MAX_EXCEPTIONS_PER_LANG) return;
    langMap[text] = targetLang;
    SavePrefs();
}

std::string GetOverride(const std::string& currentLang,
                        const std::wstring& text) {
    auto it = g_overrides.find(currentLang);
    if (it == g_overrides.end()) return {};
    auto it2 = it->second.find(text);
    if (it2 != it->second.end()) return it2->second;
    return {};
}

void ResetAll() {
    g_exceptions.clear();
    g_overrides.clear();
    // Restore factory params for every calibrated pair BEFORE clearing the
    // calibration map, otherwise Config::PairOverrides keeps the adapted
    // (delta-modified) thresholds in memory until the next restart.
    for (auto& [k, cal] : g_calibration) {
        if (!cal.baseLoaded) continue;
        Config::ApplyAdaptedParams(k.first, k.second,
                                   cal.baseConfAtMax, cal.baseMargin);
    }
    g_calibration.clear();
    DeleteFileW(g_feedbackPath.c_str());
    SavePrefs();
}

void SetLoggingEnabled(bool enabled) {
    LoggingEnabled.store(enabled);
    SavePrefs();
}

std::wstring GetDataDir() {
    return g_dataDir;
}

}  // namespace Feedback

