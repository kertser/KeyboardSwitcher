#include "FeedbackLogger.h"
#include "Config.h"

#include <windows.h>
#include <shlobj.h>          // SHGetFolderPathW
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>
#include <ctime>
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

