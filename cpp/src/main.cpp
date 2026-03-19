// KeyboardSwitcher - C++ Win32 implementation
// Equivalent of the Python main.py

#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif

#include <windows.h>
#include <shellapi.h>
#include <commctrl.h>
#include <strsafe.h>

#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <functional>
#include <atomic>
#include <fstream>
#include <sstream>
#include <cstdarg>
#include <thread>

#include <winhttp.h>
#include <nlohmann/json.hpp>

#include "Config.h"
#include "Languages.h"
#include "InputCache.h"
#include "WindowTracker.h"
#include "TrayIcon.h"
#include "../resources/resource.h"

// ============================================================
// Debug log — writes to ks_debug.log next to the exe
// ============================================================
// Disabled by default.  Toggle from the tray context menu.
// Entries are timestamped.  When the file exceeds MAX_LOG_BYTES
// the oldest half is discarded (roll).
// ============================================================
static std::string  g_debugLogPath;
static std::ofstream g_debugLog;
static bool          g_debugLogEnabled = false;
static constexpr size_t MAX_LOG_BYTES = 512 * 1024;  // 512 KB

static void DbgInit() {
    // Build the path once; the file is only opened when enabled.
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring dir(path);
    size_t pos = dir.find_last_of(L"\\/");
    if (pos != std::wstring::npos) dir = dir.substr(0, pos);
    g_debugLogPath.assign(dir.begin(), dir.end());
    g_debugLogPath += "\\ks_debug.log";
}

static void DbgOpen() {
    if (g_debugLog.is_open()) return;
    // Open in append mode so we keep prior entries from this session
    g_debugLog.open(g_debugLogPath, std::ios::app);
}

static void DbgClose() {
    if (g_debugLog.is_open()) g_debugLog.close();
}

// Roll the log file: keep the newest half, discard the oldest.
static void DbgRollIfNeeded() {
    if (!g_debugLog.is_open()) return;
    g_debugLog.flush();

    // Check current file size
    std::ifstream in(g_debugLogPath, std::ios::ate | std::ios::binary);
    if (!in) return;
    auto fileSize = static_cast<size_t>(in.tellg());
    if (fileSize <= MAX_LOG_BYTES) return;

    // Read entire file, keep the newest half starting at a newline
    in.seekg(0);
    std::string content(fileSize, '\0');
    in.read(&content[0], fileSize);
    in.close();

    size_t cutPos = fileSize / 2;
    // Advance to the next newline so we don't cut mid-line
    size_t nl = content.find('\n', cutPos);
    if (nl != std::string::npos) cutPos = nl + 1;

    std::string kept = content.substr(cutPos);

    // Re-open in truncate mode and write the kept portion
    DbgClose();
    {
        std::ofstream out(g_debugLogPath, std::ios::trunc);
        out << "[LOG ROLLED — older entries discarded]\n";
        out << kept;
    }
    DbgOpen();
}

static void DbgSetEnabled(bool enabled) {
    g_debugLogEnabled = enabled;
    if (enabled) {
        DbgOpen();
    } else {
        DbgClose();
    }
}

// Get current local time as "YYYY-MM-DD HH:MM:SS"
static std::string DbgTimestamp() {
    SYSTEMTIME st;
    GetLocalTime(&st);
    char buf[32];
    snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d",
             st.wYear, st.wMonth, st.wDay,
             st.wHour, st.wMinute, st.wSecond);
    return buf;
}

static std::string ws2s(const std::wstring& w) {
    // Simple lossy conversion for logging
    std::string s;
    for (wchar_t c : w) {
        if (c < 128) s += (char)c;
        else { s += "[U+"; char buf[8]; sprintf(buf, "%04X", (unsigned)c); s += buf; s += "]"; }
    }
    return s;
}

static void Dbg(const char* fmt, ...) {
    if (!g_debugLogEnabled || !g_debugLog.is_open()) return;
    va_list ap;
    va_start(ap, fmt);
    char buf[1024];
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    g_debugLog << "[" << DbgTimestamp() << "] " << buf << std::endl;
    g_debugLog.flush();
    DbgRollIfNeeded();
}

// ============================================================
// Update checker — queries GitHub for the latest release,
//                  downloads the installer, and runs it.
// ============================================================
#define WM_UPDATE_CHECK_RESULT  (WM_APP + 2)
#define WM_UPDATE_DOWNLOAD_DONE (WM_APP + 3)

// Result of the version check (set by bg thread, read by UI thread)
struct UpdateInfo {
    std::string version;      // latest version string (e.g. "1.3.0")
    std::string downloadUrl;  // browser_download_url for the .exe asset
};
static UpdateInfo  g_updateInfo;
static bool        g_updateCheckManual = false;
static std::wstring g_downloadedInstallerPath;  // path to the downloaded file

// Compare semver strings "X.Y.Z". Returns >0 if a > b, 0 if equal, <0 if a < b.
static int CompareVersions(const std::string& a, const std::string& b) {
    int a1 = 0, a2 = 0, a3 = 0, b1 = 0, b2 = 0, b3 = 0;
    sscanf(a.c_str(), "%d.%d.%d", &a1, &a2, &a3);
    sscanf(b.c_str(), "%d.%d.%d", &b1, &b2, &b3);
    if (a1 != b1) return a1 - b1;
    if (a2 != b2) return a2 - b2;
    return a3 - b3;
}

// ── Generic WinHTTP helpers ─────────────────────────────────

// Perform a WinHTTP GET and collect the full response body.
// |host|, |path| are the URL components; |secure| → HTTPS.
// Returns true on success (HTTP 200), body in |outBody|.
static bool WinHttpGet(const std::wstring& host,
                       const std::wstring& path,
                       bool secure,
                       std::string& outBody)
{
    HINTERNET hSession = WinHttpOpen(
        L"KeyboardSwitcher-UpdateCheck/1.0",
        WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) return false;

    HINTERNET hConnect = WinHttpConnect(
        hSession, host.c_str(),
        secure ? INTERNET_DEFAULT_HTTPS_PORT : INTERNET_DEFAULT_HTTP_PORT, 0);
    if (!hConnect) { WinHttpCloseHandle(hSession); return false; }

    DWORD flags = secure ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET hRequest = WinHttpOpenRequest(
        hConnect, L"GET", path.c_str(),
        nullptr, WINHTTP_NO_REFERER,
        WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!hRequest) {
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    // Allow automatic redirects (WinHTTP follows them by default)
    DWORD redirectPolicy = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption(hRequest, WINHTTP_OPTION_REDIRECT_POLICY,
                     &redirectPolicy, sizeof(redirectPolicy));

    BOOL ok = WinHttpSendRequest(hRequest,
        WINHTTP_NO_ADDITIONAL_HEADERS, 0,
        WINHTTP_NO_REQUEST_DATA, 0, 0, 0);
    if (!ok) {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    ok = WinHttpReceiveResponse(hRequest, nullptr);
    if (!ok) {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    // Check HTTP status
    DWORD statusCode = 0;
    DWORD statusSize = sizeof(statusCode);
    WinHttpQueryHeaders(hRequest,
        WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
        WINHTTP_HEADER_NAME_BY_INDEX, &statusCode, &statusSize,
        WINHTTP_NO_HEADER_INDEX);
    if (statusCode != 200) {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    // Read body
    outBody.clear();
    DWORD bytesAvailable = 0;
    do {
        bytesAvailable = 0;
        WinHttpQueryDataAvailable(hRequest, &bytesAvailable);
        if (bytesAvailable > 0) {
            std::vector<char> buf(bytesAvailable);
            DWORD bytesRead = 0;
            WinHttpReadData(hRequest, buf.data(), bytesAvailable, &bytesRead);
            outBody.append(buf.data(), bytesRead);
        }
    } while (bytesAvailable > 0);

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return true;
}

// ── Fetch latest release info from GitHub API ───────────────

// Queries api.github.com and fills |info| with version + download URL.
// Returns true on success.
static bool FetchLatestReleaseInfo(UpdateInfo& info) {
    std::string body;
    if (!WinHttpGet(L"api.github.com",
                    L"/repos/kertser/KeyboardSwitcher/releases/latest",
                    true, body))
        return false;

    try {
        auto j = nlohmann::json::parse(body);

        // Extract version from tag_name (strip leading 'v')
        std::string tag = j.value("tag_name", "");
        if (!tag.empty() && (tag[0] == 'v' || tag[0] == 'V'))
            tag = tag.substr(1);
        if (tag.empty()) return false;
        info.version = tag;

        // Find the NSIS installer asset (*.exe)
        info.downloadUrl.clear();
        if (j.contains("assets") && j["assets"].is_array()) {
            for (auto& asset : j["assets"]) {
                std::string name = asset.value("name", "");
                if (name.size() > 4 &&
                    name.substr(name.size() - 4) == ".exe") {
                    info.downloadUrl = asset.value(
                        "browser_download_url", "");
                    break;
                }
            }
        }
        // Fallback: construct URL from the tag
        if (info.downloadUrl.empty()) {
            info.downloadUrl =
                "https://github.com/kertser/KeyboardSwitcher/releases/download/v"
                + info.version + "/KeyboardSwitcher-" + info.version
                + "-win64.exe";
        }
        return true;
    } catch (...) {
        return false;
    }
}

// ── Download a file from a URL to a local path ──────────────

// Crack a full URL into host + path components.
static bool CrackUrl(const std::string& url,
                     std::wstring& host, std::wstring& path, bool& secure)
{
    std::wstring wUrl(url.begin(), url.end());
    URL_COMPONENTS uc = {};
    uc.dwStructSize = sizeof(uc);

    wchar_t hostBuf[256] = {};
    wchar_t pathBuf[2048] = {};
    uc.lpszHostName    = hostBuf;
    uc.dwHostNameLength = _countof(hostBuf);
    uc.lpszUrlPath     = pathBuf;
    uc.dwUrlPathLength = _countof(pathBuf);

    if (!WinHttpCrackUrl(wUrl.c_str(), 0, 0, &uc))
        return false;

    host = hostBuf;
    path = pathBuf;
    secure = (uc.nScheme == INTERNET_SCHEME_HTTPS);
    return true;
}

// Download |url| to |destPath|.  Follows redirects.
// Returns true on success.
static bool DownloadFileTo(const std::string& url,
                           const std::wstring& destPath)
{
    std::wstring host, path;
    bool secure = true;
    if (!CrackUrl(url, host, path, secure))
        return false;

    HINTERNET hSession = WinHttpOpen(
        L"KeyboardSwitcher-UpdateCheck/1.0",
        WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
        WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) return false;

    HINTERNET hConnect = WinHttpConnect(
        hSession, host.c_str(),
        secure ? INTERNET_DEFAULT_HTTPS_PORT : INTERNET_DEFAULT_HTTP_PORT, 0);
    if (!hConnect) { WinHttpCloseHandle(hSession); return false; }

    DWORD flags = secure ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET hRequest = WinHttpOpenRequest(
        hConnect, L"GET", path.c_str(),
        nullptr, WINHTTP_NO_REFERER,
        WINHTTP_DEFAULT_ACCEPT_TYPES, flags);
    if (!hRequest) {
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    DWORD redirectPolicy = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
    WinHttpSetOption(hRequest, WINHTTP_OPTION_REDIRECT_POLICY,
                     &redirectPolicy, sizeof(redirectPolicy));

    BOOL ok = WinHttpSendRequest(hRequest,
        WINHTTP_NO_ADDITIONAL_HEADERS, 0,
        WINHTTP_NO_REQUEST_DATA, 0, 0, 0);
    if (!ok) {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    ok = WinHttpReceiveResponse(hRequest, nullptr);
    if (!ok) {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    // Open local file
    HANDLE hFile = CreateFileW(destPath.c_str(), GENERIC_WRITE, 0,
                               nullptr, CREATE_ALWAYS,
                               FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    // Stream response to disk
    bool success = true;
    DWORD bytesAvailable = 0;
    do {
        bytesAvailable = 0;
        WinHttpQueryDataAvailable(hRequest, &bytesAvailable);
        if (bytesAvailable > 0) {
            std::vector<char> buf(bytesAvailable);
            DWORD bytesRead = 0;
            if (WinHttpReadData(hRequest, buf.data(),
                                bytesAvailable, &bytesRead)) {
                DWORD written = 0;
                if (!WriteFile(hFile, buf.data(), bytesRead,
                               &written, nullptr) ||
                    written != bytesRead) {
                    success = false;
                    break;
                }
            } else {
                success = false;
                break;
            }
        }
    } while (bytesAvailable > 0);

    CloseHandle(hFile);
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);

    if (!success) DeleteFileW(destPath.c_str());
    return success;
}

// Forward declarations (used by update checker, defined here early)
static HWND g_hwndHidden = nullptr;
static TrayIcon g_trayIcon;

// Run the update check in a background thread.
// |manual| = true when the user clicked "Check for Updates".
// |delaySec| = seconds to wait before checking (used for startup delay).
static void CheckForUpdatesAsync(bool manual, int delaySec = 0) {
    g_updateCheckManual = manual;
    std::thread([delaySec]() {
        if (delaySec > 0) Sleep(delaySec * 1000);
        UpdateInfo info;
        if (FetchLatestReleaseInfo(info))
            g_updateInfo = info;
        else
            g_updateInfo = {};  // clear on failure
        if (g_hwndHidden)
            PostMessage(g_hwndHidden, WM_UPDATE_CHECK_RESULT, 0, 0);
    }).detach();
}

// Start downloading the installer in a background thread.
static void DownloadUpdateAsync() {
    std::thread([]() {
        // Build destination path: %TEMP%\KeyboardSwitcher-X.Y.Z-win64.exe
        wchar_t tempDir[MAX_PATH] = {};
        GetTempPathW(MAX_PATH, tempDir);
        std::wstring dest = tempDir;
        dest += L"KeyboardSwitcher-";
        dest += std::wstring(g_updateInfo.version.begin(),
                             g_updateInfo.version.end());
        dest += L"-win64.exe";

        bool ok = DownloadFileTo(g_updateInfo.downloadUrl, dest);
        g_downloadedInstallerPath = ok ? dest : L"";
        if (g_hwndHidden)
            PostMessage(g_hwndHidden, WM_UPDATE_DOWNLOAD_DONE, 0, 0);
    }).detach();
}

// Show the update-check result dialog on the UI thread.
static void HandleUpdateCheckResult() {
    // Narrow wchar_t version to std::string (ASCII digits + dots only)
    std::string current;
    for (const wchar_t* p = Config::VERSION; *p; ++p)
        current += static_cast<char>(*p);
    const std::string& latest = g_updateInfo.version;

    if (latest.empty()) {
        // Network failure — only report if the user asked explicitly
        if (g_updateCheckManual) {
            MessageBoxW(nullptr,
                L"Could not check for updates.\n"
                L"Please verify your internet connection.",
                L"Keyboard Switcher \x2014 Update Check",
                MB_OK | MB_ICONWARNING);
        }
        return;
    }

    if (CompareVersions(latest, current) > 0) {
        // Newer version available
        std::wstring msg =
            L"A new version is available!\n\n"
            L"Current version: v" + std::wstring(Config::VERSION) + L"\n"
            L"Latest version:  v" + std::wstring(latest.begin(), latest.end()) + L"\n\n"
            L"Download and install now?";
        int res = MessageBoxW(nullptr, msg.c_str(),
            L"Keyboard Switcher \x2014 Update Available",
            MB_YESNO | MB_ICONINFORMATION);
        if (res == IDYES) {
            // Start downloading — the result arrives via
            // WM_UPDATE_DOWNLOAD_DONE on the UI thread.
            DownloadUpdateAsync();
        }
    } else {
        // Up to date — only show if the user asked explicitly
        if (g_updateCheckManual) {
            MessageBoxW(nullptr,
                (L"You are running the latest version (v" +
                 std::wstring(Config::VERSION) + L").").c_str(),
                L"Keyboard Switcher \x2014 Update Check",
                MB_OK | MB_ICONINFORMATION);
        }
    }
}

// Handle the completed download: launch the installer and exit.
static void HandleUpdateDownloadDone() {
    if (g_downloadedInstallerPath.empty()) {
        MessageBoxW(nullptr,
            L"Download failed.\n"
            L"Please download the update manually from:\n"
            L"https://github.com/kertser/KeyboardSwitcher/releases/latest",
            L"Keyboard Switcher \x2014 Update",
            MB_OK | MB_ICONERROR);
        return;
    }

    // Launch the installer — the NSIS installer handles closing
    // the running instance and replacing the exe.
    HINSTANCE hInst = ShellExecuteW(
        nullptr, L"open",
        g_downloadedInstallerPath.c_str(),
        nullptr, nullptr, SW_SHOWNORMAL);

    if ((INT_PTR)hInst > 32) {
        // Installer launched successfully — exit so it can replace us
        g_trayIcon.Remove();
        PostQuitMessage(0);
    } else {
        MessageBoxW(nullptr,
            L"Failed to launch the installer.\n"
            L"The downloaded file is in your temp folder.",
            L"Keyboard Switcher \x2014 Update",
            MB_OK | MB_ICONERROR);
    }
}

// ============================================================
// Globals
// ============================================================
static HHOOK g_keyboardHook = nullptr;
static HHOOK g_mouseHook = nullptr;
// g_hwndHidden is forward-declared above the update checker

static InputCache          g_cache;
static DetectionHistory    g_history;
static WindowTracker*      g_windows = nullptr;
static LanguageDetector*   g_detector = nullptr;
// g_trayIcon is forward-declared above the update checker
static std::atomic<bool>   g_isSendingInput{false}; // re-entrancy guard

// Focus tracking — updated by HandleFocusChange(), read by hooks.
// All accesses are on the main (message-loop) thread — no mutex needed.
static HWND          g_lastForegroundHwnd = nullptr;
static size_t        g_lastContextHash    = 0;
static HWINEVENTHOOK g_winEventHook       = nullptr;

// ============================================================
// Last correction state for Esc-to-undo
// ============================================================
struct LastCorrectionInfo {
    std::wstring originalText;      // mistyped chars (what was on screen)
    std::wstring correctedText;     // what was pasted as replacement
    std::string  originalLang;      // keyboard layout before correction
    std::string  correctedLang;     // keyboard layout after correction
    HWND         hwnd = nullptr;    // window where correction happened
    size_t       ctxHash = 0;       // context hash
    std::vector<wchar_t> postChars; // chars typed after correction (for buffered undo)
    bool         valid = false;     // is undo available?
};
static LastCorrectionInfo g_lastCorrection;
static constexpr size_t MAX_UNDO_POST_CHARS = 100;

// ============================================================
// Helper: get the executable directory
// ============================================================
static std::wstring GetExeDirectory() {
    wchar_t path[MAX_PATH];
    GetModuleFileNameW(nullptr, path, MAX_PATH);
    std::wstring dir(path);
    size_t pos = dir.find_last_of(L"\\/");
    if (pos != std::wstring::npos) dir = dir.substr(0, pos);
    return dir;
}

// ============================================================
// Helper: get current keyboard layout language string
// ============================================================
static std::string GetCurrentKeyboardLayout() {
    HWND hwnd = GetForegroundWindow();
    DWORD threadId = GetWindowThreadProcessId(hwnd, nullptr);
    HKL hkl = ::GetKeyboardLayout(threadId);
    LANGID langId = LOWORD((DWORD_PTR)hkl);
    return Config::GetLanguageFromId(langId);
}

// ============================================================
// Helper: human-readable language name for tooltip display
// ============================================================
static std::wstring GetLanguageDisplayName(const std::string& lang) {
    if (lang == "en") return L"English";
    if (lang == "ru") return L"\x0420\x0443\x0441\x0441\x043A\x0438\x0439";  // Русский
    if (lang == "he") return L"\x05E2\x05D1\x05E8\x05D9\x05EA";              // עברית
    return L"Unknown";
}

// ============================================================
// Helper: update tray tooltip to reflect current layout
// ============================================================
static void UpdateTrayTooltip() {
    std::string lang = GetCurrentKeyboardLayout();
    g_trayIcon.UpdateTooltip(L"Keyboard Switcher \x2014 " + GetLanguageDisplayName(lang));
}

// ============================================================
// Helper: compute title hash for tab-aware window tracking
// ============================================================
// For other-process windows GetWindowTextW reads the cached
// internal copy without sending WM_GETTEXT — fast & safe from hooks.
//
// The title is *normalized* before hashing so that cosmetic
// changes (Notepad++ prepending "*" for unsaved, browsers
// showing "(3)" unread-count badges, etc.) do NOT create a
// separate tracking entry.
static size_t GetWindowTitleHash(HWND hwnd) {
    wchar_t buf[256] = {};
    GetWindowTextW(hwnd, buf, 256);
    std::wstring t(buf);

    // Strip leading modification indicators: *, •, ●  and spaces
    size_t start = 0;
    while (start < t.size() &&
           (t[start] == L'*' || t[start] == L'\u2022' /*•*/  ||
            t[start] == L'\u25CF' /*●*/ || t[start] == L' '))
        ++start;

    // Strip leading notification count: "(N) "
    if (start < t.size() && t[start] == L'(') {
        size_t close = t.find(L')', start);
        if (close != std::wstring::npos && close - start <= 5) {
            bool allDigits = true;
            for (size_t i = start + 1; i < close; ++i) {
                if (!iswdigit(t[i])) { allDigits = false; break; }
            }
            if (allDigits) {
                start = close + 1;
                while (start < t.size() && t[start] == L' ') ++start;
            }
        }
    }

    return std::hash<std::wstring>{}(t.substr(start));
}

// ============================================================
// Helper: get the focused child control inside a top-level window
// ============================================================
// Uses GetGUIThreadInfo() which reads kernel state — no IPC,
// safe and fast to call from low-level hooks.
// Returns nullptr when the focus is on the top-level window itself
// or when the info cannot be obtained (modern single-HWND frameworks
// like WPF, Electron, UWP — graceful degradation).
static HWND GetFocusedChildHwnd(HWND topLevel) {
    if (!topLevel) return nullptr;
    DWORD threadId = GetWindowThreadProcessId(topLevel, nullptr);
    GUITHREADINFO gti = {};
    gti.cbSize = sizeof(gti);
    if (GetGUIThreadInfo(threadId, &gti) &&
        gti.hwndFocus && gti.hwndFocus != topLevel) {
        return gti.hwndFocus;
    }
    return nullptr;
}

// ============================================================
// Helper: compute a context hash for window/tab/field tracking
// ============================================================
// Uses the normalized window title to distinguish tabs/documents.
// The title is normalized (leading *, •, ●, and "(N)" stripped)
// so cosmetic changes don't create separate tracking entries.
//
// Note: we intentionally do NOT mix in the focused child HWND.
// Chrome (and other multi-process apps) change their focused
// child between WinEvent and mouse-click calls, making the hash
// unstable and causing saved languages to become unfindable.
static size_t GetWindowContextHash(HWND hwnd) {
    return GetWindowTitleHash(hwnd);
}


// ============================================================
// Helper: find the actual installed HKL for a language
// ============================================================
static HKL FindInstalledHKL(const std::string& lang) {
    // Get the target LANGID
    LANGID targetLangId = 0;
    for (const auto& [id, name] : Config::LANGUAGE_ID) {
        if (name == lang) { targetLangId = id; break; }
    }
    if (targetLangId == 0)
        return Config::GetHKLFromLanguage(lang); // fallback to hardcoded

    // Enumerate actually installed keyboard layouts
    int count = GetKeyboardLayoutList(0, nullptr);
    if (count > 0) {
        std::vector<HKL> layouts(count);
        GetKeyboardLayoutList(count, layouts.data());
        for (HKL h : layouts) {
            if (LOWORD((DWORD_PTR)h) == targetLangId)
                return h;
        }
    }
    return Config::GetHKLFromLanguage(lang); // fallback
}

// ============================================================
// Helper: change the foreground window keyboard layout
// ============================================================
static void ChangeKeyboardLayout(const std::string& lang) {
    HWND hwnd = GetForegroundWindow();
    if (!hwnd) return;

    HKL hkl = FindInstalledHKL(lang);

    // Use SendMessage (synchronous) so the layout is guaranteed to
    // have changed by the time this function returns.
    SendMessage(hwnd, WM_INPUTLANGCHANGEREQUEST, 0, (LPARAM)hkl);
}

// ============================================================
// Centralized focus-change handler
// ============================================================
// Called when the foreground window or its title (tab) changes,
// or when the user clicks (new typing session).  All callers
// are on the main (message-loop) thread — no locking needed for
// the g_last* globals.
//
// forceSearch: when true (mouse click), start fresh detection even
//              in a confirmed context — user may want to type in a
//              different language now.
static void HandleFocusChange(bool forceSearch = false) {
    if (g_isSendingInput.load()) return;

    HWND hwnd = GetForegroundWindow();
    if (!hwnd) return;

    // Skip focus changes to our own windows (tray menu, flyout panel)
    DWORD pid = 0;
    GetWindowThreadProcessId(hwnd, &pid);
    if (pid == GetCurrentProcessId()) return;

    size_t ctxHash = GetWindowContextHash(hwnd);
    bool sameHwnd = (hwnd == g_lastForegroundHwnd);
    bool titleOnlyChanged = sameHwnd && (ctxHash != g_lastContextHash);

    // Title-only changes while typing (cache non-empty) are cosmetic
    // (e.g. Notepad updating title bar with content). Ignore them.
    // Title-only changes with empty cache may be real tab switches.
    bool allowTitleChange = (g_cache.Size() == 0);
    bool contextChanged = !sameHwnd || (titleOnlyChanged && allowTitleChange);

    Dbg("HandleFocusChange: hwnd=%p last=%p ctxHash=%zu lastCtx=%zu "
        "sameHwnd=%d titleOnly=%d allowTitle=%d changed=%d force=%d cache=%zu",
        hwnd, g_lastForegroundHwnd, ctxHash, g_lastContextHash,
        sameHwnd, titleOnlyChanged, allowTitleChange, contextChanged,
        forceSearch, g_cache.Size());

    if (contextChanged) {
        // Invalidate undo — user moved to a different context
        g_lastCorrection.valid = false;

        size_t prevCtxHash = g_lastContextHash;
        HWND prevHwnd = g_lastForegroundHwnd;

        g_lastForegroundHwnd = hwnd;
        g_lastContextHash    = ctxHash;

        std::string currentLayout = GetCurrentKeyboardLayout();
        std::string savedLang;

        if (g_windows && Config::SaveWindowState.load()) {
            savedLang = g_windows->GetLanguage(hwnd, ctxHash);

            // If not found under new hash but same HWND, try old hash
            // and migrate (handles Notepad title drift)
            if (savedLang.empty() && sameHwnd && prevHwnd == hwnd) {
                savedLang = g_windows->GetLanguage(hwnd, prevCtxHash);
                if (!savedLang.empty()) {
                    g_windows->SetLanguage(hwnd, ctxHash, savedLang);
                    Dbg("MIGRATE-LANG: %s from hash %zu to %zu",
                        savedLang.c_str(), prevCtxHash, ctxHash);
                }
            }

            if (!savedLang.empty()) {
                // Known, confirmed context — restore its language
                if (savedLang != currentLayout) {
                    ChangeKeyboardLayout(savedLang);
                    Dbg("RESTORE-LANG: %s for hwnd=%p ctxHash=%zu",
                        savedLang.c_str(), hwnd, ctxHash);
                }
                Config::LastSetting = savedLang;
            } else {
                // Unvisited context — do NOT record yet; let detection
                // or a manual layout switch confirm the language first.
                Config::LastSetting = currentLayout;
            }
        } else {
            Config::LastSetting = currentLayout;
        }

        g_cache.Clear();
        g_history.Clear();

        // Always enable detection on context change (new window or tab
        // switch).  This catches layout mistakes even when returning to
        // a previously confirmed tab.  If the user types in the correct
        // language, detection simply confirms silently (no visible
        // correction).  The forceSearch flag only governs the same-
        // context branch below (repeated clicks in the same spot).
        Config::SEARCH.store(true);
        Dbg("SEARCH set to 1 (contextChanged, savedLang=%s)",
            savedLang.c_str());

        UpdateTrayTooltip();
    } else {
        // Same context, title may have changed but we're ignoring it.
        // On forced search (click), allow fresh detection.
        // Otherwise just clear cache for new typing position.

        std::string savedLang;
        if (g_windows && Config::SaveWindowState.load()) {
            savedLang = g_windows->GetLanguage(hwnd, g_lastContextHash);
        }

        if (forceSearch) {
            // User clicked — start fresh detection session
            g_cache.Clear();
            g_history.Clear();
            Config::SEARCH.store(true);
            g_lastCorrection.valid = false;
            Dbg("FORCE-SEARCH: click in same context, SEARCH=1");
        } else {
            // Just position change (e.g. arrow keys), keep state
            // but clear cache for new typing segment
            if (g_cache.Size() == 0) {
                // Only clear history if no active typing
                g_history.Clear();
            }
        }
    }
}

// ============================================================
// WinEvent callback — fired by the OS on foreground changes
// ============================================================
// Installed with WINEVENT_OUTOFCONTEXT so the callback is
// delivered on the message-loop thread via the message pump.
static void CALLBACK WinEventProc(
    HWINEVENTHOOK /*hWinEventHook*/, DWORD event,
    HWND /*hwnd*/, LONG /*idObject*/, LONG /*idChild*/,
    DWORD /*dwEventThread*/, DWORD /*dwmsEventTime*/)
{
    if (event == EVENT_SYSTEM_FOREGROUND) {
        if (!Config::EnableSwitcher.load()) return;
        HandleFocusChange();
    }
}


// ============================================================
// Helper: send N backspaces via SendInput
// ============================================================
static void SendBackspaces(size_t count) {
    for (size_t i = 0; i < count; ++i) {
        INPUT input = {};  // Zero-initialize every field
        input.type = INPUT_KEYBOARD;
        input.ki.wVk = VK_BACK;
        SendInput(1, &input, sizeof(INPUT));

        input.ki.dwFlags = KEYEVENTF_KEYUP;
        SendInput(1, &input, sizeof(INPUT));
    }
    // Let app process backspaces before we send new text
    Sleep(20);
}

// ============================================================
// Helper: send a string via clipboard paste
// ============================================================
static void SendString(const std::vector<wchar_t>& chars) {
    if (chars.empty()) return;

    // Open clipboard with retries
    int retries = 10;
    while (!OpenClipboard(nullptr) && retries-- > 0) {
        Sleep(10);
    }
    if (retries <= 0) {
        Dbg("SendString: failed to open clipboard");
        return;
    }

    // ── Save existing clipboard text so we can restore it after paste ──
    std::wstring savedClipText;
    bool hadClipText = false;
    {
        HANDLE hData = GetClipboardData(CF_UNICODETEXT);
        if (hData) {
            const wchar_t* p = static_cast<const wchar_t*>(GlobalLock(hData));
            if (p) {
                savedClipText = p;
                hadClipText = true;
                GlobalUnlock(hData);
            }
        }
    }

    EmptyClipboard();

    // Allocate and copy text to clipboard
    size_t size = (chars.size() + 1) * sizeof(wchar_t);
    HGLOBAL hMem = GlobalAlloc(GMEM_MOVEABLE, size);
    if (!hMem) {
        CloseClipboard();
        return;
    }

    wchar_t* pMem = static_cast<wchar_t*>(GlobalLock(hMem));
    if (!pMem) {
        GlobalFree(hMem);
        CloseClipboard();
        return;
    }

    memcpy(pMem, chars.data(), chars.size() * sizeof(wchar_t));
    pMem[chars.size()] = L'\0';
    GlobalUnlock(hMem);

    if (!SetClipboardData(CF_UNICODETEXT, hMem)) {
        GlobalFree(hMem);
        CloseClipboard();
        return;
    }

    CloseClipboard();

    // Delay to ensure clipboard is ready
    Sleep(10);

    // Paste via Ctrl+V (SendInput).
    // WM_PASTE via SendMessageW doesn't work reliably in Chrome —
    // the HWND from GetGUIThreadInfo may not forward the message to
    // the renderer.  Ctrl+V works universally: Chrome, WinUI
    // (Notepad), classic Win32, Electron, etc.
    // The injected events carry LLKHF_INJECTED so our hook passes
    // them through without caching or detection.
    INPUT inputs[4] = {};
    inputs[0].type = INPUT_KEYBOARD;
    inputs[0].ki.wVk = VK_CONTROL;

    inputs[1].type = INPUT_KEYBOARD;
    inputs[1].ki.wVk = 'V';

    inputs[2].type = INPUT_KEYBOARD;
    inputs[2].ki.wVk = 'V';
    inputs[2].ki.dwFlags = KEYEVENTF_KEYUP;

    inputs[3].type = INPUT_KEYBOARD;
    inputs[3].ki.wVk = VK_CONTROL;
    inputs[3].ki.dwFlags = KEYEVENTF_KEYUP;

    SendInput(4, inputs, sizeof(INPUT));

    // Let the target app process the Ctrl+V
    Sleep(30);

    // ── Restore previous clipboard content ──
    retries = 10;
    while (!OpenClipboard(nullptr) && retries-- > 0) {
        Sleep(10);
    }
    if (retries > 0) {
        EmptyClipboard();
        if (hadClipText && !savedClipText.empty()) {
            size_t rSize = (savedClipText.size() + 1) * sizeof(wchar_t);
            HGLOBAL hRestore = GlobalAlloc(GMEM_MOVEABLE, rSize);
            if (hRestore) {
                wchar_t* pRestore = static_cast<wchar_t*>(GlobalLock(hRestore));
                if (pRestore) {
                    memcpy(pRestore, savedClipText.c_str(), rSize);
                    GlobalUnlock(hRestore);
                    SetClipboardData(CF_UNICODETEXT, hRestore);
                } else {
                    GlobalFree(hRestore);
                }
            }
        }
        CloseClipboard();
    }
}

// ============================================================
// Helper: translate a virtual key to a wide character
// ============================================================
static wchar_t VkToWchar(DWORD vkCode) {
    // Build keyboard state from the actual hardware key states.
    // GetKeyboardState() does NOT work in a low-level hook because
    // the hook runs on our thread, not the foreground app's thread.
    // GetAsyncKeyState() queries the real physical key state.
    BYTE keyboardState[256] = {};
    bool shiftDown = false;
    if (GetAsyncKeyState(VK_SHIFT) & 0x8000) {
        keyboardState[VK_SHIFT] = 0x80;
        shiftDown = true;
    }
    if (GetAsyncKeyState(VK_LSHIFT) & 0x8000)
        keyboardState[VK_LSHIFT] = 0x80;
    if (GetAsyncKeyState(VK_RSHIFT) & 0x8000)
        keyboardState[VK_RSHIFT] = 0x80;
    if (GetAsyncKeyState(VK_CONTROL) & 0x8000)
        keyboardState[VK_CONTROL] = 0x80;
    if (GetAsyncKeyState(VK_MENU) & 0x8000)
        keyboardState[VK_MENU] = 0x80;
    // CapsLock toggle state is in the low bit
    if (GetKeyState(VK_CAPITAL) & 0x0001)
        keyboardState[VK_CAPITAL] = 0x01;

    wchar_t buffer[4] = {};
    HWND hwnd = GetForegroundWindow();
    DWORD threadId = GetWindowThreadProcessId(hwnd, nullptr);
    HKL hkl = ::GetKeyboardLayout(threadId);
    UINT scanCode = MapVirtualKeyEx(vkCode, MAPVK_VK_TO_VSC, hkl);
    int result = ToUnicodeEx(vkCode, scanCode, keyboardState, buffer, 4, 0, hkl);

    if (result > 0) {
        wchar_t ch = buffer[0];

        // On non-Latin layouts (Hebrew, Russian, etc.) pressing Shift can
        // produce a Latin letter (e.g. Shift+Z on Hebrew → 'Z' instead of 'ז').
        // Detect this and retry without Shift to get the correct character.
        if (shiftDown && ch >= L'A' && ch <= L'Z') {
            LANGID langId = LOWORD((DWORD_PTR)hkl);
            std::string lang = Config::GetLanguageFromId(langId);
            if (lang != "en") {
                // Retry without Shift
                BYTE noShiftState[256] = {};
                if (GetKeyState(VK_CAPITAL) & 0x0001)
                    noShiftState[VK_CAPITAL] = 0x01;
                wchar_t buffer2[4] = {};
                int result2 = ToUnicodeEx(vkCode, scanCode, noShiftState, buffer2, 4, 0, hkl);
                if (result2 > 0)
                    return buffer2[0];
            }
        }
        return ch;
    }
    return 0;
}

// ============================================================
// Low-level keyboard hook procedure
// ============================================================
static LRESULT CALLBACK LowLevelKeyboardProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode != HC_ACTION)
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
    if (!Config::EnableSwitcher.load())
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);

    KBDLLHOOKSTRUCT* pKb = reinterpret_cast<KBDLLHOOKSTRUCT*>(lParam);

    // During retyping: let our own injected keystrokes through,
    // but EAT (block) real user keystrokes so they don't collide.
    if (g_isSendingInput.load()) {
        if (pKb->flags & LLKHF_INJECTED)
            return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam); // our SendInput – pass
        return 1; // real user keystroke during retyping – eat it
    }
    // Outside of retyping, ignore injected keystrokes (avoid re-processing)
    if (pKb->flags & LLKHF_INJECTED)
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);

    // Only process key-down events
    if (wParam != WM_KEYDOWN && wParam != WM_SYSKEYDOWN)
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);

    DWORD vk = pKb->vkCode;

    // --- Esc to undo the last correction ---
    if (vk == VK_ESCAPE && g_lastCorrection.valid) {
        // Only undo if still in the same window
        if (GetForegroundWindow() == g_lastCorrection.hwnd) {
            // Convert post-correction chars back to original language
            std::wstring postText(g_lastCorrection.postChars.begin(),
                                  g_lastCorrection.postChars.end());
            std::wstring convertedPost;
            if (!postText.empty()) {
                convertedPost = ConvertTextBidirectional(
                    postText,
                    Layouts::GetLayoutForLanguage(g_lastCorrection.correctedLang),
                    Layouts::GetLayoutForLanguage(g_lastCorrection.originalLang));
            }

            Dbg("UNDO: reverting '%s'+'%s' back to '%s'+'%s', lang %s -> %s",
                ws2s(g_lastCorrection.correctedText).c_str(),
                ws2s(postText).c_str(),
                ws2s(g_lastCorrection.originalText).c_str(),
                ws2s(convertedPost).c_str(),
                g_lastCorrection.correctedLang.c_str(),
                g_lastCorrection.originalLang.c_str());

            g_isSendingInput.store(true);
            BOOL blocked = BlockInput(TRUE);

            // Remove corrected text + any post-correction characters
            size_t totalChars = g_lastCorrection.correctedText.size()
                              + g_lastCorrection.postChars.size();
            SendBackspaces(totalChars);

            // Switch back to original language
            ChangeKeyboardLayout(g_lastCorrection.originalLang);

            // Retype original text + converted post-correction text
            std::wstring fullOriginal = g_lastCorrection.originalText + convertedPost;
            std::vector<wchar_t> origChars(fullOriginal.begin(), fullOriginal.end());
            SendString(origChars);

            if (blocked) BlockInput(FALSE);
            g_isSendingInput.store(false);

            // Restore saved window state to original language
            if (g_windows && Config::SaveWindowState.load()) {
                g_windows->SetLanguage(g_lastCorrection.hwnd,
                                       g_lastCorrection.ctxHash,
                                       g_lastCorrection.originalLang);
            }
            Config::LastSetting = g_lastCorrection.originalLang;

            // Disable detection — user explicitly chose to keep their text
            Config::SEARCH.store(false);
            g_cache.Clear();
            g_history.Clear();

            g_lastCorrection.valid = false;
            UpdateTrayTooltip();

            return 1; // block the Esc key
        }
        g_lastCorrection.valid = false;
    }

    // --- Arrow / navigation keys break word continuity ---
    if (vk == VK_LEFT || vk == VK_RIGHT || vk == VK_UP || vk == VK_DOWN ||
        vk == VK_HOME || vk == VK_END || vk == VK_PRIOR || vk == VK_NEXT) {
        g_cache.Clear();
        g_history.Clear();
        g_lastCorrection.valid = false;
        Dbg("NAV-KEY: vk=0x%02X — cache cleared", vk);
    }

    // --- Invalidate undo on paragraph break or buffer overflow ---
    if (g_lastCorrection.valid && vk != VK_ESCAPE) {
        if (vk == VK_RETURN ||
            g_lastCorrection.postChars.size() >= MAX_UNDO_POST_CHARS) {
            g_lastCorrection.valid = false;
        }
    }

    // --- Focus-change detection (belt-and-suspenders) ---
    // EVENT_SYSTEM_FOREGROUND handles most focus changes, but we
    // also check here to catch edge cases where the WinEvent
    // callback hasn't been dispatched yet, and to detect in-window
    // tab switches (title changes) and focus-control changes
    // (Tab between text fields in classic Win32 windows).
    //
    // Context hash is only checked when the cache is empty (start of
    // a new typing session).  This avoids spurious resets when apps
    // like Notepad++ update the title mid-typing (e.g. prepending
    // "*" for unsaved changes).
    {
        HWND currentHwnd = GetForegroundWindow();
        if (currentHwnd) {
            bool hwndChanged   = (currentHwnd != g_lastForegroundHwnd);
            bool contextChanged = false;
            if (!hwndChanged && g_cache.Size() == 0) {
                // Same HWND, no chars yet — check for tab switch
                // or focused-control change (Tab between fields).
                size_t ctxHash = GetWindowContextHash(currentHwnd);
                contextChanged = (ctxHash != g_lastContextHash);
            }
            if (hwndChanged || contextChanged) {
                HandleFocusChange();
            }
        }
    }

    // --- Track manual layout changes ---
    // If the user manually switched the layout (e.g. Alt+Shift),
    // update the saved language and disable detection — the user
    // already chose a language.  Only applies when a language was
    // previously confirmed; for unvisited contexts detection will
    // handle whatever layout the user is on.
    {
        if (g_windows && g_lastForegroundHwnd) {
            std::string windowLang =
                g_windows->GetLanguage(g_lastForegroundHwnd, g_lastContextHash);
            if (!windowLang.empty()) {
                std::string currentLayout = GetCurrentKeyboardLayout();
                if (windowLang != currentLayout) {
                    // Layout changed since last confirmed → manual switch
                    Dbg("MANUAL-SWITCH: %s → %s for hwnd=%p ctxHash=%zu",
                        windowLang.c_str(), currentLayout.c_str(),
                        g_lastForegroundHwnd, g_lastContextHash);

                    if (Config::SaveWindowState.load()) {
                        g_windows->SetLanguage(g_lastForegroundHwnd,
                                               g_lastContextHash, currentLayout);

                        // Also save under live hash if different
                        size_t liveCtxHash = GetWindowContextHash(g_lastForegroundHwnd);
                        if (liveCtxHash != g_lastContextHash) {
                            g_windows->SetLanguage(g_lastForegroundHwnd,
                                                   liveCtxHash, currentLayout);
                            g_lastContextHash = liveCtxHash;
                        }
                    }
                    Config::LastSetting = currentLayout;
                    g_cache.Clear();
                    g_history.Clear();
                    Config::SEARCH.store(false);
                    g_lastCorrection.valid = false;
                    UpdateTrayTooltip();
                }
            }
        }
    }

    // --- Handle character input ---
    bool shiftHeld = (GetAsyncKeyState(VK_SHIFT) & 0x8000) != 0;
    bool capsOn = (GetKeyState(VK_CAPITAL) & 0x0001) != 0;
    bool isUpperIntent = shiftHeld ^ capsOn; // XOR: Shift flips CapsLock

    if (vk == VK_BACK) {
        g_cache.DelChar();
        // Undo buffer: pop last post-correction char, or invalidate
        // if backspace reaches into the corrected text itself
        if (g_lastCorrection.valid) {
            if (!g_lastCorrection.postChars.empty())
                g_lastCorrection.postChars.pop_back();
            else
                g_lastCorrection.valid = false;
        }
    } else if (vk == VK_SPACE) {
        g_cache.PushChar(L' ', false);
        if (g_lastCorrection.valid)
            g_lastCorrection.postChars.push_back(L' ');
    } else if (vk == VK_RETURN) {
        // Enter: placeholder (same as Python version)
    } else if ((vk >= 0x30 && vk <= 0x5A) || (vk >= VK_OEM_1 && vk <= VK_OEM_3) ||
               (vk >= VK_OEM_4 && vk <= VK_OEM_8)) {
        wchar_t ch = VkToWchar(vk);
        if (ch != 0) {
            g_cache.PushChar(ch, isUpperIntent);
            if (g_lastCorrection.valid)
                g_lastCorrection.postChars.push_back(ch);
        }
    }

    // --- Language detection (adaptive confidence) ---
    size_t cacheSize = g_cache.Size();
    if (Config::SEARCH.load() &&
        cacheSize >= static_cast<size_t>(Config::EarlyDetectionMinChars) &&
        g_detector)
    {
        // Note: per-pair parameters may require more chars than the global
        // default minimum.  The global check here is a fast early-out;
        // TypoResilientDetect performs the pair-specific min-chars check
        // after it determines the best candidate language.
        std::string currentLangId = GetCurrentKeyboardLayout();
        std::wstring text = g_cache.GetText();

        // Edge-case filter: skip detection for URLs, paths, mostly non-alpha
        bool shouldSkip = false;
        if (text.find(L"://") != std::wstring::npos ||
            text.find(L"www.") != std::wstring::npos ||
            text.find(L"http") != std::wstring::npos) {
            shouldSkip = true;
        }
        if (text.size() > 2 && text[1] == L':' && (text[2] == L'\\' || text[2] == L'/')) {
            shouldSkip = true;
        }
        {
            size_t alphaCount = 0;
            for (wchar_t c : text) {
                if (iswalpha(c)) ++alphaCount;
            }
            if (alphaCount < text.size() / 2) shouldSkip = true;
        }

        if (!shouldSkip) {
            // Generate all 6 layout conversion variants (same order as Python)
            std::vector<std::wstring> textVariants = {
                ConvertTextBidirectional(text, Layouts::english_layout, Layouts::russian_layout),
                ConvertTextBidirectional(text, Layouts::russian_layout, Layouts::english_layout),
                ConvertTextBidirectional(text, Layouts::hebrew_layout,  Layouts::english_layout),
                ConvertTextBidirectional(text, Layouts::english_layout, Layouts::hebrew_layout),
                ConvertTextBidirectional(text, Layouts::russian_layout, Layouts::hebrew_layout),
                ConvertTextBidirectional(text, Layouts::hebrew_layout,  Layouts::russian_layout),
            };

            // Deduplicate (preserving first occurrence order)
            {
                std::vector<std::wstring> unique;
                std::set<std::wstring> seen;
                for (auto& v : textVariants) {
                    if (seen.insert(v).second)
                        unique.push_back(std::move(v));
                }
                textVariants = std::move(unique);
            }

            // Typo-resilient detection: consecutive agreement + drop-one boosting
            // Uses per-language-pair parameters for confidence thresholds.
            auto detection = TypoResilientDetect(
                *g_detector, textVariants, currentLangId, cacheSize, g_history);

            if (detection.has_value()) {
                const std::string& bestLang = detection->language;
                bool didCorrection = false;

                Dbg("DETECTION: currentLang=%s bestLang=%s conf=%.3f cacheText=%s",
                    currentLangId.c_str(), bestLang.c_str(), detection->confidence,
                    ws2s(text).c_str());

                if (currentLangId != bestLang) {
                    size_t cacheLen = g_cache.Size();
                    auto cacheChars = g_cache.GetCache();

                    // Convert cached text to the detected language
                    std::wstring cachedText(cacheChars.begin(), cacheChars.end());
                    std::wstring correctedText = ConvertTextBidirectional(
                        cachedText,
                        Layouts::GetLayoutForLanguage(currentLangId),
                        Layouts::GetLayoutForLanguage(bestLang)
                    );

                    Dbg("CORRECT: cacheLen=%zu cached=%s corrected=%s",
                        cacheLen, ws2s(cachedText).c_str(), ws2s(correctedText).c_str());

                    // Preserve first-letter capitalization
                    if (g_cache.WasFirstCharShifted() && !correctedText.empty()) {
                        wchar_t buf[2] = { correctedText[0], L'\0' };
                        CharUpperW(buf);
                        correctedText[0] = buf[0];
                    }

                    g_isSendingInput.store(true);
                    BOOL blocked = BlockInput(TRUE);

                    Dbg("SENDING: backspaces=%zu text=%s",
                        cacheLen > 1 ? cacheLen - 1 : 0, ws2s(correctedText).c_str());

                    if (cacheLen > 1)
                        SendBackspaces(cacheLen - 1);

                    ChangeKeyboardLayout(bestLang);

                    std::vector<wchar_t> correctedChars(correctedText.begin(), correctedText.end());
                    SendString(correctedChars);

                    if (blocked) BlockInput(FALSE);
                    g_isSendingInput.store(false);
                    didCorrection = true;

                    // Save correction info for Esc-to-undo
                    g_lastCorrection.originalText  = cachedText;
                    g_lastCorrection.correctedText = correctedText;
                    g_lastCorrection.originalLang  = currentLangId;
                    g_lastCorrection.correctedLang = bestLang;
                    g_lastCorrection.hwnd          = g_lastForegroundHwnd;
                    g_lastCorrection.ctxHash       = g_lastContextHash;
                    g_lastCorrection.postChars.clear();
                    g_lastCorrection.valid         = true;

                    UpdateTrayTooltip();
                }

                // Confirm the detected language for this context —
                // whether or not a correction was needed.  This marks
                // the context as "decided" so future clicks in the same
                // window/tab/field don't restart detection.
                if (g_windows && g_lastForegroundHwnd) {
                    // Save under stable context hash
                    g_windows->SetLanguage(g_lastForegroundHwnd,
                                           g_lastContextHash, bestLang);

                    // Also save under current live hash if different
                    // (handles Notepad title drift during correction)
                    size_t liveCtxHash = GetWindowContextHash(g_lastForegroundHwnd);
                    if (liveCtxHash != g_lastContextHash) {
                        g_windows->SetLanguage(g_lastForegroundHwnd,
                                               liveCtxHash, bestLang);
                        // Update tracking so the next keystroke doesn't
                        // see a phantom context change from title drift
                        g_lastContextHash = liveCtxHash;
                    }

                    Dbg("SAVE-LANG: lang=%s hwnd=%p ctxHash=%zu liveCtxHash=%zu",
                        bestLang.c_str(), g_lastForegroundHwnd,
                        g_lastContextHash, liveCtxHash);
                }
                Config::LastSetting = bestLang;

                g_cache.Clear();
                g_history.Clear();
                Config::SEARCH.store(false);

                if (didCorrection) {
                    return 1; // Block the current keystroke
                }
            }
        }
    }

    return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
}

// ============================================================
// Low-level mouse hook procedure
// ============================================================
static LRESULT CALLBACK LowLevelMouseProc(int nCode, WPARAM wParam, LPARAM lParam) {
    if (nCode != HC_ACTION)
        return CallNextHookEx(g_mouseHook, nCode, wParam, lParam);
    if (!Config::EnableSwitcher.load())
        return CallNextHookEx(g_mouseHook, nCode, wParam, lParam);

    if (wParam == WM_LBUTTONDOWN || wParam == WM_RBUTTONDOWN || wParam == WM_MBUTTONDOWN) {
        // Click anywhere = new typing session.
        // HandleFocusChange checks for window/tab change, restores
        // saved language if needed, clears cache, and enables SEARCH.
        // forceSearch=true because user clicked — allow fresh detection.
        HandleFocusChange(true);
    }

    return CallNextHookEx(g_mouseHook, nCode, wParam, lParam);
}

// ============================================================
// Confidence Flyout Panel (appears near the tray icon)
// ============================================================
static const wchar_t CONF_FLYOUT_CLASS[] = L"KS_ConfFlyout";
static HWND g_hConfFlyout = nullptr;

// Compile-time defaults for Reset
static constexpr float ORIG_CONF_AT_MIN_CHARS = 0.97f;
static constexpr float ORIG_CONF_AT_MAX_CHARS = 0.55f;

struct ConfFlyoutControls {
    HWND hSliderMin = nullptr;
    HWND hSliderMax = nullptr;
    HWND hLabelMin  = nullptr;
    HWND hLabelMax  = nullptr;
};
static ConfFlyoutControls g_fly;

static void UpdateSliderLabel(HWND hLabel, int sliderValue) {
    wchar_t buf[16];
    swprintf_s(buf, L"%.2f", sliderValue / 100.0f);
    SetWindowTextW(hLabel, buf);
}

static void ApplyConfidenceFromSliders() {
    float minC = SendMessage(g_fly.hSliderMin, TBM_GETPOS, 0, 0) / 100.0f;
    float maxC = SendMessage(g_fly.hSliderMax, TBM_GETPOS, 0, 0) / 100.0f;
    Config::DefaultParams.ConfidenceAtMinChars = minC;
    Config::DefaultParams.ConfidenceAtMaxChars = maxC;
    for (auto& [pair, params] : Config::PairOverrides) {
        params.ConfidenceAtMinChars = minC;
        params.ConfidenceAtMaxChars = maxC;
    }
}

// Ensure short-text confidence ≥ long-text confidence at all times.
static void EnforceSliderConstraint(HWND changedSlider) {
    int minVal = static_cast<int>(SendMessage(g_fly.hSliderMin, TBM_GETPOS, 0, 0));
    int maxVal = static_cast<int>(SendMessage(g_fly.hSliderMax, TBM_GETPOS, 0, 0));

    if (changedSlider == g_fly.hSliderMin && minVal < maxVal) {
        SendMessage(g_fly.hSliderMax, TBM_SETPOS, TRUE, minVal);
        UpdateSliderLabel(g_fly.hLabelMax, minVal);
    } else if (changedSlider == g_fly.hSliderMax && maxVal > minVal) {
        SendMessage(g_fly.hSliderMin, TBM_SETPOS, TRUE, maxVal);
        UpdateSliderLabel(g_fly.hLabelMin, maxVal);
    }
}

static LRESULT CALLBACK ConfFlyoutWndProc(HWND hwnd, UINT msg,
                                           WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_CREATE: {
        HINSTANCE hInst = GetModuleHandle(nullptr);
        HFONT hFont = static_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
        const int pad = 12;
        const int cw  = 296;   // usable content width
        int y = 10;

        // ── Row 1: Confidence at short text ──
        HWND hS1 = CreateWindowExW(0, L"STATIC",
            L"Confidence (short text):",
            WS_CHILD | WS_VISIBLE, pad, y, 210, 16,
            hwnd, nullptr, hInst, nullptr);
        SendMessage(hS1, WM_SETFONT, (WPARAM)hFont, TRUE);

        g_fly.hLabelMin = CreateWindowExW(0, L"STATIC", L"",
            WS_CHILD | WS_VISIBLE | SS_RIGHT,
            pad + cw - 42, y, 42, 16,
            hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_LABEL_MIN_CONF)),
            hInst, nullptr);
        SendMessage(g_fly.hLabelMin, WM_SETFONT, (WPARAM)hFont, TRUE);

        y += 18;
        g_fly.hSliderMin = CreateWindowExW(0, TRACKBAR_CLASSW, nullptr,
            WS_CHILD | WS_VISIBLE | TBS_HORZ | TBS_NOTICKS,
            pad, y, cw, 25,
            hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_SLIDER_MIN_CONF)),
            hInst, nullptr);
        SendMessage(g_fly.hSliderMin, TBM_SETRANGE, TRUE, MAKELPARAM(50, 100));
        int initMin = static_cast<int>(Config::ConfidenceAtMinChars * 100 + 0.5f);
        SendMessage(g_fly.hSliderMin, TBM_SETPOS, TRUE, initMin);
        UpdateSliderLabel(g_fly.hLabelMin, initMin);

        // ── Row 2: Confidence at long text ──
        y += 32;
        HWND hS2 = CreateWindowExW(0, L"STATIC",
            L"Confidence (long text):",
            WS_CHILD | WS_VISIBLE, pad, y, 210, 16,
            hwnd, nullptr, hInst, nullptr);
        SendMessage(hS2, WM_SETFONT, (WPARAM)hFont, TRUE);

        g_fly.hLabelMax = CreateWindowExW(0, L"STATIC", L"",
            WS_CHILD | WS_VISIBLE | SS_RIGHT,
            pad + cw - 42, y, 42, 16,
            hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_LABEL_MAX_CONF)),
            hInst, nullptr);
        SendMessage(g_fly.hLabelMax, WM_SETFONT, (WPARAM)hFont, TRUE);

        y += 18;
        g_fly.hSliderMax = CreateWindowExW(0, TRACKBAR_CLASSW, nullptr,
            WS_CHILD | WS_VISIBLE | TBS_HORZ | TBS_NOTICKS,
            pad, y, cw, 25,
            hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_SLIDER_MAX_CONF)),
            hInst, nullptr);
        SendMessage(g_fly.hSliderMax, TBM_SETRANGE, TRUE, MAKELPARAM(50, 100));
        int initMax = static_cast<int>(Config::ConfidenceAtMaxChars * 100 + 0.5f);
        SendMessage(g_fly.hSliderMax, TBM_SETPOS, TRUE, initMax);
        UpdateSliderLabel(g_fly.hLabelMax, initMax);

        // ── Reset button ──
        y += 34;
        HWND hReset = CreateWindowExW(0, L"BUTTON", L"Reset Defaults",
            WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
            (320 - 110) / 2, y, 110, 26,
            hwnd, reinterpret_cast<HMENU>(static_cast<INT_PTR>(IDC_BTN_RESET)),
            hInst, nullptr);
        SendMessage(hReset, WM_SETFONT, (WPARAM)hFont, TRUE);

        return 0;
    }

    case WM_HSCROLL: {
        HWND hCtrl = reinterpret_cast<HWND>(lParam);
        if (hCtrl == g_fly.hSliderMin) {
            UpdateSliderLabel(g_fly.hLabelMin,
                static_cast<int>(SendMessage(g_fly.hSliderMin, TBM_GETPOS, 0, 0)));
            EnforceSliderConstraint(g_fly.hSliderMin);
        } else if (hCtrl == g_fly.hSliderMax) {
            UpdateSliderLabel(g_fly.hLabelMax,
                static_cast<int>(SendMessage(g_fly.hSliderMax, TBM_GETPOS, 0, 0)));
            EnforceSliderConstraint(g_fly.hSliderMax);
        }
        ApplyConfidenceFromSliders();
        return 0;
    }

    case WM_COMMAND:
        if (LOWORD(wParam) == IDC_BTN_RESET) {
            int defMin = static_cast<int>(ORIG_CONF_AT_MIN_CHARS * 100 + 0.5f);
            int defMax = static_cast<int>(ORIG_CONF_AT_MAX_CHARS * 100 + 0.5f);
            SendMessage(g_fly.hSliderMin, TBM_SETPOS, TRUE, defMin);
            SendMessage(g_fly.hSliderMax, TBM_SETPOS, TRUE, defMax);
            UpdateSliderLabel(g_fly.hLabelMin, defMin);
            UpdateSliderLabel(g_fly.hLabelMax, defMax);
            ApplyConfidenceFromSliders();
        }
        return 0;

    case WM_ACTIVATE:
        // Auto-dismiss when the user clicks away
        if (LOWORD(wParam) == WA_INACTIVE)
            DestroyWindow(hwnd);
        return 0;

    case WM_DESTROY:
        g_hConfFlyout = nullptr;
        g_fly = {};
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

static void ShowConfidenceFlyout() {
    // Toggle: if already open, close it
    if (g_hConfFlyout) {
        DestroyWindow(g_hConfFlyout);
        return;
    }

    HINSTANCE hInst = GetModuleHandle(nullptr);

    static bool classRegistered = false;
    if (!classRegistered) {
        WNDCLASSEXW wc = {};
        wc.cbSize        = sizeof(wc);
        wc.lpfnWndProc   = ConfFlyoutWndProc;
        wc.hInstance      = hInst;
        wc.hbrBackground  = reinterpret_cast<HBRUSH>(COLOR_BTNFACE + 1);
        wc.lpszClassName  = CONF_FLYOUT_CLASS;
        wc.hCursor        = LoadCursor(nullptr, IDC_ARROW);
        RegisterClassExW(&wc);
        classRegistered = true;
    }

    const int w = 320, h = 170;

    // Position the flyout just above the cursor (tray icon area)
    POINT pt;
    GetCursorPos(&pt);
    int x = pt.x - w / 2;
    int y = pt.y - h - 4;

    // Clamp to work area so it never goes off-screen
    RECT wa;
    SystemParametersInfo(SPI_GETWORKAREA, 0, &wa, 0);
    if (x < wa.left)         x = wa.left;
    if (x + w > wa.right)    x = wa.right - w;
    if (y < wa.top)          y = wa.top;
    if (y + h > wa.bottom)   y = wa.bottom - h;

    g_hConfFlyout = CreateWindowExW(
        WS_EX_TOOLWINDOW | WS_EX_TOPMOST,
        CONF_FLYOUT_CLASS, nullptr,
        WS_POPUP | WS_BORDER,
        x, y, w, h,
        g_hwndHidden, nullptr, hInst, nullptr);

    ShowWindow(g_hConfFlyout, SW_SHOW);
    SetForegroundWindow(g_hConfFlyout);
}

// ============================================================
// Hidden window procedure (tray icon messages + context menu)
// ============================================================
static LRESULT CALLBACK HiddenWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_TRAYICON:
        // Left-click: toggle confidence flyout
        if (lParam == WM_LBUTTONUP) {
            ShowConfidenceFlyout();
        }
        // Right-click: context menu
        else if (lParam == WM_RBUTTONUP || lParam == WM_CONTEXTMENU) {
            POINT pt;
            GetCursorPos(&pt);
            SetForegroundWindow(hwnd);

            HMENU hMenu = CreatePopupMenu();
            if (hMenu) {
                AppendMenuW(hMenu,
                    MF_STRING | (Config::EnableSwitcher.load() ? MF_CHECKED : MF_UNCHECKED),
                    ID_TRAY_ENABLE_SWITCHER, L"Enable Switcher");
                AppendMenuW(hMenu,
                    MF_STRING
                    | (Config::SaveWindowState.load() ? MF_CHECKED : MF_UNCHECKED)
                    | (Config::EnableSwitcher.load() ? 0 : MF_GRAYED),
                    ID_TRAY_SAVE_WINDOW, L"Save window state");
                AppendMenuW(hMenu,
                    MF_STRING
                    | (Config::EnableTypoResilience ? MF_CHECKED : MF_UNCHECKED)
                    | (Config::EnableSwitcher.load() ? 0 : MF_GRAYED),
                    ID_TRAY_TYPO_RESILIENCE, L"Typo Resilience");
                AppendMenuW(hMenu,
                    MF_STRING | (g_debugLogEnabled ? MF_CHECKED : MF_UNCHECKED),
                    ID_TRAY_DEBUG_LOG, L"Debug Log");

                AppendMenuW(hMenu, MF_SEPARATOR, 0, nullptr);
                AppendMenuW(hMenu, MF_STRING, ID_TRAY_CHECK_UPDATE, L"Check for Updates\x2026");
                AppendMenuW(hMenu, MF_STRING, ID_TRAY_ABOUT, L"About");
                AppendMenuW(hMenu, MF_SEPARATOR, 0, nullptr);
                AppendMenuW(hMenu, MF_STRING, ID_TRAY_EXIT, L"Exit");

                TrackPopupMenu(hMenu, TPM_BOTTOMALIGN | TPM_LEFTALIGN,
                               pt.x, pt.y, 0, hwnd, nullptr);
                DestroyMenu(hMenu);
            }
        }
        return 0;

    case WM_COMMAND:
        switch (LOWORD(wParam)) {
        case ID_TRAY_ENABLE_SWITCHER:
            Config::EnableSwitcher.store(!Config::EnableSwitcher.load());
            break;
        case ID_TRAY_SAVE_WINDOW:
            Config::SaveWindowState.store(!Config::SaveWindowState.load());
            break;
        case ID_TRAY_TYPO_RESILIENCE:
            Config::EnableTypoResilience = !Config::EnableTypoResilience;
            break;
        case ID_TRAY_DEBUG_LOG:
            DbgSetEnabled(!g_debugLogEnabled);
            if (g_debugLogEnabled) {
                Dbg("=== Debug logging enabled (v%s) ===",
                    std::string(Config::VERSION,
                                Config::VERSION + wcslen(Config::VERSION)).c_str());
            }
            break;
        case ID_TRAY_ABOUT:
            MessageBoxW(nullptr,
                L"Auto-detects En/He/Ru and corrects layout.\n\n"
                L"Esc \x2014 undo last correction\n"
                L"Left-click tray \x2014 confidence sliders\n"
                L"Right-click tray \x2014 settings\n\n"
                L"\x00A9 2025-2026 Alpha-Numerical",
                (std::wstring(L"Keyboard Switcher v") + Config::VERSION).c_str(),
                MB_OK | MB_ICONINFORMATION);
            break;
        case ID_TRAY_CHECK_UPDATE:
            CheckForUpdatesAsync(true);
            break;
        case ID_TRAY_EXIT:
            g_trayIcon.Remove();
            PostQuitMessage(0);
            break;
        }
        return 0;

    case WM_DESTROY:
        g_trayIcon.Remove();
        PostQuitMessage(0);
        return 0;

    case WM_UPDATE_CHECK_RESULT:
        HandleUpdateCheckResult();
        return 0;

    case WM_UPDATE_DOWNLOAD_DONE:
        HandleUpdateDownloadDone();
        return 0;

    case WM_TIMER:
        // Periodic cleanup of closed windows from the tracker
        if (g_windows) {
            g_windows->Cleanup();
        }
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

// ============================================================
// WinMain - application entry point
// ============================================================
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int) {
    // ---- Initialize debug log ----
    DbgInit();
    Dbg("=== KeyboardSwitcher started ===");

    // ---- Common controls (trackbar for confidence sliders) ----
    INITCOMMONCONTROLSEX icex = { sizeof(icex), ICC_BAR_CLASSES };
    InitCommonControlsEx(&icex);

    // ---- Single-instance guard (named mutex) ----
    HANDLE hMutex = CreateMutexW(nullptr, TRUE, L"Global\\KeyboardSwitcher_SingleInstance");
    if (hMutex == nullptr || GetLastError() == ERROR_ALREADY_EXISTS) {
        // Another instance is already running — exit silently
        if (hMutex) CloseHandle(hMutex);
        return 0;
    }

    // Get executable directory for locating data files
    std::wstring exeDir = GetExeDirectory();
    std::wstring modelPath = exeDir + L"\\lang_model.onnx";
    std::wstring dictPath  = exeDir + L"\\dictionary.json";

    // Load the ONNX model
    auto detector = std::make_unique<LanguageDetector>();
    if (!detector->Load(modelPath, dictPath)) {
        MessageBoxW(nullptr,
            L"Failed to load language model or dictionary.\n"
            L"Make sure lang_model.onnx and dictionary.json are next to the exe.",
            L"Keyboard Switcher - Error", MB_ICONERROR);
        return 1;
    }
    g_detector = detector.get();

    // Initialize the window tracker (no default language — each window
    // is unvisited until the user first interacts with it)
    auto windowTracker = std::make_unique<WindowTracker>();
    g_windows = windowTracker.get();
    std::string initialLang = GetCurrentKeyboardLayout();
    Config::LastSetting = initialLang;

    // Seed focus-tracking globals so the first keystroke doesn't
    // trigger a spurious HandleFocusChange.
    g_lastForegroundHwnd = GetForegroundWindow();
    g_lastContextHash = g_lastForegroundHwnd
                        ? GetWindowContextHash(g_lastForegroundHwnd)
                        : 0;

    // Register hidden window class
    const wchar_t CLASS_NAME[] = L"KeyboardSwitcherHiddenWindow";
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = HiddenWndProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    RegisterClassExW(&wc);

    // Create hidden message-only window for tray messages
    g_hwndHidden = CreateWindowExW(0, CLASS_NAME, L"Keyboard Switcher",
                                   0, 0, 0, 0, 0,
                                   HWND_MESSAGE, nullptr, hInstance, nullptr);
    if (!g_hwndHidden) {
        MessageBoxW(nullptr, L"Failed to create message window.", L"Error", MB_ICONERROR);
        return 1;
    }

    // Create system tray icon
    HICON hIcon = LoadIconW(hInstance, MAKEINTRESOURCEW(IDI_KEYBOARD_ICON));
    if (!hIcon) {
        hIcon = LoadIconW(nullptr, IDI_APPLICATION);
    }
    g_trayIcon.Create(g_hwndHidden, hIcon);
    UpdateTrayTooltip();

    // Periodic cleanup of stale window entries (every 60 seconds)
    SetTimer(g_hwndHidden, 1, 60000, nullptr);

    // Install low-level keyboard and mouse hooks
    g_keyboardHook = SetWindowsHookExW(WH_KEYBOARD_LL, LowLevelKeyboardProc, hInstance, 0);
    g_mouseHook    = SetWindowsHookExW(WH_MOUSE_LL, LowLevelMouseProc, hInstance, 0);

    if (!g_keyboardHook || !g_mouseHook) {
        MessageBoxW(nullptr, L"Failed to install keyboard/mouse hooks.",
                    L"Error", MB_ICONERROR);
        g_trayIcon.Remove();
        return 1;
    }

    // Install WinEvent hook for foreground-window changes.
    // Covers Alt+Tab, Win+Tab, taskbar clicks, Alt+F4 exposing the
    // next window, virtual-desktop switches, etc.
    // WINEVENT_OUTOFCONTEXT → callback is delivered via the message pump
    //                         on this thread — no cross-thread issues.
    // WINEVENT_SKIPOWNPROCESS → ignore focus changes to our own windows
    //                           (tray menu, confidence flyout).
    g_winEventHook = SetWinEventHook(
        EVENT_SYSTEM_FOREGROUND, EVENT_SYSTEM_FOREGROUND,
        nullptr, WinEventProc, 0, 0,
        WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS);

    // Check for updates in the background (3-second delay so the app
    // starts up smoothly; silent if already on the latest version).
    CheckForUpdatesAsync(false, 3);

    // Message loop (required for low-level hooks to work)
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    // Cleanup
    if (g_winEventHook) UnhookWinEvent(g_winEventHook);
    if (g_keyboardHook) UnhookWindowsHookEx(g_keyboardHook);
    if (g_mouseHook)    UnhookWindowsHookEx(g_mouseHook);
    g_detector = nullptr;
    g_windows  = nullptr;

    // Release single-instance mutex
    if (hMutex) {
        ReleaseMutex(hMutex);
        CloseHandle(hMutex);
    }

    return static_cast<int>(msg.wParam);
}

