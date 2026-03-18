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

#include "Config.h"
#include "Languages.h"
#include "InputCache.h"
#include "WindowTracker.h"
#include "TrayIcon.h"
#include "../resources/resource.h"

// ============================================================
// Globals
// ============================================================
static HHOOK g_keyboardHook = nullptr;
static HHOOK g_mouseHook = nullptr;
static HWND  g_hwndHidden = nullptr;

static InputCache          g_cache;
static DetectionHistory    g_history;
static WindowTracker*      g_windows = nullptr;
static LanguageDetector*   g_detector = nullptr;
static TrayIcon            g_trayIcon;
static std::atomic<bool>   g_isSendingInput{false}; // re-entrancy guard

// Focus tracking — updated by HandleFocusChange(), read by hooks.
// All accesses are on the main (message-loop) thread — no mutex needed.
static HWND          g_lastForegroundHwnd = nullptr;
static size_t        g_lastTitleHash      = 0;
static HWINEVENTHOOK g_winEventHook       = nullptr;

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
static void HandleFocusChange() {
    if (g_isSendingInput.load()) return;

    HWND hwnd = GetForegroundWindow();
    if (!hwnd) return;

    // Skip focus changes to our own windows (tray menu, flyout panel)
    DWORD pid = 0;
    GetWindowThreadProcessId(hwnd, &pid);
    if (pid == GetCurrentProcessId()) return;

    size_t titleHash = GetWindowTitleHash(hwnd);
    bool contextChanged = (hwnd != g_lastForegroundHwnd) ||
                          (titleHash != g_lastTitleHash);

    if (contextChanged) {
        g_lastForegroundHwnd = hwnd;
        g_lastTitleHash      = titleHash;

        std::string currentLayout = GetCurrentKeyboardLayout();

        if (g_windows && Config::SaveWindowState.load()) {
            std::string savedLang = g_windows->GetLanguage(hwnd, titleHash);
            if (!savedLang.empty()) {
                // Known window/tab — restore its language
                if (savedLang != currentLayout) {
                    ChangeKeyboardLayout(savedLang);
                }
                Config::LastSetting = savedLang;
            } else {
                // New window/tab — record current layout
                g_windows->SetLanguage(hwnd, titleHash, currentLayout);
                Config::LastSetting = currentLayout;
            }
        } else {
            Config::LastSetting = currentLayout;
        }

        UpdateTrayTooltip();
    }

    // Always clear cache and enable detection
    // (any focus change or click = new typing session)
    g_cache.Clear();
    g_history.Clear();
    Config::SEARCH.store(true);
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
    // Batch all backspace key-down/key-up pairs into a single SendInput call
    // to minimise the time window where real keystrokes could slip through.
    std::vector<INPUT> inputs(count * 2);
    for (size_t i = 0; i < count; ++i) {
        inputs[i * 2].type = INPUT_KEYBOARD;
        inputs[i * 2].ki.wVk = VK_BACK;
        inputs[i * 2].ki.dwFlags = 0;
        inputs[i * 2 + 1].type = INPUT_KEYBOARD;
        inputs[i * 2 + 1].ki.wVk = VK_BACK;
        inputs[i * 2 + 1].ki.dwFlags = KEYEVENTF_KEYUP;
    }
    SendInput(static_cast<UINT>(inputs.size()), inputs.data(), sizeof(INPUT));
}

// ============================================================
// Helper: send a string via SendInput (Unicode chars)
// ============================================================
static void SendString(const std::vector<wchar_t>& chars) {
    // Batch all Unicode key-down/key-up pairs into a single SendInput call
    // to minimise the time window where real keystrokes could collide.
    std::vector<INPUT> inputs(chars.size() * 2);
    for (size_t i = 0; i < chars.size(); ++i) {
        inputs[i * 2].type = INPUT_KEYBOARD;
        inputs[i * 2].ki.wScan = chars[i];
        inputs[i * 2].ki.dwFlags = KEYEVENTF_UNICODE;
        inputs[i * 2 + 1].type = INPUT_KEYBOARD;
        inputs[i * 2 + 1].ki.wScan = chars[i];
        inputs[i * 2 + 1].ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP;
    }
    SendInput(static_cast<UINT>(inputs.size()), inputs.data(), sizeof(INPUT));
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

    // --- Focus-change detection (belt-and-suspenders) ---
    // EVENT_SYSTEM_FOREGROUND handles most focus changes, but we
    // also check here to catch edge cases where the WinEvent
    // callback hasn't been dispatched yet, and to detect in-window
    // tab switches (title changes, e.g. Ctrl+Tab in a browser).
    //
    // Title-hash is only checked when the cache is empty (start of
    // a new typing session).  This avoids spurious resets when apps
    // like Notepad++ update the title mid-typing (e.g. prepending
    // "*" for unsaved changes).
    {
        HWND currentHwnd = GetForegroundWindow();
        if (currentHwnd) {
            bool hwndChanged  = (currentHwnd != g_lastForegroundHwnd);
            bool titleChanged = false;
            if (!hwndChanged && g_cache.Size() == 0) {
                // Same HWND, no chars yet — check for tab switch.
                // GetWindowTextW on another process's window reads the
                // OS-cached caption — no cross-process SendMessage.
                size_t titleHash = GetWindowTitleHash(currentHwnd);
                titleChanged = (titleHash != g_lastTitleHash);
            }
            if (hwndChanged || titleChanged) {
                HandleFocusChange();
            }
        }
    }

    // --- Track manual layout changes ---
    // If the user manually switched the layout (e.g. Alt+Shift),
    // disable detection until the next click / focus change —
    // the user already chose a language.
    {
        std::string currentLayout = GetCurrentKeyboardLayout();
        if (g_windows && g_lastForegroundHwnd) {
            std::string windowLang =
                g_windows->GetLanguage(g_lastForegroundHwnd, g_lastTitleHash);
            if (windowLang.empty()) {
                // First visit to this window/tab — record the current layout
                g_windows->SetLanguage(g_lastForegroundHwnd, g_lastTitleHash,
                                       currentLayout);
                Config::LastSetting = currentLayout;
            } else if (windowLang != currentLayout) {
                // Layout changed since we last recorded it → manual switch
                if (Config::SaveWindowState.load()) {
                    g_windows->SetLanguage(g_lastForegroundHwnd, g_lastTitleHash,
                                           currentLayout);
                }
                Config::LastSetting = currentLayout;
                g_cache.Clear();
                g_history.Clear();
                Config::SEARCH.store(false);
                UpdateTrayTooltip();
            }
        }
    }

    // --- Handle character input ---
    bool shiftHeld = (GetAsyncKeyState(VK_SHIFT) & 0x8000) != 0;
    bool capsOn = (GetKeyState(VK_CAPITAL) & 0x0001) != 0;
    bool isUpperIntent = shiftHeld ^ capsOn; // XOR: Shift flips CapsLock

    if (vk == VK_BACK) {
        g_cache.DelChar();
    } else if (vk == VK_SPACE) {
        g_cache.PushChar(L' ', false);
    } else if (vk == VK_RETURN) {
        // Enter: placeholder (same as Python version)
    } else if ((vk >= 0x30 && vk <= 0x5A) || (vk >= VK_OEM_1 && vk <= VK_OEM_3) ||
               (vk >= VK_OEM_4 && vk <= VK_OEM_8)) {
        wchar_t ch = VkToWchar(vk);
        if (ch != 0) {
            g_cache.PushChar(ch, isUpperIntent);
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

                    // Preserve first-letter capitalization
                    if (g_cache.WasFirstCharShifted() && !correctedText.empty()) {
                        wchar_t buf[2] = { correctedText[0], L'\0' };
                        CharUpperW(buf);
                        correctedText[0] = buf[0];
                    }

                    g_isSendingInput.store(true);
                    // Block physical keyboard/mouse input for the duration
                    // of the correction.  BlockInput() only blocks hardware
                    // events — our SendInput() calls are still delivered.
                    BOOL blocked = BlockInput(TRUE);

                    // Current keystroke not delivered yet; erase N-1 on screen
                    if (cacheLen > 1)
                        SendBackspaces(cacheLen - 1);

                    // Change keyboard layout to the detected language
                    // (SendMessage is synchronous — layout is active on return)
                    ChangeKeyboardLayout(bestLang);
                    if (g_windows && g_lastForegroundHwnd) {
                        g_windows->SetLanguage(g_lastForegroundHwnd,
                                               g_lastTitleHash, bestLang);
                    }
                    Config::LastSetting = bestLang;

                    // Re-type the full corrected text (KEYEVENTF_UNICODE — layout-independent)
                    std::vector<wchar_t> correctedChars(correctedText.begin(), correctedText.end());
                    SendString(correctedChars);

                    if (blocked) BlockInput(FALSE);
                    g_isSendingInput.store(false);
                    didCorrection = true;
                    UpdateTrayTooltip();
                }

                // Clear cache and disable search until next mouse click
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
        HandleFocusChange();
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

                AppendMenuW(hMenu, MF_SEPARATOR, 0, nullptr);
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
        case ID_TRAY_ABOUT:
            g_trayIcon.ShowBalloon((std::wstring(L"Keyboard Switcher v") + Config::VERSION).c_str(),
                                   L"Click on text area and start typing");
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
    g_lastTitleHash = g_lastForegroundHwnd
                        ? GetWindowTitleHash(g_lastForegroundHwnd)
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

