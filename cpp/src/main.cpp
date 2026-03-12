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
#include <strsafe.h>

#include <string>
#include <vector>
#include <set>
#include <algorithm>
#include <thread>
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
static WindowTracker*      g_windows = nullptr;
static LanguageDetector*   g_detector = nullptr;
static TrayIcon            g_trayIcon;
static std::atomic<bool>   g_isSendingInput{false}; // re-entrancy guard

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
// Helper: send N backspaces via SendInput
// ============================================================
static void SendBackspaces(size_t count) {
    for (size_t i = 0; i < count; ++i) {
        INPUT inputs[2] = {};
        inputs[0].type = INPUT_KEYBOARD;
        inputs[0].ki.wVk = VK_BACK;
        inputs[0].ki.dwFlags = 0;
        inputs[1].type = INPUT_KEYBOARD;
        inputs[1].ki.wVk = VK_BACK;
        inputs[1].ki.dwFlags = KEYEVENTF_KEYUP;
        SendInput(2, inputs, sizeof(INPUT));
        Sleep(5);
    }
}

// ============================================================
// Helper: send a string via SendInput (Unicode chars)
// ============================================================
static void SendString(const std::vector<wchar_t>& chars) {
    for (wchar_t ch : chars) {
        INPUT inputs[2] = {};
        inputs[0].type = INPUT_KEYBOARD;
        inputs[0].ki.wScan = ch;
        inputs[0].ki.dwFlags = KEYEVENTF_UNICODE;
        inputs[1].type = INPUT_KEYBOARD;
        inputs[1].ki.wScan = ch;
        inputs[1].ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP;
        SendInput(2, inputs, sizeof(INPUT));
        Sleep(5);
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

    // Skip injected / synthetic keystrokes (from SendInput)
    if (g_isSendingInput.load() || (pKb->flags & LLKHF_INJECTED))
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);

    // Only process key-down events
    if (wParam != WM_KEYDOWN && wParam != WM_SYSKEYDOWN)
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);

    DWORD vk = pKb->vkCode;

    // --- Alt+Tab detection ---
    if (vk == VK_LMENU || vk == VK_RMENU) {
        Config::alt_pressed.store(true);
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
    }
    if (Config::alt_pressed.load() && vk == VK_TAB) {
        Config::alt_pressed.store(false);
        // After Alt+Tab, schedule a re-check via a short delay
        std::thread([]() {
            Sleep(500);
            std::string currentLayout = GetCurrentKeyboardLayout();
            if (g_windows) {
                std::string windowLang = g_windows->GetActiveWindowLanguage();
                if (Config::SaveWindowState.load()) {
                    if (windowLang != currentLayout && currentLayout != Config::LastSetting) {
                        g_windows->SetActiveWindowLanguage(currentLayout);
                        Config::LastSetting = currentLayout;
                    } else {
                        if (windowLang.empty()) windowLang = Config::LastSetting;
                        ChangeKeyboardLayout(windowLang);
                        Config::LastSetting = windowLang;
                    }
                }
            }
            g_cache.Clear();
            Config::SEARCH.store(true);
        }).detach();
        return CallNextHookEx(g_keyboardHook, nCode, wParam, lParam);
    }
    if (vk != VK_LMENU && vk != VK_RMENU) {
        Config::alt_pressed.store(false);
    }

    // --- Track manual layout changes ---
    // If the user manually switched the layout (e.g. Alt+Shift), disable
    // detection until the next mouse click – the user already chose a language.
    std::string currentLayout = GetCurrentKeyboardLayout();
    if (g_windows) {
        std::string windowLang = g_windows->GetActiveWindowLanguage();
        if (windowLang != currentLayout) {
            if (Config::SaveWindowState.load()) {
                g_windows->SetActiveWindowLanguage(currentLayout);
                Config::LastSetting = currentLayout;
            }
            // Manual switch detected → stop auto-detection until next click
            g_cache.Clear();
            Config::SEARCH.store(false);
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
        // Compute the confidence bar for the current text length.
        // Short text → very high bar; longer text → lower bar.
        float requiredConfidence = Config::GetRequiredConfidence(cacheSize);

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

            // Run prediction WITH CONFIDENCE on each variant.
            // Pick the variant whose predicted language has the highest
            // softmax probability – that is the most likely interpretation.
            std::string bestLang;
            float       bestConf = 0.0f;

            for (const auto& variant : textVariants) {
                auto result = g_detector->PredictLanguageWithConfidence(variant);
                if (result.has_value() && result->confidence > bestConf) {
                    bestConf = result->confidence;
                    bestLang = result->language;
                }
            }

            // Act only when confidence meets the adaptive threshold.
            // If the bar is not met we keep accumulating characters –
            // the next keystroke will lower the bar and try again.
            if (!bestLang.empty() && bestConf >= requiredConfidence) {
                std::string currentLangId = GetCurrentKeyboardLayout();
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

                    // Current keystroke not delivered yet; erase N-1 on screen
                    if (cacheLen > 1)
                        SendBackspaces(cacheLen - 1);

                    // Change keyboard layout to the detected language
                    ChangeKeyboardLayout(bestLang);
                    if (g_windows) {
                        g_windows->SetActiveWindowLanguage(bestLang);
                    }
                    Config::LastSetting = bestLang;

                    Sleep(50);

                    // Re-type the full corrected text
                    std::vector<wchar_t> correctedChars(correctedText.begin(), correctedText.end());
                    SendString(correctedChars);

                    g_isSendingInput.store(false);
                    didCorrection = true;
                }

                // Clear cache and disable search until next mouse click
                g_cache.Clear();
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
        g_cache.Clear();
        Config::SEARCH.store(true);

        std::string currentLayout = GetCurrentKeyboardLayout();
        if (g_windows && Config::SaveWindowState.load()) {
            std::string windowLang = g_windows->GetActiveWindowLanguage();
            if (windowLang != currentLayout && currentLayout != Config::LastSetting) {
                g_windows->SetActiveWindowLanguage(currentLayout);
                Config::LastSetting = currentLayout;
            } else {
                if (windowLang.empty()) windowLang = Config::LastSetting;
                ChangeKeyboardLayout(windowLang);
                Config::LastSetting = windowLang;
            }
        }
    }

    return CallNextHookEx(g_mouseHook, nCode, wParam, lParam);
}

// ============================================================
// Hidden window procedure (tray icon messages + context menu)
// ============================================================
static LRESULT CALLBACK HiddenWndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_TRAYICON:
        if (lParam == WM_RBUTTONUP || lParam == WM_CONTEXTMENU) {
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
        case ID_TRAY_ABOUT:
            g_trayIcon.ShowBalloon(L"Keyboard Switcher v1.1",
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

    // Initialize the window tracker with the current keyboard layout
    std::string initialLang = GetCurrentKeyboardLayout();
    auto windowTracker = std::make_unique<WindowTracker>(initialLang);
    g_windows = windowTracker.get();
    Config::LastSetting = initialLang;

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

    // Message loop (required for low-level hooks to work)
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    // Cleanup
    if (g_keyboardHook) UnhookWindowsHookEx(g_keyboardHook);
    if (g_mouseHook)    UnhookWindowsHookEx(g_mouseHook);
    g_detector = nullptr;
    g_windows  = nullptr;

    return static_cast<int>(msg.wParam);
}

