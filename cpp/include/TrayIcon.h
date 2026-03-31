#pragma once

#include <windows.h>
#include <shellapi.h>
#include <string>
#include <functional>

class TrayIcon {
public:
    TrayIcon();
    ~TrayIcon();

    // Initialize the tray icon with the given HWND and icon
    bool Create(HWND hwnd, HICON hIcon);

    // Remove the tray icon
    void Remove();

    // Show a balloon notification
    void ShowBalloon(const std::wstring& title, const std::wstring& message);

    // Update the hover tooltip text
    void UpdateTooltip(const std::wstring& text);

    // Change the displayed icon (e.g. to reflect detection state)
    void UpdateIcon(HICON hIcon);

private:
    NOTIFYICONDATAW nid_ = {};
    bool created_ = false;
};

