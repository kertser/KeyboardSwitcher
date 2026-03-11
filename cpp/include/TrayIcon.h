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

private:
    NOTIFYICONDATAW nid_ = {};
    bool created_ = false;
};

