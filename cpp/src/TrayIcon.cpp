#include "TrayIcon.h"
#include <strsafe.h>

TrayIcon::TrayIcon() = default;

TrayIcon::~TrayIcon() {
    Remove();
}

bool TrayIcon::Create(HWND hwnd, HICON hIcon) {
    ZeroMemory(&nid_, sizeof(nid_));
    nid_.cbSize = sizeof(NOTIFYICONDATAW);
    nid_.hWnd = hwnd;
    nid_.uID = 1;
    nid_.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    nid_.uCallbackMessage = WM_APP + 1;
    nid_.hIcon = hIcon;
    StringCchCopyW(nid_.szTip, ARRAYSIZE(nid_.szTip), L"Keyboard Switcher");

    created_ = Shell_NotifyIconW(NIM_ADD, &nid_) != FALSE;
    return created_;
}

void TrayIcon::Remove() {
    if (created_) {
        Shell_NotifyIconW(NIM_DELETE, &nid_);
        created_ = false;
    }
}

void TrayIcon::ShowBalloon(const std::wstring& title, const std::wstring& message) {
    nid_.uFlags = NIF_INFO;
    nid_.dwInfoFlags = NIIF_INFO;
    StringCchCopyW(nid_.szInfoTitle, ARRAYSIZE(nid_.szInfoTitle), title.c_str());
    StringCchCopyW(nid_.szInfo, ARRAYSIZE(nid_.szInfo), message.c_str());
    Shell_NotifyIconW(NIM_MODIFY, &nid_);
}

void TrayIcon::UpdateTooltip(const std::wstring& text) {
    if (!created_) return;
    nid_.uFlags = NIF_TIP;
    StringCchCopyW(nid_.szTip, ARRAYSIZE(nid_.szTip), text.c_str());
    Shell_NotifyIconW(NIM_MODIFY, &nid_);
}

void TrayIcon::UpdateIcon(HICON hIcon) {
    if (!created_ || !hIcon) return;
    nid_.uFlags = NIF_ICON;
    nid_.hIcon = hIcon;
    Shell_NotifyIconW(NIM_MODIFY, &nid_);
}

