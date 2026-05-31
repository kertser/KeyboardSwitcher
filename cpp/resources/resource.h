#pragma once

#define IDI_KEYBOARD_ICON 101
#define IDR_TRAY_MENU     102

#define ID_TRAY_ENABLE_SWITCHER  1001
#define ID_TRAY_SAVE_WINDOW      1002
#define ID_TRAY_ABOUT            1003
#define ID_TRAY_EXIT             1004
#define ID_TRAY_TYPO_RESILIENCE  1005
#define ID_TRAY_DEBUG_LOG        1006
#define ID_TRAY_CHECK_UPDATE     1007
#define ID_TRAY_COLLECT_FEEDBACK 1008
#define ID_TRAY_OPEN_FEEDBACK    1009
#define ID_TRAY_RESET_LEARNING   1010

// Detection-settings flyout control IDs
#define IDC_SLIDER_MIN_CONF   2001   // global short-text confidence slider
#define IDC_LABEL_MIN_CONF    2003   // global short-text confidence label
#define IDC_BTN_RESET         2005   // reset-defaults button

// Per-direction controls (6 language pairs, indices 0-5)
#define IDC_EDIT_PAIR_BASE    2100   // 2100-2105 : min-chars edit boxes
#define IDC_SPIN_PAIR_BASE    2110   // 2110-2115 : spin buddies
#define IDC_SLIDER_PAIR_BASE  2120   // 2120-2125 : confidence-floor sliders
#define IDC_LABEL_PAIR_BASE   2130   // 2130-2135 : confidence-floor labels

#define WM_TRAYICON (WM_APP + 1)
