# KeyboardSwitcher – Python implementation (synced with C++ v1.2.3)
#
# Importing the necessary packages
import threading
import pystray
import Languages
from Languages import (convert_text_bidirectional, get_layout_for_language,
                       DetectionHistory, typo_resilient_detect)
from PIL import Image
from dataclasses import dataclass, field
import py_win_keyboard_layout
from pynput import keyboard, mouse
import ctypes
import sys
import config
import time
import warnings
import pygetwindow as gw

# Ignore all warnings:
warnings.filterwarnings("ignore")

# ============================================================
# Global tray icon reference (for tooltip updates)
# ============================================================
g_tray_icon = None


# ============================================================
# InputCache – tracks typed characters and shift state
# (mirrors C++ InputCache with shift tracking)
# ============================================================
@dataclass
class InputCache:
    cache: list = field(default_factory=list)
    shift_state: list = field(default_factory=list)

    def push_char(self, char: str, shift_held: bool = False):
        self.cache.append(char)
        self.shift_state.append(shift_held)

    def del_char(self):
        if len(self.cache) > 0:
            self.cache.pop()
            self.shift_state.pop()

    def clear(self):
        self.cache.clear()
        self.shift_state.clear()

    def __len__(self):
        return len(self.cache)

    def get_text(self) -> str:
        return ''.join(self.cache)

    def get_cache(self) -> list:
        return list(self.cache)

    def was_first_char_shifted(self) -> bool:
        """Returns True if Shift was held when the first character was typed."""
        if not self.shift_state:
            return False
        return self.shift_state[0]


# ============================================================
# OpenWindows – per-window language tracker
# (mirrors C++ WindowTracker)
# ============================================================
@dataclass
class OpenWindows:

    def __init__(self, language):
        self.windows_language = {}
        available_windows = gw.getAllTitles()
        for window in available_windows:
            self.windows_language[window] = config.LANGUAGE_ID[language]

    def get_active_window_langage(self):
        try:
            return self.windows_language[gw.getActiveWindowTitle()]
        except Exception:
            return None

    def set_active_window_langage(self, language):
        try:
            self.windows_language[gw.getActiveWindowTitle()] = language
        except Exception:
            pass

    def cleanup(self):
        """Remove entries for windows that no longer exist (like C++ Cleanup)."""
        available_windows = set(gw.getAllTitles())
        self.windows_language = {
            w: lang for w, lang in self.windows_language.items()
            if w in available_windows
        }

    # Keep backward-compatible alias
    def update_window_titles(self):
        self.cleanup()


# ============================================================
# Helper: get current keyboard layout as language string
# ============================================================
def get_keyboard_layout():
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    klid = ctypes.windll.user32.GetKeyboardLayout(
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, 0))
    lang_id = klid & 0xFFFF
    return config.LANGUAGE_ID.get(str(lang_id), 'en')


def get_keyboard_layout_info():
    return py_win_keyboard_layout.get_foreground_window_keyboard_layout() & 0xFFFF


# ============================================================
# Helper: update tray tooltip to show current language
# ============================================================
def update_tray_tooltip():
    global g_tray_icon
    if g_tray_icon is not None:
        lang = get_keyboard_layout()
        display = config.get_language_display_name(lang)
        try:
            g_tray_icon.title = f"Keyboard Switcher \u2014 {display}"
        except Exception:
            pass


# ============================================================
# Edge-case filter: skip detection for URLs, paths, mostly non-alpha
# (mirrors C++ filtering logic)
# ============================================================
def should_skip_detection(text: str) -> bool:
    if '://' in text or 'www.' in text or 'http' in text:
        return True
    if len(text) > 2 and text[1] == ':' and text[2] in ('\\', '/'):
        return True
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < len(text) / 2:
        return True
    return False


# ============================================================
# Order-preserving deduplication
# (mirrors C++ seen-set approach instead of set())
# ============================================================
def deduplicate_ordered(items):
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ============================================================
# Send a backspace keystroke
# ============================================================
def send_backspace():
    keyboard_controller = keyboard.Controller()
    keyboard_controller.press(keyboard.Key.backspace)
    keyboard_controller.release(keyboard.Key.backspace)
    time.sleep(0.005)


# ============================================================
# Send a string character by character
# ============================================================
def send_string(chars):
    keyboard_controller = keyboard.Controller()
    for char in chars:
        keyboard_controller.press(char)
        keyboard_controller.release(char)
        time.sleep(0.005)


# ============================================================
# Background task: keyboard & mouse hooks
# ============================================================
def background_task(cache):

    # Track shift state via pynput modifier events
    shift_held = False
    history = DetectionHistory()

    def on_keypress(key):
        nonlocal shift_held

        if not config.EnableSwitcher:
            return

        # --- Track Shift state ---
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            shift_held = True

        # --- Alt+Tab detection (mirrors C++ Alt+Tab handler) ---
        if key in (keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
            config.alt_pressed = True
            return

        if config.alt_pressed and key == keyboard.Key.tab:
            config.alt_pressed = False

            def alt_tab_handler():
                time.sleep(0.5)
                current_layout = get_keyboard_layout()
                if config.SaveWindowState:
                    window_lang = windows.get_active_window_langage()
                    if window_lang != current_layout and current_layout != config.LastSetting:
                        windows.set_active_window_langage(current_layout)
                        config.LastSetting = current_layout
                    else:
                        if window_lang is None:
                            window_lang = config.LastSetting
                        try:
                            py_win_keyboard_layout.change_foreground_window_keyboard_layout(
                                config.LANGUAGE_CODES[window_lang])
                        except Exception:
                            pass
                        config.LastSetting = window_lang
                cache.clear()
                history.clear()
                config.SEARCH = True
                update_tray_tooltip()

            threading.Thread(target=alt_tab_handler, daemon=True).start()
            return

        if key not in (keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
            config.alt_pressed = False

        # --- Track manual layout changes ---
        # If the user manually switched the layout (e.g. Alt+Shift), disable
        # detection until the next mouse click.
        current_keyboard_layout = get_keyboard_layout()
        window_lang = windows.get_active_window_langage()

        if window_lang != current_keyboard_layout:
            if config.SaveWindowState:
                windows.set_active_window_langage(current_keyboard_layout)
                config.LastSetting = current_keyboard_layout
            # Manual switch detected -> stop auto-detection until next click
            cache.clear()
            history.clear()
            config.SEARCH = False
            update_tray_tooltip()

        # --- Handle character input ---
        caps_on = False
        try:
            caps_on = ctypes.windll.user32.GetKeyState(0x14) & 0x0001
        except Exception:
            pass
        is_upper_intent = shift_held ^ bool(caps_on)

        if key == keyboard.Key.backspace:
            cache.del_char()
        elif key == keyboard.Key.space:
            cache.push_char(' ', False)
        elif key == keyboard.Key.enter:
            pass  # placeholder
        elif hasattr(key, 'char') and key.char is not None:
            cache.push_char(key.char, is_upper_intent)

        # --- Language detection (adaptive confidence) ---
        cache_size = len(cache)
        if (config.SEARCH and
                cache_size >= config.EarlyDetectionMinChars):

            # Note: per-pair parameters may require more chars than the
            # global minimum.  The global check here is a fast early-out;
            # typo_resilient_detect performs the pair-specific check after
            # it determines the best candidate language.
            current_lang_id = get_keyboard_layout()
            text = cache.get_text()

            if not should_skip_detection(text):
                # Generate all 6 layout conversion variants (same order as C++)
                text_variants = [
                    Languages.convert_text_bidirectional(text, Languages.english_layout, Languages.russian_layout),
                    Languages.convert_text_bidirectional(text, Languages.russian_layout, Languages.english_layout),
                    Languages.convert_text_bidirectional(text, Languages.hebrew_layout,  Languages.english_layout),
                    Languages.convert_text_bidirectional(text, Languages.english_layout, Languages.hebrew_layout),
                    Languages.convert_text_bidirectional(text, Languages.russian_layout, Languages.hebrew_layout),
                    Languages.convert_text_bidirectional(text, Languages.hebrew_layout,  Languages.russian_layout),
                ]

                # Order-preserving deduplication (not set())
                text_variants = deduplicate_ordered(text_variants)

                # Typo-resilient detection: consecutive agreement + drop-one boosting
                # Uses per-language-pair parameters for confidence thresholds.
                detection = typo_resilient_detect(
                    text_variants, current_lang_id, cache_size, history,
                    *model_parameters,
                    enable_typo_resilience=config.EnableTypoResilience,
                )

                if detection is not None:
                    best_lang = detection.language
                    did_correction = False

                    if current_lang_id != best_lang:
                        cached_text = cache.get_text()
                        cache_len = len(cache)

                        # Convert cached text to the detected language
                        corrected_text = Languages.convert_text_bidirectional(
                            cached_text,
                            get_layout_for_language(current_lang_id),
                            get_layout_for_language(best_lang)
                        )

                        # Preserve first-letter capitalization
                        if cache.was_first_char_shifted() and corrected_text:
                            corrected_text = corrected_text[0].upper() + corrected_text[1:]

                        # Erase the cached characters from screen
                        for _ in range(cache_len):
                            send_backspace()

                        # Change the keyboard layout to the detected language
                        py_win_keyboard_layout.change_foreground_window_keyboard_layout(
                            config.LANGUAGE_CODES[best_lang])
                        windows.set_active_window_langage(best_lang)
                        config.LastSetting = best_lang

                        time.sleep(0.05)

                        # Re-type the corrected text
                        send_string(corrected_text)

                        did_correction = True
                        update_tray_tooltip()

                    # Clear cache and disable search until next mouse click
                    cache.clear()
                    history.clear()
                    config.SEARCH = False

    def on_key_release(key):
        nonlocal shift_held
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            shift_held = False

    # Function to run when a mouse click is detected
    def on_click(x, y, button, pressed):

        if not config.EnableSwitcher:
            return

        if pressed:
            cache.clear()
            history.clear()
            config.SEARCH = True

            current_keyboard_layout = get_keyboard_layout()
            if config.SaveWindowState:
                window_lang = windows.get_active_window_langage()
                if (window_lang != current_keyboard_layout and
                        current_keyboard_layout != config.LastSetting):
                    windows.set_active_window_langage(current_keyboard_layout)
                    config.LastSetting = current_keyboard_layout
                else:
                    if window_lang is None:
                        window_lang = config.LastSetting
                    try:
                        py_win_keyboard_layout.change_foreground_window_keyboard_layout(
                            config.LANGUAGE_CODES[window_lang])
                        config.LastSetting = window_lang
                    except Exception:
                        window_lang = config.LastSetting
                        py_win_keyboard_layout.change_foreground_window_keyboard_layout(
                            config.LANGUAGE_CODES[window_lang])

            update_tray_tooltip()

    # Create and start keyboard and mouse listeners
    keyboard_listener = keyboard.Listener(on_press=on_keypress, on_release=on_key_release)
    mouse_listener = mouse.Listener(on_click=on_click)

    keyboard_listener.start()
    mouse_listener.start()


# ============================================================
# Periodic cleanup of stale window entries (every 60 seconds)
# (mirrors C++ WM_TIMER cleanup)
# ============================================================
def periodic_cleanup():
    while True:
        time.sleep(60)
        try:
            windows.cleanup()
        except Exception:
            pass


# ============================================================
# System tray icon
# ============================================================
def create_system_tray_icon():
    global g_tray_icon
    image = Image.open("keyboard.ico")

    min_chars_submenu = pystray.Menu(
        pystray.MenuItem('3 characters', lambda icon, item: set_min_chars(3),
                         checked=lambda item: config.EarlyDetectionMinChars == 3),
        pystray.MenuItem('4 characters', lambda icon, item: set_min_chars(4),
                         checked=lambda item: config.EarlyDetectionMinChars == 4),
        pystray.MenuItem('5 characters', lambda icon, item: set_min_chars(5),
                         checked=lambda item: config.EarlyDetectionMinChars == 5),
    )

    menu = pystray.Menu(
        pystray.MenuItem('Enable Switcher', enable_switcher,
                         checked=lambda item: config.EnableSwitcher, radio=True),
        pystray.MenuItem('Save window state', enable_window_state,
                         checked=lambda item: config.SaveWindowState, radio=True,
                         enabled=lambda item: config.EnableSwitcher),
        pystray.MenuItem('Typo Resilience', toggle_typo_resilience,
                         checked=lambda item: config.EnableTypoResilience, radio=True,
                         enabled=lambda item: config.EnableSwitcher),
        pystray.MenuItem('Min Chars Before Detection', min_chars_submenu,
                         enabled=lambda item: config.EnableSwitcher),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem('About', tray_about),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem('Exit', on_tray_exit)
    )
    g_tray_icon = pystray.Icon("KeyboardSwitcher", image, "Keyboard Switcher", menu)
    g_tray_icon.run()


def enable_switcher(icon, item):
    config.EnableSwitcher = not config.EnableSwitcher


def enable_window_state(icon, item):
    config.SaveWindowState = not config.SaveWindowState


def toggle_typo_resilience(icon, item):
    config.EnableTypoResilience = not config.EnableTypoResilience


def set_min_chars(value):
    config.EarlyDetectionMinChars = value


def on_tray_exit(icon, item):
    if item.text == 'Exit':
        icon.stop()


def tray_about(icon, item):
    icon.notify('Click on text area and start typing',
                f'Keyboard Switcher v{config.VERSION}')
    time.sleep(3)
    icon.remove_notification()


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":

    # Create keyboard cache
    cache = InputCache()

    # Create window tracker with current keyboard language
    language_id = str(get_keyboard_layout_info())
    windows = OpenWindows(language_id)

    # Load the ONNX model
    model_parameters = Languages.load_model()

    # Start the background task in a separate thread
    bg_thread = threading.Thread(target=background_task, args=(cache,), daemon=True)
    bg_thread.start()

    # Start periodic window cleanup thread (mirrors C++ SetTimer)
    cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
    cleanup_thread.start()

    print('Model is loaded and the background task is running')

    create_system_tray_icon()
    sys.exit(0)


