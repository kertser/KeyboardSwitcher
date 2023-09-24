# Importing the necessary packages
import keyboard
import threading
import pystray
import Languages
from Languages import convert_text_bidirectional
from PIL import Image
from dataclasses import dataclass
import py_win_keyboard_layout
from pynput import keyboard, mouse
import ctypes # for keyboard layout codes
import sys
import config # project constants and settings
import time
import warnings
import pygetwindow as gw # window title search

# Ignore all warnings:
warnings.filterwarnings("ignore")

# Class to hold the keyboard layout
@dataclass
class InputCache():
    def __init__(self):
        self.cache = []

    def push_char(self, char:str):
        self.cache.append(char)

    def del_char(self):
        if len(self.cache) == 0:
            return None
        return self.cache[:-1]

    def clear(self):
        self.cache.clear()

    def __len__(self):
        return len(self.cache)

    def checkLang(self):
        pass

@dataclass
class OpenWindows():

    # Initialize the list of open windows
    def __init__(self,language):

        # Dictionary to hold the window title and the language setup
        self.windows_language = {}
        available_windows = gw.getAllTitles()
        for window in available_windows:
            self.windows_language[window] = config.LANGUAGE_ID[language]
            # print(f"Window: {window} Language: {self.windows_language[window]}")

    # Get the language setup for the active window
    def get_active_window_langage(self):
        # Get the active window title
        try:
            return self.windows_language[gw.getActiveWindowTitle()]
        except:
            # No active window title found
            return None

    # Set the language for the active window
    def set_active_window_langage(self,language):
        self.windows_language[gw.getActiveWindowTitle()] = language

    # update windows
    def update_window_titles(self):
        available_windows = gw.getAllTitles()
        updated_windows_language = {}
        for window in available_windows:
            if window in self.windows_language.keys():
                updated_windows_language[window] = self.windows_language[window]
        self.windows_language = updated_windows_language

def get_keyboard_layout():
    # Get the keyboard layout for windows
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    klid = ctypes.windll.user32.GetKeyboardLayout(ctypes.windll.user32.GetWindowThreadProcessId(hwnd, 0))

    # Extract the language ID from the keyboard layout
    lang_id = klid & 0xFFFF

    return config.LANGUAGE_ID[str(lang_id)]


# Function to run in the background and listen for keyboard events
def background_task(cache):

    # Keyboard listener
    def on_keypress(key):

        # If the switcher is disabled, do nothing
        if config.EnableSwitcher == False:
            return

        # Check if Alt+Tab has been pressed
        if key == keyboard.Key.alt_l or key == keyboard.Key.alt_r or key == keyboard.Key.alt_gr:
            config.alt_pressed = True
        elif config.alt_pressed and key == keyboard.Key.tab:
            config.alt_pressed = False
            # Initiate the on_click event

            current_window = gw.getActiveWindow()

            # Check if there's an active window
            if current_window is not None:
                # Bring the current window to the front and click

                ##! Minor bug with window switching delay
                #current_window.activate()
                #time.sleep(0.1)  # Wait for the window to be activated
                on_click(0, 0, 0, True) # Click the window

            return


        # Get current keyboard layout
        current_keyboard_layout = get_keyboard_layout()

        # if language is manually changed by keyboard, update the window titles
        if windows.get_active_window_langage() != current_keyboard_layout:

            if  config.SaveWindowState == True:
                windows.set_active_window_langage(current_keyboard_layout)
                config.LastSetting = windows.get_active_window_langage()

        # If char key is pressed, add it to the cache
        if hasattr(key,'char'):
            cache.push_char(key.char)
        elif (key == keyboard.Key.space):
            cache.push_char(' ')
        elif (key == keyboard.Key.enter):
            pass # placeholder
        elif key == keyboard.Key.backspace:
            if  len(cache)>0:
                cache.del_char()
        else: # Ignore other keys
            pass
            # print(f"Special key pressed: {key}")

        if config.SEARCH:

            text = ''.join(cache.cache)
            text_variants = [Languages.convert_text_bidirectional(text, Languages.english_layout, Languages.russian_layout),
                             Languages.convert_text_bidirectional(text, Languages.russian_layout, Languages.english_layout),
                             Languages.convert_text_bidirectional(text, Languages.hebrew_layout, Languages.english_layout),
                             Languages.convert_text_bidirectional(text, Languages.english_layout, Languages.hebrew_layout),
                             Languages.convert_text_bidirectional(text, Languages.russian_layout, Languages.hebrew_layout),
                             Languages.convert_text_bidirectional(text, Languages.hebrew_layout, Languages.russian_layout),]

            text_variants = list(set(text_variants))
            # print(text_variants) # For Debugging

            if len(cache) > 3:

                detected_language = [Languages.predict_language(text_variant,*model_parameters) for text_variant in text_variants]
                # Filter the 'N/A' values from the list
                detected_language = list(filter(None, detected_language))


                if (detected_language != []):

                    language_id = str(get_keyboard_layout_info())
                    # print("current layout was",config.LANGUAGE_ID[language_id])

                    if config.LANGUAGE_ID[language_id] != detected_language[0]:

                        # Erase the cached string
                        for _ in range(len(cache)):
                            send_backspace()

                        # Change the keyboard layout to the new language
                        py_win_keyboard_layout.change_foreground_window_keyboard_layout(config.LANGUAGE_CODES[detected_language[0]])
                        windows.set_active_window_langage(detected_language[0])
                        config.LastSetting = detected_language[0] # Set the last setting to the new language

                        # Print on the screen the new language (the cache with the new language)
                        send_string(cache.cache)

                    # Clear the cache
                    cache.clear()

                    # Disable the search functionality until next mouse click
                    config.SEARCH = False # Disable the search functionality until next mouse click

    # Function to run when a mouse click is detected
    def on_click(x, y, button, pressed):

        # If the switcher is disabled, do nothing
        if config.EnableSwitcher == False:
            return

        if pressed: # If the mouse button is pressed

            cache.clear()  # Clear the key cache
            config.SEARCH = True  # Set the Language Change SEARCH flag to true
            #print(f"Mouse clicked at ({x}, {y}) with button {button}")

        # Get current keyboard layout
        current_keyboard_layout = get_keyboard_layout()

        # if language is manually changed, update the window language
        if (windows.get_active_window_langage() != current_keyboard_layout) and (current_keyboard_layout != config.LastSetting):
            # manual_setting = True
            if config.SaveWindowState == True:
                windows.set_active_window_langage(current_keyboard_layout)
                config.LastSetting = current_keyboard_layout
        else:
            # manual_setting = False

            if config.SaveWindowState == True:
                # Check if the active window language is None (meaning no active window)
                active_window_language = windows.get_active_window_langage()

                if active_window_language is None:
                    active_window_language = config.LastSetting

                # Change the keyboard layout to the window language
                try:
                    py_win_keyboard_layout.change_foreground_window_keyboard_layout(
                        config.LANGUAGE_CODES[active_window_language])
                    config.LastSetting = active_window_language  # Set the last setting to the new language
                except:
                    # unable to change keyboard layout properly
                    active_window_language = config.LastSetting
                    py_win_keyboard_layout.change_foreground_window_keyboard_layout(
                        config.LANGUAGE_CODES[active_window_language])

    # Create and start a keyboard and mouse listeners on start
    keyboard_listener = keyboard.Listener(on_press=on_keypress)
    mouse_listener = mouse.Listener(on_click=on_click)

    keyboard_listener.start()
    mouse_listener.start()

# Get the keyboard layout info
def get_keyboard_layout_info():
    return py_win_keyboard_layout.get_foreground_window_keyboard_layout() & 0xFFFF

# Create a system tray icon
def create_system_tray_icon():
    image = Image.open("keyboard.ico")

    menu = pystray.Menu(
        #set radio button for search
        pystray.MenuItem('Enable Switcher', enable_switcher, checked=lambda item: config.EnableSwitcher, radio=True),
        pystray.MenuItem('Save window state', enable_window_state, checked=lambda item: config.SaveWindowState, radio=True, enabled=lambda item: config.EnableSwitcher),
        pystray.MenuItem('About', tray_about),
        pystray.Menu.SEPARATOR,  # Separator
        pystray.MenuItem('Exit', on_tray_exit)
        )
    icon = pystray.Icon("KeyboardSwitcher", image, "Keyboard Switcher", menu)
    icon.run()

# Enable/Disable the Application
def enable_switcher(icon, item):
    config.EnableSwitcher = not(config.EnableSwitcher)
    # Disable menu item "Save window state" if the switcher is disable

# Enable/Disable the window state saving
def enable_window_state(icon, item):
    config.SaveWindowState = not(config.SaveWindowState)

# Function to run when the system tray exit button is clicked
def on_tray_exit(icon, item):
    if item.text == 'Exit':
        icon.stop()

# Display "About" information in the system tray
def tray_about(icon, item):
    icon.notify('Click on text area and start typing','Keyboard Switcher v1.1')
    time.sleep(3)
    icon.remove_notification()


# Send a backspace to the keyboard
def send_backspace():
    keyboard_controller = keyboard.Controller()
    with keyboard_controller.pressed(keyboard.Key.backspace):
        pass
        #time.sleep(0.01)  # Adjust the duration as needed (in seconds)

# Send a key to the keyboard
def send_string(cache):
    # Create a keyboard controller
    keyboard_controller = keyboard.Controller()

    for char in cache:
        # Simulate pressing and releasing the character key
        with keyboard_controller.pressed(char):
            pass
            #time.sleep(0.01)  # Adjust the duration as needed (in seconds)


def switcher(text, layout_from, layout_to):
    converted_text = convert_text_bidirectional(text, layout_from, layout_to)
    return converted_text


if __name__ == "__main__":

    # Create kerboard cache
    cache = InputCache()

    # Create a dataclass of open windows
    language_id = str(get_keyboard_layout_info())
    windows = OpenWindows(language_id)

    model_parameters = Languages.load_model() # Loading the model parameters

    # Start the background task in a separate thread
    bg_thread = threading.Thread(target=background_task(cache), daemon=True)
    bg_thread.start()
    print('Model is loaded and the background task is running')

    create_system_tray_icon()
    sys.exit(0)




