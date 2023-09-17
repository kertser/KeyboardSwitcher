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
import Constants # project constants
import time
import warnings

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


# Function to run in the background and listen for keyboard events
def background_task(cache):

    def on_keypress(key):
        if hasattr(key,'char') :
            cache.push_char(key.char)
        elif (key == keyboard.Key.space):
            cache.push_char(' ')
        elif (key == keyboard.Key.enter):
            pass # placeholder
        elif  key == keyboard.Key.backspace:
            if  len(cache)>0:
                cache.del_char()
        else: # Ignore other keys
            pass
            #print(f"Special key pressed: {key}")

        if Constants.SEARCH:

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

                # print(detected_language)

                if (detected_language != []):

                    # Get the current keyboard layout

                    language_id = str(get_keyboard_layout_info())
                    # print("current layout was",Constants.LANGUAGE_ID[language_id])

                    # Erase the cached string
                    for _ in range(len(cache)):
                        send_backspace()

                    # Change the keyboard layout to the new language
                    py_win_keyboard_layout.change_foreground_window_keyboard_layout(Constants.LANGUAGE_CODES[detected_language[0]])

                    # Print on the screen the new language (the cache with the new language)
                    send_string(cache.cache)

                    # Clear the cache
                    cache.clear()
                    # Disable the search functionality until next mouse click
                    Constants.SEARCH = False # Disable the search functionality until next mouse click


    # Function to run when a mouse click is detected
    def on_click(x, y, button, pressed):
        if pressed:
            cache.clear()  # Clear the cache
            Constants.SEARCH = True  # Set the Language Change SEARCH flag to true
            #print(f"Mouse clicked at ({x}, {y}) with button {button}")

    # Set up the keyboard listener
    keyboard_listener = keyboard.Listener(on_press=on_keypress)
    keyboard_listener.start()

    # Set up the mouse listener
    mouse_listener = mouse.Listener(on_click=on_click)
    mouse_listener.start()

# Get the keyboard layout info
def get_keyboard_layout_info():
    klid = ctypes.windll.user32.GetKeyboardLayout(0)
    language_id = klid & 0xFFFF
    return language_id

# Create a system tray icon
def create_system_tray_icon():
    image = Image.open("keyboard.ico")
    menu = pystray.Menu(pystray.MenuItem('Exit', on_tray_clicked))
    icon = pystray.Icon("KeyboardSwitcher", image, "Keyboard Switcher", menu)
    icon.run()



# Function to run when the system tray icon is clicked
def on_tray_clicked(icon, item):
    if item.text == 'Exit':
        icon.stop()


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
    model_parameters = Languages.load_model() # Loading the model parameters

    # Start the background task in a separate thread
    bg_thread = threading.Thread(target=background_task(cache), daemon=True)
    bg_thread.start()
    print('Model is loaded and the background task is running')

    create_system_tray_icon()
    sys.exit(0)




