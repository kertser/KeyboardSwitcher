EnableSwitcher = True # Enable switching between languages
SEARCH = False # Disable search functionality until a language is selected
SaveWindowState = True # Save the window state when switching between languages
LastSetting = 'en' # Last language selected

# Language codes for English, Russian, and Hebrew
LANGUAGE_CODES = {
    'en': 0x04090409, # English (United States)
    'ru': 0x4190419, # Russian
    'he': -0xFC2FBF3 # Hebrew (Israel)
}

LANGUAGE_ID = {
    '1033' : 'en', # English (United States)
    '1049' : 'ru', # Russian
    '1037' : 'he' # Hebrew (Israel)
}