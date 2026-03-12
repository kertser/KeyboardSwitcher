# Application version (keep in sync with C++ Config.h)
VERSION = "1.2.0"

EnableSwitcher = True # Enable switching between languages
SEARCH = False # Disable search functionality until a language is selected
SaveWindowState = True # Save the window state when switching between languages
LastSetting = 'en' # Last language selected

# ── Adaptive confidence curve parameters ──
# Detection starts at EarlyDetectionMinChars but requires very high confidence.
# As the user types more characters the required confidence drops linearly
# until FullConfidenceChars, after which the floor applies.
EarlyDetectionMinChars = 3       # earliest detection fires here
FullConfidenceChars    = 10      # confidence floor kicks in here
ConfidenceAtMinChars   = 0.97    # near-certainty at few chars
ConfidenceAtMaxChars   = 0.55    # relaxed after enough chars

def get_required_confidence(num_chars: int) -> float:
    """Compute the required softmax confidence for a given number of typed chars."""
    if num_chars < EarlyDetectionMinChars:
        return 1.1  # impossible → no detection
    if num_chars >= FullConfidenceChars:
        return ConfidenceAtMaxChars  # floor
    # Linear interpolation
    t = (num_chars - EarlyDetectionMinChars) / (FullConfidenceChars - EarlyDetectionMinChars)
    return ConfidenceAtMinChars + t * (ConfidenceAtMaxChars - ConfidenceAtMinChars)

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

# Human-readable language names (for tray tooltip)
LANGUAGE_DISPLAY_NAMES = {
    'en': 'English',
    'ru': 'Русский',
    'he': 'עברית',
}

def get_language_display_name(lang: str) -> str:
    return LANGUAGE_DISPLAY_NAMES.get(lang, 'Unknown')

# Initialize the variable to track Alt key presses
alt_pressed = False