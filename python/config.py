# Application version (keep in sync with C++ Config.h)
VERSION = "1.2.3"

EnableSwitcher = True # Enable switching between languages
SEARCH = False # Disable search functionality until a language is selected
SaveWindowState = True # Save the window state when switching between languages
LastSetting = 'en' # Last language selected

# ================================================================
# Per-language-pair adaptive switching parameters
# ================================================================
# Groups all tunable detection parameters. Each (from→to) language
# pair can have its own set of thresholds. A global default is used
# as the fallback for any pair that is not explicitly overridden.

class SwitchingParams:
    """Holds all adaptive confidence-curve and typo-resilience knobs."""
    __slots__ = ('EarlyDetectionMinChars', 'FullConfidenceChars',
                 'ConfidenceAtMinChars', 'ConfidenceAtMaxChars',
                 'ConsecutiveAgreementCount', 'BorderlineZoneFactor')

    def __init__(self, *,
                 EarlyDetectionMinChars: int = 3,
                 FullConfidenceChars: int = 10,
                 ConfidenceAtMinChars: float = 0.97,
                 ConfidenceAtMaxChars: float = 0.55,
                 ConsecutiveAgreementCount: int = 2,
                 BorderlineZoneFactor: float = 0.85):
        self.EarlyDetectionMinChars = EarlyDetectionMinChars
        self.FullConfidenceChars = FullConfidenceChars
        self.ConfidenceAtMinChars = ConfidenceAtMinChars
        self.ConfidenceAtMaxChars = ConfidenceAtMaxChars
        self.ConsecutiveAgreementCount = ConsecutiveAgreementCount
        self.BorderlineZoneFactor = BorderlineZoneFactor

    def get_required_confidence(self, num_chars: int) -> float:
        """Compute the required softmax confidence for a given number of typed chars."""
        if num_chars < self.EarlyDetectionMinChars:
            return 1.1  # impossible → no detection
        if num_chars >= self.FullConfidenceChars:
            return self.ConfidenceAtMaxChars  # floor
        # Linear interpolation
        t = (num_chars - self.EarlyDetectionMinChars) / (self.FullConfidenceChars - self.EarlyDetectionMinChars)
        return self.ConfidenceAtMinChars + t * (self.ConfidenceAtMaxChars - self.ConfidenceAtMinChars)


# ── Global default parameters ──────────────────────────────────
DefaultParams = SwitchingParams()

# ── Per-pair overrides ─────────────────────────────────────────
# Different language pairs have different detection characteristics:
#
#  en↔ru : Distinct scripts (Latin vs Cyrillic). The model is very
#          confident even on short input → standard defaults work.
#
#  en↔he : Distinct scripts (Latin vs Hebrew). Hebrew has no upper-
#          case, so typed-on-English is always lowercase — slightly
#          easier to confuse with short English. Require a bit more
#          confidence at short lengths.
#
#  ru↔he : Both non-Latin. Physical key positions overlap less, and
#          the model may need more context. Slightly higher min-chars
#          and confidence give more room to discriminate.
#
# Pairs not listed here automatically fall back to DefaultParams.
PairOverrides: dict[tuple[str, str], SwitchingParams] = {
    # ── English ↔ Russian ─── (standard — same as default)
    ("en", "ru"): SwitchingParams(),
    ("ru", "en"): SwitchingParams(),

    # ── English ↔ Hebrew ─── (slightly stricter at short lengths)
    ("en", "he"): SwitchingParams(ConfidenceAtMinChars=0.98, ConfidenceAtMaxChars=0.60),
    ("he", "en"): SwitchingParams(ConfidenceAtMinChars=0.98, ConfidenceAtMaxChars=0.60),

    # ── Russian ↔ Hebrew ─── (both non-Latin; need more context)
    ("ru", "he"): SwitchingParams(EarlyDetectionMinChars=4, ConfidenceAtMinChars=0.98,
                                   ConfidenceAtMaxChars=0.60, BorderlineZoneFactor=0.80),
    ("he", "ru"): SwitchingParams(EarlyDetectionMinChars=4, ConfidenceAtMinChars=0.98,
                                   ConfidenceAtMaxChars=0.60, BorderlineZoneFactor=0.80),
}


def get_params_for_pair(from_lang: str, to_lang: str) -> SwitchingParams:
    """Look up the switching parameters for a specific (from→to) pair.
    Returns the pair-specific override if present, otherwise DefaultParams."""
    return PairOverrides.get((from_lang, to_lang), DefaultParams)


# ── Legacy global accessors (read from DefaultParams) ──────────
# These are kept so that existing code (tray menu, etc.) can read/write
# the global default values directly.
EarlyDetectionMinChars = DefaultParams.EarlyDetectionMinChars
FullConfidenceChars    = DefaultParams.FullConfidenceChars
ConfidenceAtMinChars   = DefaultParams.ConfidenceAtMinChars
ConfidenceAtMaxChars   = DefaultParams.ConfidenceAtMaxChars
ConsecutiveAgreementCount = DefaultParams.ConsecutiveAgreementCount
BorderlineZoneFactor   = DefaultParams.BorderlineZoneFactor

# Master toggle for typo resilience (applies to all pairs)
EnableTypoResilience   = True


def get_required_confidence(num_chars: int) -> float:
    """Compute required confidence using the global default params."""
    return DefaultParams.get_required_confidence(num_chars)

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