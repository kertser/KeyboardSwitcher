# Keyboard Switcher v1.3.0

Automatically detect and switch the keyboard language (**En ↔ He ↔ Ru**) on **Windows**.

[Download Latest Release](https://github.com/kertser/KeyboardSwitcher/releases/latest)

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Detection Pipeline](#detection-pipeline)
- [Project Structure](#project-structure)
- [Model Training & Export](#model-training--export)
- [C++ Version](#c-version)
- [Packaging / Installer](#packaging--installer)
- [Windows Defender / SmartScreen](#windows-defender--smartscreen)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The main idea was to resolve the annoying situation when you are working with multiple
languages (English, Hebrew, and Russian) and constantly have to switch between them —
for example when switching between the browser (English layout for the URL) and a text
chat (Hebrew or Russian layout).

A large dataset was collected from English, Hebrew, and Russian dictionaries and used to
train a deep-learning model with an LSTM architecture that detects the intended language
with >99 % accuracy on the test set. The PyTorch model is trained on GPU but runs on
CPU without any problem; the C++ version uses the model exported to ONNX format for fast
native inference via ONNX Runtime.

## Features

### Core behaviour

| Feature | Description |
|---|---|
| **Automatic language detection** | Detects the intended language and re-types the text in the correct layout |
| **Esc to undo** | Press Escape after a correction to revert — even after typing more (up to 100 extra characters are buffered and converted back); invalidated by click, focus change, Enter, arrow / navigation keys, or manual layout switch |
| **Learned exceptions** | Rejected corrections are remembered per-word so the same mistake is never repeated; when an exception blocks one language a fallback detection tries the next-best language so the full word is still corrected. **Prefix-aware** matching blocks a correction if the proposed text is a prefix of any stored exception. **Switch-away detection**: switching away from the corrected language within 10 s of a correction is treated as a rejection — this catches the common erase + switch + retype flow. **Learned overrides**: rejecting a correction stores the reverse direction so the same text is corrected the right way next time. Per-language cap: 500 exceptions |
| **Adaptive confidence curve** | Short input requires near-certain confidence; longer input lowers the bar — reduces both false positives and false negatives |
| **Per-language-pair tuning** | Each (from→to) language pair has its own confidence thresholds, min-chars, agreement count, and borderline zone — e.g. en↔ru uses standard defaults while en/ru→he uses a tighter floor (0.75) to reduce false positives |
| **Typo resilience** | Two-tier protection: *consecutive-agreement* (requires 2+ keystrokes agreeing on a language) and *drop-one boosting* (drops one character at a time in the borderline zone to recover from a typo) |
| **Capitalization preservation** | If the first letter was typed with Shift the corrected text keeps the capital letter |

### Guards & noise filtering

| Guard | Skip reason logged | Description |
|---|---|---|
| **URL / path filter** | `skip_url_or_path` | Skips detection for `://`, `www.`, `http`, and `X:\` drive-path patterns |
| **Low-alpha filter** | `skip_low_alpha` | Requires at least *EarlyDetectionMinChars* real letter characters before detection runs |
| **Low known-chars gate** | `skip_low_known_chars` | Skips ONNX inference when fewer than *MinKnownCharsForInference* (default **2**) characters appear in the model vocabulary — prevents noise or symbol-only input from reaching the model |
| **Leaked-layout guard** | — | Rejects corrections where the source→target layout conversion would produce unmapped (leaked) characters |
| **File-dialog protection** | `skip_file_dialog_en_protection` | When the active window is a file-open / save-as dialog (`#32770` + `DirectUIHWND` or `ComboBoxEx32`) and the current layout is English, auto-switching to Hebrew or Russian is blocked — filename entry is almost always Latin. Manual layout switches (Alt+Shift) are never affected. Controlled by **DisableAutoSwitchFromEnglishInFileDialogs** (default: on) |

### Hebrew-specific improvements

| Feature | Description |
|---|---|
| **Final-form normalisation** | When a detection variant contains Hebrew characters the model also receives a normalised copy with sofit (word-final) forms replaced by their base equivalents — ך→כ, ם→מ, ן→נ, ף→פ, ץ→צ. The higher confidence score of the two runs is kept. The user's text is **never** modified by this normalisation |
| **Hebrew Script Coverage Gate** | During variant iteration, any variant with ≥ 90 % Hebrew Unicode characters (U+05D0–U+05EA) is assigned a virtual "he" confidence (coverage × 0.78), provided the ONNX model did **not** strongly favour another language for that variant. Recovers phrases like "כך רציתי" where the model returns only ~56 % English. The `AddOriginalTextAsVariant` flag (original typed text prepended to the variant list) acts as an FP suppresser — real English/Russian words are recognised from the original text and block the gate from firing |
| **Persistent Moderate Confidence Gate** | Fires when all of the last 5 frames had the same top-1 language and the average top-1 confidence ≥ 0.52. Catches "flat signal" phrases like "כל הכבוד" where the model consistently returns ~58 % Hebrew without a growing trend |
| **Cumulative Weak Score Gate** | Tracks the average softmax score for the Hebrew class (index 2) over the last 7 frames, even when it is not the top-1 winner. Fires when the average ≥ 0.28. Helps when "he" scores ~0.30–0.35 persistently but never wins the per-frame argmax |

### Window & session management

| Feature | Description |
|---|---|
| **Per-window language memory** | Remembers the language for each window/tab and restores it on focus change. Context hash is based on normalised window title; dual-hash save ensures apps whose title drifts slightly (e.g. Notepad++ prepending `*`) still restore the correct language |
| **Manual switch detection** | If the user switches the layout manually (e.g. Alt+Shift), the saved language is updated and detection is confirmed for that context. Sets `SEARCH=false` until the next context change or mouse click |
| **Alt+Tab / focus awareness** | Restores the per-window language after Alt+Tab, taskbar clicks, and virtual-desktop switches via WinEvent hooks |
| **Post-correction grace period** | Suppresses false "manual switch" detection when a window (e.g. a common file dialog) reverts the layout change that the switcher just applied |

### UI & observability

| Feature | Description |
|---|---|
| **LED state indicator** | Tray icon shows a glowing red dot while the language is undetected, switching to green once detected or manually confirmed |
| **Dynamic tray tooltip** | Hovering the tray icon shows the current language (e.g. "Keyboard Switcher — English") |
| **Confidence tuning** | Left-click the tray icon to adjust short / long text confidence thresholds via sliders |
| **System tray menu** | Right-click to enable / disable the switcher, toggle window-state saving, typo resilience, debug log, feedback collection, or exit |
| **Debug log** | Toggle from the tray menu — timestamped entries written to `ks_debug.log` next to the exe (auto-rolls at 512 KB). All skip-reason codes are logged inline. Every 60 s a **GUARD-STATS** line reports aggregate skip-reason counters so false-positive patterns are easy to spot |
| **Auto-update check** | On startup (with a short delay) and via _Check for Updates_ in the tray menu, the app queries the latest GitHub release; if a newer version is found it downloads the installer and runs it automatically |
| **Input collision guard** | Blocks real keystrokes during the retyping phase to prevent garbled output |

## Detection Pipeline

The pipeline runs on every keystroke while detection is active (`SEARCH = true`):

```
1. collect     InputCache accumulates raw keystrokes

2. sanitize    Build detectionText: alpha + spaces, trimmed
               Guards: skip_url_or_path → skip_low_alpha

3. variants    Generate up to 6 layout-converted variants (en↔ru↔he)
               + original typed text prepended as extra variant (AddOriginalTextAsVariant — FP suppresser)
               + Hebrew final-form normalised copies of any Hebrew variant
               Deduplicate

4. infer       TypoResilientDetect over all variants:
                 Per variant → PredictLanguageWithConfidence
                   → MinKnownCharsForInference gate (skip_low_known_chars)
                   → ONNX inference (softmax, 4 classes: N/A, en, he, ru)
                   → Hebrew sofit normalisation boost (if Hebrew chars present)
                   → Hebrew script coverage gate — override bestLang → "he"
                     if variant is ≥90 % Hebrew Unicode and ONNX does not
                     contradict (Tier 3-C)
                 Pick best-confidence language across all variants
                 Consecutive-agreement gate  (Tier 1)   OR
                 Trend gate                             (Tier 2)   OR
                 Drop-one boosting in borderline zone   (Tier 2)   OR
                 Persistent moderate confidence gate    (Tier 3-A) OR
                 Cumulative weak score gate             (Tier 3-B)

5. post-filter skip_file_dialog_en_protection  (en→he/ru in file dialogs)
               Leaked-layout guard
               Learned-exception check → fallback detection if blocked

6. correct     Backspace cached text → switch layout → retype corrected text
               Save language for this window / context
               Record LastCorrection for Esc-undo
               ++Guards.correctionsApplied
```

`Config::Guards` counters are incremented at each guard and dumped to the debug log
every 60 s as `GUARD-STATS: guards: emptyTok=N lowKnown=N …`.

## Project Structure

```
KeyboardSwitcher/
├── model/                   # Model training, export & tuning (Python)
│   ├── LangModel.ipynb      # Training notebook (PyTorch LSTM)
│   ├── convert_to_onnx.py   # Export trained model to ONNX
│   ├── Languages.py         # ONNX inference & layout utilities
│   ├── Languages_torch.py   # PyTorch model class & inference
│   ├── tune_confidence.py   # Offline confidence-threshold tuning
│   ├── evaluate_transitions.py  # Offline validation of all language transitions
│   ├── requirements.txt
│   ├── dictionary.json
│   ├── dictionary.pkl
│   ├── lang_model.onnx
│   ├── lang_model.pth
│   ├── results_test.csv
│   └── vocabulary/
├── cpp/                     # C++ (Win32) production app
│   ├── CMakeLists.txt
│   ├── build_release.bat    # One-click release build & package script
│   ├── lang_model.onnx
│   ├── dictionary.json
│   ├── keyboard.ico
│   ├── include/
│   │   ├── Config.h         # Version, adaptive params, language maps,
│   │   │                    #   MinKnownCharsForInference, file-dialog flag,
│   │   │                    #   SkipCounters (diagnostic guards)
│   │   ├── Languages.h      # LanguageDetector, NormalizeHebrewFinals,
│   │   │                    #   GetCachedConversionMap, TypoResilientDetect
│   │   ├── FeedbackLogger.h # User exception list & learned correction overrides,
│   │   │                    #   prefix-aware matching, switch-away detection,
│   │   │                    #   exception-fallback, per-language cap (500)
│   │   ├── InputCache.h
│   │   ├── WindowTracker.h
│   │   └── TrayIcon.h
│   ├── src/
│   │   ├── main.cpp         # Hook, detection pipeline, IsFileDialogContext,
│   │   │                    #   Hebrew-norm variant injection, guard counters,
│   │   │                    #   clipboard save/restore, stale-timer guard,
│   │   │                    #   Esc-undo, switch-away detection
│   │   ├── Config.cpp       # Params, per-pair overrides, SkipCounters impl
│   │   ├── Languages.cpp    # ONNX inference, RunInference helper,
│   │   │                    #   NormalizeHebrewFinals, GetCachedConversionMap,
│   │   │                    #   HistoryFrame (scores[4] softmax vector),
│   │   │                    #   MAX_WINDOW=10, Iteration-3 gate logic
│   │   ├── FeedbackLogger.cpp
│   │   ├── InputCache.cpp
│   │   ├── WindowTracker.cpp
│   │   └── TrayIcon.cpp
│   ├── resources/
│   │   ├── resource.h
│   │   ├── resource.rc.in   # Template with VERSIONINFO (processed by CMake)
│   │   └── resource.rc
│   ├── onnxruntime/         # ONNX Runtime SDK (not checked in)
│   └── dist/                # Release build output (not checked in)
├── LICENSE
└── README.md
```

## Model Training & Export

The `model/` directory contains the PyTorch LSTM training pipeline and supporting utilities.
The C++ app does **not** depend on Python at runtime — it uses the exported ONNX model.

```bash
cd model
pip install -r requirements.txt
# Train the model in LangModel.ipynb, then export:
python convert_to_onnx.py
# Copy lang_model.onnx and dictionary.json into cpp/
```

Key files:

| File | Purpose |
|---|---|
| `LangModel.ipynb` | Training notebook (PyTorch LSTM) |
| `convert_to_onnx.py` | Export `.pth` → `.onnx` |
| `Languages.py` | ONNX inference & keyboard-layout conversion utilities |
| `Languages_torch.py` | PyTorch model class (used during training & export) |
| `tune_confidence.py` | Offline grid-search for adaptive confidence-curve parameters |
| `evaluate_transitions.py` | Offline validation of all directed language transitions (en↔ru↔he) on vocabulary dictionaries |
| `vocabulary/` | Word lists (English, Hebrew, Russian) for tuning |

## C++ Version

### Prerequisites
- **CMake** 3.20+
- **MinGW-w64** or **MSVC** (Visual Studio 2019/2022) — any C++17 compiler on Windows
- **ONNX Runtime** C++ SDK ([download](https://github.com/microsoft/onnxruntime/releases))

### Build

1. Download ONNX Runtime and extract it to `cpp/onnxruntime/`:
   ```
   cpp/onnxruntime/
   ├── include/
   │   └── onnxruntime_cxx_api.h  (and other headers)
   └── lib/
       ├── onnxruntime.lib
       ├── onnxruntime.dll
       └── onnxruntime_providers_shared.dll
   ```

2. **Debug build** (from CLion or command line):
   ```bash
   cd cpp
   cmake -B cmake-build-debug
   cmake --build cmake-build-debug
   ```

3. **Release build for distribution** (self-contained, no MinGW DLLs needed):
   ```bash
   cd cpp
   cmake -B cmake-build-release -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles"
   cmake --build cmake-build-release --config Release
   cmake --install cmake-build-release --prefix ./dist
   ```
   The `dist/` folder will contain everything needed to run the application:
   ```
   dist/
   ├── KeyboardSwitcher.exe
   ├── onnxruntime.dll
   ├── onnxruntime_providers_shared.dll
   ├── lang_model.onnx
   ├── dictionary.json
   └── keyboard.ico
   ```

> **Note:** When building with MinGW, the CMake configuration automatically
> applies `-static-libgcc -static-libstdc++ -static` so the resulting
> executable only depends on standard Windows system DLLs and the bundled
> ONNX Runtime DLL — no MinGW runtime DLLs are required on the target machine.

### Key configuration knobs (`Config.h` / `Config.cpp`)

| Parameter | Default | Description |
|---|---|---|
| `EarlyDetectionMinChars` | 3 | Global minimum alpha chars before any detection runs |
| `FullConfidenceChars` | 15 | Chars at which the confidence floor kicks in |
| `ConfidenceAtMinChars` | 0.99 | Required confidence at *EarlyDetectionMinChars* |
| `ConfidenceAtMaxChars` | 0.70 | Confidence floor at *FullConfidenceChars* and beyond |
| `ConsecutiveAgreementCount` | 2 | Consecutive keystrokes that must agree before switching |
| `BorderlineZoneFactor` | 0.85 | Drop-one boosting fires in `[threshold × factor, threshold]` |
| `MinKnownCharsForInference` | 2 | Minimum chars in the model vocabulary; below this the inference call is skipped (`skip_low_known_chars`) |
| `DisableAutoSwitchFromEnglishInFileDialogs` | true | Block en→he/ru auto-switch when a file-open/save dialog is active |
| `EnableTypoResilience` | true | Master toggle for consecutive-agreement and drop-one boosting |
| `EnableHebrewScriptGate` | true | Enable the Hebrew script coverage gate (Tier 3-C) |
| `EnablePersistentConfGate` | true | Enable the persistent moderate confidence gate (Tier 3-A) |
| `EnableWeakScoreGate` | true | Enable the cumulative weak score gate (Tier 3-B) |
| `AddOriginalTextAsVariant` | true | Prepend original typed text as a variant to suppress false positives |
| `HebrewScriptCoverageThreshold` | 0.90 | Fraction of alpha chars that must be Hebrew Unicode for the script gate to fire |
| `HebrewScriptVirtualConf` | 0.78 | Virtual confidence assigned to a variant that passes the script coverage gate |
| `PersistentMinSteps` | 5 | Number of consecutive history frames needed for the persistent confidence gate |
| `PersistentMinAvgConf` | 0.52 | Average top-1 confidence threshold for the persistent gate |
| `WeakScoreWindow` | 7 | History window (frames) for the cumulative weak score gate |
| `WeakScoreMinAvg` | 0.28 | Average Hebrew softmax score needed to fire the weak score gate |
| `WeakScoreClassIdx` | 2 | Softmax class index tracked by the weak score gate (2 = Hebrew) |

Per-pair overrides (e.g. `{"en","he"}` and `{"ru","he"}`) raise `ConfidenceAtMaxChars`
to **0.75**, keep `BorderlineZoneFactor` at **0.88**, and tighten `PhraseConfScale`
to **0.72** (down from 0.80) to reduce false positives on pairs where one script
is detected more ambiguously.

### Dependencies
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — ONNX model inference
- [nlohmann/json](https://github.com/nlohmann/json) — JSON parsing (fetched automatically by CMake)
- Win32 API — keyboard/mouse hooks, system tray, keyboard layout switching

## Packaging / Installer

The build system uses **CPack** to produce distributable packages.
Run the following commands from the release build directory (`cmake-build-release/`):

| Format | Command | Output | Requires |
|---|---|---|---|
| **ZIP** (portable) | `cpack -G ZIP` | `KeyboardSwitcher-<version>-win64.zip` | Nothing extra |
| **NSIS** (installer) | `cpack -G NSIS` | `KeyboardSwitcher-<version>-win64.exe` | [NSIS](https://nsis.sourceforge.io/Download) on PATH |

The **NSIS installer** provides a standard Windows setup wizard with:
- Install / uninstall via *Add or Remove Programs*
- Start-menu shortcut
- License agreement page

The **ZIP** is a portable archive — just extract and run.

### Full example (build + package)

**One-click script** (recommended):
```bash
cd cpp
build_release.bat
```

Or manually:
```bash
cd cpp
cmake -B cmake-build-release -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles"
cmake --build cmake-build-release --config Release
cmake --install cmake-build-release --prefix ./dist
cd cmake-build-release
cpack -G ZIP        # portable archive
cpack -G NSIS       # installer (requires NSIS)
```

## Windows Defender / SmartScreen

Because the executable is not code-signed, Windows SmartScreen may show a warning on
first run. Click **More info → Run anyway** to proceed. The source code is fully open
and the build is reproducible from this repository.

## Usage

1. Run the program — it appears as a tray icon with a coloured LED dot:
   **red** while the language is undetected, **green** once detected or manually confirmed.
2. Click on a text area and start typing.
3. The switcher detects the intended language and corrects the input automatically.
4. **Press Esc** to undo the last correction (works even after typing more — up to 100 extra characters).
5. **Left-click** the tray icon to adjust confidence thresholds via sliders.
6. **Right-click** the tray icon for settings:
   - Enable / disable the switcher
   - Toggle per-window language memory
   - Toggle typo resilience
   - Toggle debug log (`ks_debug.log` next to the exe)
   - Toggle feedback collection
   - Reset learned exceptions
   - Check for updates
   - Exit
7. Hover the tray icon to see the current keyboard layout.
8. On startup the app silently checks GitHub for a newer release; if one is found you are prompted to download and install it. You can also check manually via **Check for Updates…** in the tray menu.

### Troubleshooting with the debug log

Enable **Debug Log** from the tray menu. The log file `ks_debug.log` (next to the exe)
records every detection decision and every guard skip with its reason code:

| Log tag | Meaning |
|---|---|
| `DETECTION:` | Model fired — language, confidence, alpha count |
| `CORRECT:` | Correction applied — cached text → corrected text |
| `SKIP: skip_low_alpha` | Not enough letter characters yet |
| `SKIP: skip_url_or_path` | URL or file-path pattern detected |
| `SKIP: skip_low_known_chars` | Fewer than *MinKnownCharsForInference* chars in vocabulary |
| `SKIP: skip_file_dialog_en_protection` | En→He/Ru blocked in a file-open/save dialog |
| `REJECT: leaked chars` | Layout conversion would produce unmapped characters |
| `GUARD-STATS:` | Aggregate counter dump (every 60 s while debug log is on) |
| `UNDO:` | Esc-undo reverting a correction |
| `MANUAL-SWITCH:` | User switched layout manually |
| `RESTORE-LANG:` | Per-window saved language restored on focus change |

## Contributing

Pull requests are welcome!

## License

This project is licensed under the MIT License — Copyright © 2025-2026 Alpha-Numerical.
