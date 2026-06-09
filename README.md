# Keyboard Switcher v1.4.1

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
| **Learned exceptions** | Rejected corrections are remembered per-word so the same mistake is never repeated; when an exception blocks one language a fallback detection tries the next-best language so the full word is still corrected |
| **Adaptive confidence curve** | Short input requires near-certain confidence; longer input lowers the bar — reduces both false positives and false negatives |
| **Per-language-pair tuning** | Each (from→to) pair has its own `EarlyDetectionMinChars`, `ConfidenceAtMinChars`, `ConfidenceAtMaxChars`, `ConsecutiveAgreementCount`, and `BorderlineZoneFactor`. For example, `en/ru→he` uses `EarlyMin=3` and a relaxed floor of `0.60` (sweep-validated at 84 % TP, <2 % FP on 200-word vocabulary samples) |
| **Adaptive calibration** | Silently tunes per-pair `ConfidenceAtMaxChars` and `MinTop1Top2Margin` based on real usage. Rejected corrections and manual overrides signal "too aggressive"; accepted corrections (no undo within 10 s) signal "just right"; unprompted manual switches signal "too strict". An EWMA controller (α=0.2, min 5 events per batch, hysteresis band 0.15) steps thresholds by ≤ 0.01 / 0.005 per adaptation with asymmetric clamps — tightening up to +0.10 / +0.08, loosening limited to −0.05 / −0.01 relative to the factory baseline. State is persisted in `user_prefs.json` alongside exceptions, invalidated on version change, and reset via the tray menu. Fully invisible: no UI changes, adaptation noted only in the debug log |
| **Typo resilience** | Two-tier protection: *consecutive-agreement* (requires 2+ keystrokes agreeing on a language) and *drop-one boosting* (drops one character at a time in the borderline zone to recover from a typo) |
| **Capitalization preservation** | Per-character: every alpha character typed with Shift intent is uppercased in the corrected text. Works for sentence-initial caps ("Hello"), CamelCase ("iPhone"), and ALL-CAPS abbreviations ("IDF"). On layouts without uppercase (Hebrew), `VkToWchar` stores the base character + shift flag; the correction path restores uppercase from the recorded `shiftState_` vector via `GetShiftStates()` |
| **Async correction (race-free)** | The backspace + retype sequence is dispatched to a detached worker thread so the hook returns `1` immediately — the triggering key is reliably blocked even when ONNX inference takes longer than Windows' `LowLevelHooksTimeout`. Real user keystrokes are eaten by the `g_isSendingInput` guard until the worker finishes |

### Guards & noise filtering

| Guard | Skip / log tag | Description |
|---|---|---|
| **URL / path filter** | `skip_url_or_path` | Skips detection for `://`, `www.`, `http`, and `X:\` drive-path patterns |
| **Low-alpha filter** | `skip_low_alpha` | Requires at least *EarlyDetectionMinChars* real letter characters before detection runs |
| **Low known-chars gate** | `skip_low_known_chars` | Skips ONNX inference when fewer than *MinKnownCharsForInference* (default **2**) characters appear in the model vocabulary — prevents noise or symbol-only input from reaching the model |
| **Leaked-layout guard** | — | Rejects corrections where the source→target layout conversion would produce unmapped (leaked) characters |
| **File-dialog protection** | `skip_file_dialog_en_protection` | When the active window is a file-open / save-as dialog (`#32770` + `DirectUIHWND` or `ComboBoxEx32`) and the current layout is English, auto-switching to Hebrew or Russian is blocked — filename entry is almost always Latin. Manual layout switches (Alt+Shift) are never affected. Controlled by **DisableAutoSwitchFromEnglishInFileDialogs** (default: on) |
| **Case-signal Hebrew exclusion** | `EXCL: Hebrew excluded` | Hebrew has no uppercase letters. When the cached word contains ≥ *CaseExclusionMinCaps* (default **2**) alpha characters typed with Shift/CapsLock intent, **or** any internal capital (e.g. `iPhone`, `myVar`), Hebrew is removed from the candidate set before inference. Sentence-initial capitals (`"Hello"`) are **not** counted. Validated: 100 % of model Hebrew FPs on ALL-CAPS / CamelCase words eliminated; 0 real Hebrew TPs blocked |

### Hebrew-specific improvements

| Feature | Description |
|---|---|
| **Final-form normalisation** | When a detection variant contains Hebrew characters the model also receives a normalised copy with sofit (word-final) forms replaced by their base equivalents — ך→כ, ם→מ, ן→נ, ף→פ, ץ→צ. The higher confidence score of the two runs is kept. The user's text is **never** modified by this normalisation |

### Window & session management

| Feature | Description |
|---|---|
| **Per-window language memory** | Remembers the language for each window/tab and restores it on focus change |
| **Manual switch detection** | If the user switches the layout manually (e.g. Alt+Shift), the saved language is updated and detection is confirmed for that context |
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
1. collect     InputCache accumulates raw keystrokes + per-character Shift/CapsLock state

2. sanitize    Build detectionText: alpha + spaces, trimmed
               Guards: skip_url_or_path → skip_low_alpha

3. case-excl   Inspect InputCache shift state:
               if UpperCount >= CaseExclusionMinCaps  OR  HasInternalCapital:
                   add "he" to excludedLangs           -> EXCL log tag
               (sentence-initial capitals are ignored; Hebrew text never triggers this)

4. variants    Generate source-restricted layout-conversion variants:
               identity (text as typed on currentLang)
               + currentLang -> each other layout  [up to 2 more variants]
               Lossy conversions (HasLeakedLayoutChars) are skipped.
               + Hebrew final-form normalised copies of any Hebrew variant
               Deduplicate.
               Restriction to currentLang-sourced conversions removes a
               latent variant/direction mismatch and halves ONNX inferences,
               keeping the hook well under LowLevelHooksTimeout.

5. infer       TypoResilientDetect over all variants (excludedLangs passed in):
                 Per variant -> PredictLanguageWithConfidence
                   -> MinKnownCharsForInference gate (skip_low_known_chars)
                   -> ONNX inference (softmax, 4 classes: N/A, en, he, ru)
                   -> Hebrew sofit normalisation boost (if Hebrew chars present)
                 Skip candidates in excludedLangs
                 Pick best-confidence language across remaining variants
                 Consecutive-agreement gate  (Tier 1)   <- always runs, even with exclusions
                 Drop-one boosting in borderline zone    (Tier 2)
                 Thresholds read from live-calibrated PairOverrides entry

6. post-filter skip_file_dialog_en_protection  (en->he/ru in file dialogs)
               Leaked-layout guard
               Learned-exception check -> fallback detection if blocked
                 (fallback inherits excludedLangs, skips agreement gate)

7. correct     ASYNC dispatch to detached worker thread (race-free):
                 BackspaceN-1 -> switch layout -> retype corrected text
               Shift states from InputCache applied per-character to
               restore original capitalisation (ALL-CAPS, CamelCase).
               Save language for this window / context.
               Record LastCorrection for Esc-undo.
               ++Guards.correctionsApplied

8. feedback    Outcome signals routed to adaptive calibration controller:
               FP  <- Esc-undo or manual layout override within 10 s of correction
               TP  <- correction window expires with no rejection (context change,
                      nav key, Enter, or buffer overflow after 10 s)
               FN  <- unprompted manual layout switch (no recent correction)
               Controller updates per-pair EWMA; adapts ConfidenceAtMaxChars and
               MinTop1Top2Margin when batch >= 5 events and |pressure| > 0.15
```

`Config::Guards` counters are incremented at each guard and dumped to the debug log
every 60 s as `GUARD-STATS: guards: emptyTok=N lowKnown=N ...`.

## Project Structure

```
KeyboardSwitcher/
├── model/                   # Model training, export & tuning (Python)
│   ├── LangModel.ipynb      # Training notebook (PyTorch LSTM)
│   ├── convert_to_onnx.py   # Export trained model to ONNX
│   ├── Languages.py         # ONNX inference & keyboard-layout conversion utilities
│   ├── Languages_torch.py   # PyTorch model class (used during training & export)
│   ├── tune_confidence.py   # Offline grid-search for global adaptive confidence-curve parameters
│   ├── evaluate_transitions.py  # Per-directed-pair TP/FP validation on vocabulary
│   ├── sweep_he_params.py   # Parameter sweep for Hebrew-target pairs
│   ├── test_case_exclusion.py   # Validates case-signal Hebrew exclusion (ALL-CAPS, CamelCase)
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
│   ├── test_calibration.py  # Unit tests for the adaptive calibration controller
│   ├── lang_model.onnx
│   ├── dictionary.json
│   ├── keyboard.ico
│   ├── include/
│   │   ├── Config.h         # Version, adaptive params, language maps,
│   │   │                    #   MinKnownCharsForInference, file-dialog flag,
│   │   │                    #   case-exclusion flags, SkipCounters (guards),
│   │   │                    #   ApplyAdaptedParams (calibration sink)
│   │   ├── Languages.h      # LanguageDetector, NormalizeHebrewFinals,
│   │   │                    #   GetCachedConversionMap, TypoResilientDetect
│   │   │                    #   (excludedLangs + isFallback parameters)
│   │   ├── FeedbackLogger.h # User exception list, learned correction overrides,
│   │   │                    #   adaptive calibration (Outcome enum,
│   │   │                    #   RecordOutcome, ResetCalibration)
│   │   ├── InputCache.h     # Per-keystroke buffer; UpperCount, HasInternalCapital,
│   │   │                    #   GetShiftStates (per-char capitalisation restore)
│   │   ├── WindowTracker.h
│   │   └── TrayIcon.h
│   ├── src/
│   │   ├── main.cpp         # Hook, detection pipeline, case-signal exclusion,
│   │   │                    #   source-restricted variant generation,
│   │   │                    #   SyncGlobalEarlyOut (startup + flyout),
│   │   │                    #   async correction dispatch (race-free),
│   │   │                    #   IsFileDialogContext, Hebrew-norm variant injection,
│   │   │                    #   per-char capitalisation restore, guard counters,
│   │   │                    #   MaybeConfirmTP + calibration signal wiring
│   │   ├── Config.cpp       # Params, per-pair overrides, SkipCounters impl,
│   │   │                    #   ApplyAdaptedParams (live calibration sink)
│   │   ├── Languages.cpp    # ONNX inference, RunInference helper,
│   │   │                    #   NormalizeHebrewFinals, GetCachedConversionMap,
│   │   │                    #   TypoResilientDetect (isFallback decoupled);
│   │   │                    #   PredictLanguage removed (dead code)
│   │   ├── FeedbackLogger.cpp  # EWMA calibration controller, PairCalibration
│   │   │                    #   state machine, exceptions, overrides,
│   │   │                    #   user_prefs.json persistence
│   │   ├── InputCache.cpp   # UpperCount, HasInternalCapital, GetShiftStates implementations
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
| `tune_confidence.py` | Offline grid-search for global adaptive confidence-curve parameters |
| `evaluate_transitions.py` | Per-directed-pair TP/FP validation on vocabulary word lists (mirrors `Config::PairOverrides` and the source-restricted variant generation in `main.cpp`) |
| `sweep_he_params.py` | Sweep `EarlyDetectionMinChars` and `ConfidenceAtMaxChars` for Hebrew-target pairs to find the TP/FP Pareto frontier |
| `test_case_exclusion.py` | Validates the case-signal Hebrew exclusion: ALL-CAPS English/Russian + CamelCase → 100 % of model FPs eliminated, 0 real Hebrew TPs blocked |
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

Global defaults (apply to any pair not listed in `PairOverrides`):

| Parameter | Default | Description |
|---|---|---|
| `EarlyDetectionMinChars` | **3** (runtime) | Global fast-path gate — set at startup by `SyncGlobalEarlyOut()` to the smallest per-pair value. Default in code is 4; runtime minimum is 3 (en/ru→he pairs). Synced on launch AND when the flyout applies settings |
| `FullConfidenceChars` | 15 | Chars at which the confidence floor kicks in |
| `ConfidenceAtMinChars` | 0.99 | Required confidence at *EarlyDetectionMinChars* |
| `ConfidenceAtMaxChars` | 0.70 | Confidence floor at *FullConfidenceChars* and beyond — **live-adjusted** per pair by the calibration controller |
| `ConsecutiveAgreementCount` | 2 | Consecutive keystrokes that must agree before switching |
| `BorderlineZoneFactor` | 0.85 | Drop-one boosting fires in `[threshold × factor, threshold]` |
| `MinTop1Top2Margin` | 0.05 | Required gap between top-1 and top-2 softmax probabilities — **live-adjusted** per pair by the calibration controller |
| `MinKnownCharsForInference` | 2 | Minimum chars in the model vocabulary; below this the inference call is skipped (`skip_low_known_chars`) |
| `DisableAutoSwitchFromEnglishInFileDialogs` | true | Block en→he/ru auto-switch when a file-open/save dialog is active |
| `EnableTypoResilience` | true | Master toggle for consecutive-agreement and drop-one boosting |
| `EnableCaseBasedHeExclusion` | **true** | Exclude Hebrew when the cached word has uppercase-intent characters (ALL-CAPS / CamelCase) |
| `CaseExclusionMinCaps` | **2** | Minimum number of Shift/CapsLock alpha chars before "he" is excluded (sentence-initial caps do **not** count) |

Per-pair overrides (`Config::PairOverrides`) — factory / baseline values, validated by offline sweep on 200-word vocabulary samples. `ConfidenceAtMaxChars` and `MinTop1Top2Margin` are the live targets of the adaptive calibration controller (the factory values are stored as baselines; the controller applies a bounded delta on top):

| Pair | EarlyMin | FullConf | ConfAtMin | ConfAtMax ¹ | Agreement | Borderline | Margin ¹ | TP rate | FP rate |
|---|---|---|---|---|---|---|---|---|---|
| en→ru | 4 | 15 | 0.99 | 0.70 | 2 | 0.85 | 0.05 | 96.5 % | 0 % |
| ru→en | 4 | 15 | 0.99 | 0.70 | 2 | 0.85 | 0.05 | 92.0 % | 0 % |
| **en→he** | **3** | 15 | 0.99 | **0.60** | 2 | 0.88 | 0.10 | **84.0 %** | <1 %* |
| he→en | 4 | 15 | 0.99 | 0.70 | 2 | 0.85 | 0.05 | 92.5 % | 0 % |
| **ru→he** | **3** | 15 | 0.99 | **0.60** | 2 | 0.88 | 0.10 | **84.0 %** | <2 %* |
| he→ru | 4 | 15 | 0.99 | 0.70 | 2 | 0.80 | 0.05 | 96.5 % | 0 % |

> ¹ Factory (baseline) values. At runtime the calibration controller can raise `ConfAtMax` by up to +0.10 and `Margin` by up to +0.08 (too many FPs), or lower them by up to −0.05 / −0.01 (too many missed detections). Calibrated deltas are persisted in `%APPDATA%\KeyboardSwitcher\user_prefs.json`.
>
> \* Harness FP without case-exclusion guard. The C++ app additionally
> suppresses these via `EnableCaseBasedHeExclusion` and `HasLeakedLayoutChars`,
> so real-world FP is lower.
>
> `en→he` and `ru→he` use `EarlyMin=3` (vs. 4 for other pairs): sweep showed +4–7 pp TP
> gain. `ConfAtMax=0.60` replaces the old conservative 0.75 floor.
> Remaining missed detections (~16 %) are model-bound.
> All TP/FP numbers measured on the source-restricted harness (200-word vocabulary samples,
> `evaluate_transitions.py --sample 200`).

### Adaptive calibration internals

The controller lives entirely in `FeedbackLogger.cpp` and runs on the main hook thread.
No UI is involved; the only observable effect is the updated thresholds and a `CALIB:` line
in the debug log when an adaptation step fires.

| Signal | Source | Effect on EWMA |
|---|---|---|
| **FalsePositive** | Esc-undo, manual override within 10 s of correction | `ewmaFpRate` ← α·1 + (1−α)·ewmaFpRate |
| **TruePositive** | No rejection within 10 s (`MaybeConfirmTP`) | both rates ← α·0 + (1−α)·rate |
| **FalseNegative** | Unprompted manual switch (no recent correction) | `ewmaFnRate` ← α·1 + (1−α)·ewmaFnRate |

Adaptation fires when `batchEvents ≥ 5` and `|ewmaFpRate − ewmaFnRate| > 0.15`:

- **Tighten** (pressure > 0): `ConfAtMax += 0.01`, `Margin += 0.005`; delta clamped to `[0, +0.10]` / `[0, +0.08]`
- **Loosen** (pressure < 0): `ConfAtMax −= 0.01`, `Margin −= 0.005`; delta clamped to `[−0.05, 0]` / `[−0.01, 0]`
- Absolute limits applied after base+delta: conf ∈ [0.50, 0.995], margin ∈ [0.005, 0.25]
- When the tighten ceiling is reached, `ewmaFpRate` is damped by ×0.70 to prevent permanent lock-up
- State persisted in `user_prefs.json`; invalidated (reset to factory) on application version change

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
   - Reset learned exceptions & calibration (clears per-word exceptions, learned overrides, and adaptive calibration deltas — factory thresholds restored)
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
| `SENDING(async):` | Async correction worker dispatched — backspace count + target text |
| `CAPITALIZE[N]:` | Character at position N uppercased in corrected text |
| `EXCL:` | Language excluded from candidates (e.g. Hebrew excluded by case signal) |
| `SKIP: skip_low_alpha` | Not enough letter characters yet |
| `SKIP: skip_url_or_path` | URL or file-path pattern detected |
| `SKIP: skip_low_known_chars` | Fewer than *MinKnownCharsForInference* chars in vocabulary |
| `SKIP: skip_file_dialog_en_protection` | En→He/Ru blocked in a file-open/save dialog |
| `REJECT: leaked chars` | Layout conversion would produce unmapped characters |
| `GUARD-STATS:` | Aggregate counter dump (every 60 s while debug log is on) |
| `UNDO:` | Esc-undo reverting a correction |
| `MANUAL-SWITCH:` | User switched layout manually |
| `RESTORE-LANG:` | Per-window saved language restored on focus change |
| `CALIB: tp_confirmed` | Accepted auto-correction counted as true positive; pair EWMA updated |
| `CALIB: fn_manual_switch` | Unprompted manual layout switch counted as missed detection; pair EWMA updated |

### Persistent user data

All learned state is stored in `%APPDATA%\KeyboardSwitcher\`:

| File | Contents |
|---|---|
| `user_prefs.json` | Per-word exceptions, learned correction overrides, adaptive calibration deltas (EWMA state + current delta per pair), logging preference |
| `feedback.jsonl` | Timestamped feedback event log (only written when _Feedback collection_ is enabled) |

The calibration block in `user_prefs.json` looks like:
```json
"calibration": {
  "en>ru": { "ewma_fp": 0.08, "ewma_fn": 0.0, "delta_conf_max": 0.01,
             "delta_margin": 0.005, "base_conf_max": 0.70, "base_margin": 0.05,
             "batch_events": 2 }
}
```
This file is safe to delete — the app recreates it with factory defaults on next launch.

## Contributing

Pull requests are welcome!

## License

This project is licensed under the MIT License — Copyright © 2025-2026 Alpha-Numerical.
