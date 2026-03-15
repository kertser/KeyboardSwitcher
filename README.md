# Keyboard Switcher v1.2.3

Automatically detect and switch the keyboard language (**En ↔ He ↔ Ru**) on **Windows**.

[Download Latest Release](https://github.com/kertser/KeyboardSwitcher/releases/latest)

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Project Structure](#project-structure)
- [Python Version](#python-version)
- [C++ Version](#c-version)
- [Packaging / Installer](#packaging--installer)
- [Windows Defender / SmartScreen](#windows-defender--smartscreen)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Description
The main idea was to resolve the annoying situation when you are working with different
languages (in our case Russian, Hebrew and English) and constantly have to switch between
them — for example when switching between the browser (English layout for the URL)
and a text chat (Hebrew or Russian layout).

To get the result we collected a large dataset based on English, Hebrew and Russian dictionaries
and trained a deep-learning model with a relatively simple LSTM architecture to detect the
language properly.

The PyTorch model is trained on GPU but runs on CPU without any problem.
The accuracy is above 99% on the test set.

The C++ version uses the model exported to ONNX format for fast native inference
via ONNX Runtime.

## Features

| Feature | Description |
|---|---|
| **Automatic language detection** | Detects the intended language and re-types the text in the correct layout |
| **Adaptive confidence curve** | Short input requires near-certain confidence; longer input lowers the bar — reduces both false positives and false negatives |
| **Typo resilience** | Two-tier protection against single-character typos: *consecutive-agreement* (requires 2+ keystrokes to agree on a language before switching) and *drop-one boosting* (if confidence is borderline, tries removing each character to recover from a typo) |
| **Per-window language memory** | Remembers the language for each window and restores it on focus change |
| **Manual switch detection** | If the user switches the layout manually (e.g. Alt+Shift), detection is paused until the next mouse click |
| **Capitalization preservation** | If the first letter was typed with Shift, the corrected text keeps the capital letter |
| **Dynamic tray tooltip** | Hovering the tray icon shows the current language (e.g. "Keyboard Switcher — English") |
| **System tray menu** | Right-click to enable/disable the switcher, toggle window-state saving, or exit |
| **Alt+Tab awareness** | Restores the per-window language after Alt+Tab switching |
| **Edge-case filtering** | Skips detection for URLs, file paths, and mostly non-alphabetic input |
| **Input collision guard** | Blocks real keystrokes during the retyping phase to prevent garbled output |

## Project Structure

```
KeyboardSwitcher/
├── python/                  # Original Python implementation + model training
│   ├── main.py
│   ├── config.py
│   ├── Languages.py
│   ├── Languages_torch.py
│   ├── convert_to_onnx.py
│   ├── LangModel.ipynb
│   ├── requirements.txt
│   ├── dictionary.pkl
│   ├── lang_model.pth
│   └── vocabulary/
├── cpp/                     # C++ (Win32) implementation
│   ├── CMakeLists.txt
│   ├── lang_model.onnx
│   ├── dictionary.json
│   ├── keyboard.ico
│   ├── include/
│   │   ├── Config.h         # Version, adaptive curve parameters, language maps
│   │   ├── Languages.h
│   │   ├── InputCache.h
│   │   ├── WindowTracker.h
│   │   └── TrayIcon.h
│   ├── src/
│   │   ├── main.cpp
│   │   ├── Config.cpp
│   │   ├── Languages.cpp
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

## Python Version

```bash
cd python
pip install -r requirements.txt
python main.py
```

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

### Full example (build + package):
```bash
cd cpp
cmake -B cmake-build-release -DCMAKE_BUILD_TYPE=Release -G "MinGW Makefiles"
cmake --build cmake-build-release --config Release
cd cmake-build-release
cpack -G ZIP        # portable archive
cpack -G NSIS       # installer (requires NSIS)
```

## Usage
1. Run the program — it appears as a tray icon.
2. Click on a text area and start typing.
3. The switcher detects the intended language and corrects the input automatically.
4. Right-click the tray icon to enable/disable the switcher or toggle per-window memory.
5. Hover the tray icon to see the current keyboard layout.

## Contributing

Pull requests are welcome!

## License
This project is licensed under the MIT License — Copyright © 2025-2026 Alpha-Numerical.
