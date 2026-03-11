# Keyboard Switcher

Automatically detect and switch the language (_En<->He<->Ru_) for **Windows**.

## Table of Contents
- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [Python Version](#python-version)
- [C++ Version](#c-version)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Description
The main idea was to resolve the annoying situation, when you are working with different
languages (in our case it is russian, hebrew and english) and all the time have to switch between
them. Like for example when you are switching between the browser (english layout for the url) 
and text chat (hebrew or russian layout)<br>

To get the result we have collected a large dataset, based on english, hebrew and russian dictionaries.<br>
We have trained a DL model with a relatively simple LSTM architecture to detect the 
language properly.<br>

Pytorch model trained on GPU but can be used on CPU without any problem<br>
The accuracy is above 99% on test set.

The program is operated as a background task, that is initiated on mouse click.
It detects the input language (most probable) and corrects the input, getting into
the sleep mode until the next mouse click.

## Project Structure

```
KeyboardSwitcher/
├── python/                  # Original Python implementation + training
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
│   │   ├── Config.h
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
│   └── resources/
│       ├── resource.h
│       └── resource.rc
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
- **MSVC** (Visual Studio 2019/2022) or any C++17 compiler on Windows
- **ONNX Runtime** C++ SDK ([download](https://github.com/microsoft/onnxruntime/releases))

### Build

1. Download ONNX Runtime and extract it (e.g. to `cpp/onnxruntime/`):
   ```
   cpp/onnxruntime/
   ├── include/
   │   └── onnxruntime_cxx_api.h  (and other headers)
   └── lib/
       ├── onnxruntime.lib
       └── onnxruntime.dll
   ```

2. Configure and build:
   ```bash
   cd cpp
   cmake -B build -DONNXRUNTIME_ROOT=./onnxruntime
   cmake --build build --config Release
   ```

3. The built executable and all required files will be in `build/Release/`.

### Dependencies
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) — ONNX model inference
- [nlohmann/json](https://github.com/nlohmann/json) — JSON parsing (fetched automatically by CMake)
- Win32 API — keyboard/mouse hooks, system tray, keyboard layout switching

## Usage
Just run the program, make a mouse click, type and enjoy.<br>
**If there are some bugs, let me know and I will fix'em...**

## Contributing

Me, myself and I. :) <br>

Pull requests are welcomed!

## License
This project is licensed under the MIT License.
