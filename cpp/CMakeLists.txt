cmake_minimum_required(VERSION 3.21)
project(KeyboardSwitcher VERSION 1.3.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── nlohmann/json via FetchContent ──────────────────────────
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(json)

# ── ONNX Runtime ────────────────────────────────────────────
# Set ONNX Runtime root path. User can override with -DONNXRUNTIME_ROOT=...
if(NOT DEFINED ONNXRUNTIME_ROOT)
    if(DEFINED ENV{ONNXRUNTIME_ROOT})
        set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
    else()
        set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime" CACHE PATH
            "Path to ONNX Runtime SDK (contains include/ and lib/)")
    endif()
endif()

message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_ROOT}")

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
    NO_DEFAULT_PATH
)
# On MinGW, find_library won't find .lib files by default, so add .lib suffix
if(MINGW)
    list(APPEND CMAKE_FIND_LIBRARY_SUFFIXES ".lib")
endif()
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
)
find_file(ONNXRUNTIME_DLL
    NAMES onnxruntime.dll
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "Could not find ONNX Runtime.\n"
        "Please set ONNXRUNTIME_ROOT to the ONNX Runtime SDK directory.\n"
        "Expected layout:\n"
        "  <ONNXRUNTIME_ROOT>/include/onnxruntime_cxx_api.h\n"
        "  <ONNXRUNTIME_ROOT>/lib/onnxruntime.lib\n"
        "  <ONNXRUNTIME_ROOT>/lib/onnxruntime.dll\n"
        "\n"
        "Download from: https://github.com/microsoft/onnxruntime/releases"
    )
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime lib:     ${ONNXRUNTIME_LIBRARY}")

# ── Sources ─────────────────────────────────────────────────
set(SOURCES
    src/main.cpp
    src/Config.cpp
    src/Languages.cpp
    src/InputCache.cpp
    src/WindowTracker.cpp
    src/TrayIcon.cpp
    src/FeedbackLogger.cpp
)

set(HEADERS
    include/Config.h
    include/Languages.h
    include/InputCache.h
    include/WindowTracker.h
    include/TrayIcon.h
    include/FeedbackLogger.h
    resources/resource.h
)

# ── Generate resource file with absolute icon path ──────────
set(KEYBOARD_ICO_PATH "${CMAKE_CURRENT_SOURCE_DIR}/keyboard.ico")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/resources/resource.rc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/resource.rc"
    @ONLY
)

# ── Executable (WIN32 = no console window) ──────────────────
add_executable(${PROJECT_NAME} WIN32
    ${SOURCES}
    ${HEADERS}
    "${CMAKE_CURRENT_BINARY_DIR}/resource.rc"
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
# SYSTEM suppresses SAL-annotation redefinition warnings from the ONNX headers
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${ONNXRUNTIME_LIBRARY}
    nlohmann_json::nlohmann_json
    user32
    shell32
    comctl32
    dwmapi
    winhttp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    UNICODE _UNICODE
    APP_VERSION_STRING=L"${PROJECT_VERSION}"
)

# ── Static-link MinGW runtimes so no libstdc++/libgcc DLLs needed ──
if(MINGW)
    target_link_options(${PROJECT_NAME} PRIVATE
        -static-libgcc
        -static-libstdc++
        -static
    )
endif()

# ── Copy data files to build output ─────────────────────────
set(DATA_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/lang_model.onnx"
    "${CMAKE_CURRENT_SOURCE_DIR}/dictionary.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/keyboard.ico"
)

foreach(DATA_FILE ${DATA_FILES})
    get_filename_component(FNAME "${DATA_FILE}" NAME)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DATA_FILE}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/${FNAME}"
        COMMENT "Copying ${FNAME} to output directory"
    )
endforeach()

# Copy ONNX Runtime DLLs to output
if(ONNXRUNTIME_DLL)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DLL}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/onnxruntime.dll"
        COMMENT "Copying onnxruntime.dll to output directory"
    )
endif()

# Also copy onnxruntime_providers_shared.dll if it exists
find_file(ONNXRUNTIME_PROVIDERS_DLL
    NAMES onnxruntime_providers_shared.dll
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
)
if(ONNXRUNTIME_PROVIDERS_DLL)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_PROVIDERS_DLL}"
            "$<TARGET_FILE_DIR:${PROJECT_NAME}>/onnxruntime_providers_shared.dll"
        COMMENT "Copying onnxruntime_providers_shared.dll to output directory"
    )
endif()

# ── Install / Export target ─────────────────────────────────
# Usage: cmake --install <build-dir> --prefix <output-folder>
#   e.g. cmake --install cmake-build-release --prefix ./dist
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION .)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/lang_model.onnx"
    "${CMAKE_CURRENT_SOURCE_DIR}/dictionary.json"
    "${CMAKE_CURRENT_SOURCE_DIR}/keyboard.ico"
    DESTINATION .
)
if(ONNXRUNTIME_DLL)
    install(FILES "${ONNXRUNTIME_DLL}" DESTINATION .)
endif()
if(ONNXRUNTIME_PROVIDERS_DLL)
    install(FILES "${ONNXRUNTIME_PROVIDERS_DLL}" DESTINATION .)
endif()

# ── CPack / Installer ───────────────────────────────────────
# Generates (version from PROJECT_VERSION):
#   NSIS  → KeyboardSwitcher-<version>-win64.exe  (installer wizard)
#   ZIP   → KeyboardSwitcher-<version>-win64.zip  (portable archive)
#
# Usage (from the release build directory):
#   cpack -G NSIS   — requires NSIS installed (https://nsis.sourceforge.io)
#   cpack -G ZIP    — no extra tools needed
set(CPACK_PACKAGE_NAME "KeyboardSwitcher")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VENDOR "Alpha-Numerical")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "Automatic keyboard language switcher (En / He / Ru)")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "KeyboardSwitcher")

# License shown in the installer wizard
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")

# NSIS-specific settings
set(CPACK_NSIS_DISPLAY_NAME "Keyboard Switcher ${PROJECT_VERSION}")
set(CPACK_NSIS_PACKAGE_NAME "Keyboard Switcher")
set(CPACK_NSIS_INSTALLED_ICON_NAME "KeyboardSwitcher.exe")
set(CPACK_NSIS_MUI_ICON "${CMAKE_CURRENT_SOURCE_DIR}/keyboard.ico")
set(CPACK_NSIS_MUI_UNIICON "${CMAKE_CURRENT_SOURCE_DIR}/keyboard.ico")
set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)

# Start-menu shortcut
set(CPACK_NSIS_CREATE_ICONS_EXTRA
    "CreateShortCut '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\Keyboard Switcher.lnk' '$INSTDIR\\\\KeyboardSwitcher.exe'")
set(CPACK_NSIS_DELETE_ICONS_EXTRA
    "Delete '$SMPROGRAMS\\\\$START_MENU\\\\Keyboard Switcher.lnk'")

# Default generators: NSIS installer + portable ZIP
set(CPACK_GENERATOR "NSIS;ZIP")

include(CPack)
