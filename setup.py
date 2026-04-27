"""py2app setup for Voice Transcriber for EN/KR.

Build (alias mode, fast — for iterating on Info.plist / permissions):
    uv run python setup.py py2app -A

Build (real bundle, slow + multi-GB):
    uv run python setup.py py2app
"""
from setuptools import setup

APP = ["app.py"]

OPTIONS = {
    "argv_emulation": False,
    "plist": {
        "CFBundleIdentifier": "com.noos.voicetranscriberenkr",
        "CFBundleName": "Voice Transcriber for EN-KR",
        "CFBundleDisplayName": "Voice Transcriber for EN-KR",
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        "LSUIElement": True,
        "LSMinimumSystemVersion": "12.0",
        "NSMicrophoneUsageDescription":
            "Voice Transcriber needs microphone access to transcribe your speech.",
        "NSAppleEventsUsageDescription":
            "Voice Transcriber sends Cmd+V via System Events to paste transcribed text.",
        "NSAppleScriptEnabled": True,
    },
    "packages": [
        "mlx_whisper",
        "parakeet_mlx",
        "rumps",
        "pynput",
        "pyaudio",
        "mlx",
        "numpy",
        "pyperclip",
    ],
    "includes": [
        "queue",
        "subprocess",
        "tempfile",
        "wave",
        "json",
        "threading",
        "concurrent.futures",
    ],
    "excludes": [
        "tkinter",
        "pytest",
        "IPython",
        "jupyter",
        "notebook",
        "matplotlib",
        "pandas",
    ],
}

setup(
    app=APP,
    name="Voice Transcriber for EN-KR",
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
