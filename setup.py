"""py2app setup for Voice Transcriber for EN/KR.

Build (alias mode, fast — for iterating on Info.plist / permissions):
    uv run python setup.py py2app -A

Build (real bundle, slow + multi-GB):
    uv run python setup.py py2app
"""
import sys
from pathlib import Path

# py2app's modulegraph recurses through torch/scipy/numba's import tree and
# blows the default 1000-frame stack. Bumping fixes "RecursionError: maximum
# recursion depth exceeded" during the full bundle build.
sys.setrecursionlimit(5000)

# `mlx` is shipped as a PEP 420 namespace package (no __init__.py). py2app's
# modulegraph uses imp.find_module() which can't discover namespace packages,
# so it fails with "No module named 'mlx'" during the full build. Writing an
# empty __init__.py converts it to a regular package without changing runtime
# behavior (mlx.core etc. continue to work identically).
try:
    import mlx as _mlx
    _mlx_init = Path(_mlx.__path__[0]) / "__init__.py"
    if not _mlx_init.exists():
        _mlx_init.touch()
except ImportError:
    pass

# uv-managed Python statically links zlib into the interpreter, so `zlib`
# has no __file__ attribute. py2app blindly does `copy_file(zlib.__file__, ...)`
# and crashes with AttributeError. Point zlib at an empty stub file — py2app
# copies it into the bundle but it's never actually loaded (zlib is already
# in the bundled python binary).
import zlib as _zlib
if not hasattr(_zlib, "__file__"):
    _stub = Path(__file__).parent / "build" / "_zlib_stub"
    _stub.parent.mkdir(parents=True, exist_ok=True)
    _stub.touch()
    _zlib.__file__ = str(_stub)

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
