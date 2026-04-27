# Voice Transcriber for EN-KR

A macOS menu bar voice-to-text app powered by MLX. Hold or tap a hotkey, speak, and the transcription is pasted into whatever window has focus.

Originally forked from [borinomi/mlx-whisper](https://github.com/borinomi/mlx-whisper); reworked for reliability (audio device hot-swap, silence guards, osascript paste, profile-based language/model bundling) and packaged as a standalone `.app` so macOS permissions attach to the app itself rather than to whatever terminal launched it.

Two transcription engines are bundled and selectable from the menu bar via "profiles":

- **한국어 (Whisper Turbo)** — `mlx-community/whisper-large-v3-turbo`, multilingual, used for Korean
- **English (Parakeet)** — `mlx-community/parakeet-tdt-0.6b-v2`, English-only, much less prone to silence hallucinations than Whisper

Picking a profile sets both the model and the language at once so the two can't get out of sync.

---

## Prerequisites

- macOS on Apple Silicon (M1/M2/M3/M4)
- [Homebrew](https://brew.sh)
- [uv](https://docs.astral.sh/uv/) (`brew install uv`)
- Python **3.12** (3.14 is too new for `rumps` / `pyobjc` / `pyaudio`)

System libraries:

```bash
brew install portaudio ffmpeg
```

`ffmpeg` is required by `mlx-whisper` to decode the captured WAV.

## Install

```bash
git clone https://github.com/noos/voice-transcriber-for-en-kr
cd voice-transcriber-for-en-kr
uv venv --python 3.12
uv pip install -r requirements.txt
```

### Optional: build a standalone `.app`

If you'd rather have macOS permissions attached to the app itself (instead of to your terminal), build a `.app` bundle with py2app and drop it into `/Applications`:

```bash
uv pip install py2app
rm -rf build dist
uv run python setup.py py2app          # ~5 GB output, takes a few minutes
mv "dist/Voice Transcriber for EN-KR.app" /Applications/
xattr -cr "/Applications/Voice Transcriber for EN-KR.app"
```

Then launch from Spotlight or `/Applications`. macOS will prompt for Microphone the first time; add the bundle to `System Settings → Privacy & Security → {Accessibility, Input Monitoring}` manually.

For faster iteration on the bundle config, use `uv run python setup.py py2app -A` (alias mode — symlinks back to your venv, no multi-GB copy).

## Run

```bash
uv run python -u app.py
```

The 🎤 icon appears in the macOS menu bar. The terminal stays attached for log output (`-u` keeps it unbuffered). On first launch the active model is downloaded from HuggingFace (~1.5 GB for Whisper Turbo, ~2.5 GB for Parakeet) — only fetched the first time you select that profile.

### macOS permissions

The app needs three permissions:

- **Microphone** — for `pyaudio` to capture audio
- **Accessibility** — for `pynput` to listen to the global hotkey and for `pyautogui` to send the paste keystroke
- **Input Monitoring** — same reason

Grant them to the **terminal app you launch the script from** (Ghostty, iTerm, Terminal.app, VS Code, etc.) — not to the Python binary itself. The uv-managed Python binary at `~/.local/share/uv/python/cpython-3.12.*/bin/python3.12` is unsigned and macOS will not let you add it to Accessibility/Input Monitoring at all (it shows up greyed out in the file picker). The child Python process inherits the permissions of its parent terminal, which is enough.

Add the terminal under `System Settings → Privacy & Security → {Accessibility, Input Monitoring, Microphone}`. If you switch terminals, you'll need to grant the new one as well.

## Usage

| Action | Shortcut |
|---|---|
| Start / stop recording | **Right Shift** |
| Cancel an in-progress recording (drop audio, no paste) | **Esc** |
| Cycle profile (Korean ↔ English) | **⌘⇧Space** |

Workflow: tap right-shift, speak, tap right-shift again. The transcript is copied to your clipboard and auto-pasted into the focused window. The menu bar icon shows recording state plus a language badge: `🎤KR`, `🔴EN`, `⏳KR`, etc.

The configuration file lives at `~/.config/voice-recorder/config.json`.

## Behavior notes

- **Cold-start warmup** — MLX kernels JIT-compile on first inference. The app runs a silent warmup transcription on startup so your first real recording isn't garbled. Real transcriptions wait for warmup completion (you'll see `[transcribe] waiting for warmup…` in the log if you record very early).
- **Switching profiles** triggers a fresh warmup of the new engine. Loaded models are cached, so toggling back and forth is fast.
- **Silence guards** — recordings shorter than 0.5 s, or with RMS / peak amplitude below thresholds, are dropped before being sent to the model. This sharply reduces Whisper's tendency to hallucinate `"v"`, `"you"`, `"thanks for watching"`, etc. on near-silent input. A small known-hallucination output filter is also applied as a backstop.
- **Mic hot-swap** — if PortAudio fails to open the input device (e.g., AirPods went to sleep mid-session, sample-rate mismatch), the app recreates `PyAudio` to clear stale CoreAudio state and retries with the device's native sample rate. The WAV header records the actual sample rate so `ffmpeg` resamples to 16 kHz correctly.
- **Single MLX worker thread** — all engine calls (warmup, transcription) are serialized through one worker thread. Required because MLX GPU streams are thread-local and a model loaded on one thread cannot be invoked from another (`RuntimeError: There is no Stream(gpu, N) in current thread`).
