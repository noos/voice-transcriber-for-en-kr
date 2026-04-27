# mlx-whisper

A macOS menu bar voice-to-text app powered by MLX. Hold or tap a hotkey, speak, and the transcription is pasted into whatever window has focus.

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
git clone https://github.com/borinomi/mlx-whisper
cd mlx-whisper
uv venv --python 3.12
uv pip install -r requirements.txt
```

## Run

```bash
uv run python -u app.py
```

The 🎤 icon appears in the macOS menu bar. The terminal stays attached for log output (`-u` keeps it unbuffered). On first launch the active model is downloaded from HuggingFace (~1.5 GB for Whisper Turbo, ~2.5 GB for Parakeet) — only fetched the first time you select that profile.

### macOS permissions

The first time you run the app, macOS will prompt for:

- **Microphone** — for `pyaudio` to capture audio
- **Accessibility** — for `pynput` to listen to the global hotkey
- **Input Monitoring** — same reason

Permissions are granted to the binary that's *actually* running, which is the uv-managed Python at `~/.local/share/uv/python/cpython-3.12.*/bin/python3.12`, not your terminal app. If macOS doesn't auto-prompt, add it manually under `System Settings → Privacy & Security → {Accessibility, Input Monitoring}`.

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
