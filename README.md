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

### Recommended: build a standalone `.app`

So macOS permissions attach to *the app itself* instead of to whatever terminal launched the script. There's a one-shot installer:

```bash
./install.sh
```

That script verifies prerequisites (Apple Silicon, Homebrew, `uv`, `portaudio`, `ffmpeg`), creates the venv if needed, installs requirements + py2app, builds the bundle, copies it to `/Applications`, strips the macOS quarantine flag, resets stale TCC grants, and launches the app. Re-running it cleanly rebuilds and reinstalls — safe to run any time you change `app.py`.

For faster iteration on the bundle config, run `uv run python setup.py py2app -A` directly (alias mode — symlinks back to your venv, no copy). Alias mode is ideal for `Info.plist` debugging.

#### Granting bundle permissions

The app needs four TCC permissions:

1. **Microphone** — auto-prompted the first time you record
2. **Accessibility** — `System Settings → Privacy & Security → Accessibility` → `+` → add `/Applications/Voice Transcriber for EN-KR.app`
3. **Input Monitoring** — same panel, same procedure
4. **Automation** (`System Events`) — auto-prompted the first time the app pastes; appears as *"Voice Transcriber wants access to control System Events"*

After granting Accessibility/Input Monitoring, **quit and re-launch** the bundle so `pynput` picks up the new grants.

#### Bundling gotchas (already handled in `setup.py`)

These are baked into [setup.py](setup.py); listed for reference if you fork:

- `sys.setrecursionlimit(5000)` — py2app's modulegraph blows the default stack on torch/scipy/numba's import tree.
- An empty `mlx/__init__.py` is created at build time — `mlx` is a PEP 420 namespace package and py2app's `imp.find_module` can't discover it otherwise.
- A stub file is pointed to as `zlib.__file__` — uv-managed Python statically links zlib, so it has no `__file__` and py2app crashes when it tries to copy "the zlib library".
- [app.py](app.py) prepends `/opt/homebrew/bin` and `/usr/local/bin` to `PATH` at import time so the bundled subprocess can find `ffmpeg` and `osascript`.
- [app.py](app.py) redirects `sys.stdout`/`sys.stderr` to `~/Library/Logs/voice-transcriber.log` (UTF-8) so the otherwise-discarded `print()` output survives the bundle launcher. **This log is your debug trail** — if anything misbehaves in the bundle, `cat ~/Library/Logs/voice-transcriber.log` is the first place to look.

#### Re-builds invalidate permissions

Each `setup.py py2app` run produces a new ad-hoc signature, and macOS TCC keys grants on bundle ID **plus** signature hash. Every rebuild therefore orphans your previous grants. The cleanest reset is:

```bash
tccutil reset All com.noos.voicetranscriberenkr
```

…then re-add the bundle to the three permission panels above.

#### Still required on the host

The bundle does **not** include `ffmpeg` or `portaudio` — both are loaded from system locations. Anyone who installs the `.app` still needs Homebrew + `brew install portaudio ffmpeg` on their machine. Bundling those is a future improvement.

### Source-only run (no `.app`)

## Run

```bash
uv run python -u app.py
```

The 🎤 icon appears in the macOS menu bar. The terminal stays attached for log output (`-u` keeps it unbuffered). On first launch the active model is downloaded from HuggingFace (~1.5 GB for Whisper Turbo, ~2.5 GB for Parakeet) — only fetched the first time you select that profile.

### macOS permissions when running from source

The script needs Microphone, Accessibility, Input Monitoring (and Automation, for the `osascript` paste). Grant them to the **terminal app you launch the script from** (Ghostty, iTerm, Terminal.app, VS Code, etc.) — not to the Python binary itself. The uv-managed Python binary at `~/.local/share/uv/python/cpython-3.12.*/bin/python3.12` is unsigned and macOS won't let you add it to Accessibility/Input Monitoring (greyed out in the file picker). The child Python process inherits its parent terminal's grants.

If you switch terminals, you'll need to re-grant the new one. **This is exactly the over-broad-permission situation the bundled `.app` (above) avoids** — recommend installing the bundle for daily use.

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
