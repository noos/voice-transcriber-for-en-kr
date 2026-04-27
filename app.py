import os
import sys
from pathlib import Path

# Bundled .app inherits a stripped PATH (just /usr/bin:/bin) and can't find
# Homebrew binaries like ffmpeg. mlx_whisper and parakeet_mlx both shell out
# to ffmpeg to decode the captured WAV; without this prepend we get
# "FFmpeg is not installed or not in your PATH" at transcribe time.
os.environ["PATH"] = ":".join([
    "/opt/homebrew/bin",  # Apple Silicon brew
    "/usr/local/bin",     # Intel brew (harmless on AS)
    os.environ.get("PATH", "/usr/bin:/bin"),
])

# Bundled .app drops stdout/stderr to /dev/null. Redirect to a log file so
# transcribe/warmup/silence-guard prints survive the launcher.
_LOG_PATH = Path.home() / "Library" / "Logs" / "voice-transcriber.log"
try:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(_LOG_PATH, "a", buffering=1, encoding="utf-8", errors="replace")
    sys.stdout = _log_file
    sys.stderr = _log_file
    print(f"\n=== app start pid={os.getpid()} ===", flush=True)
except Exception:
    pass

import rumps
import pyaudio
import threading
import tempfile
import wave
import json
import queue
import subprocess
import traceback
import time
from concurrent.futures import ThreadPoolExecutor

import mlx_whisper
import numpy as np
import pyperclip
from pynput import keyboard

LANG_BADGE = {"ko": "KR", "en": "EN"}

MODELS = [
    {
        "key": "whisper-turbo",
        "label": "Whisper Large V3 Turbo (다국어)",
        "engine": "whisper",
        "repo": "mlx-community/whisper-large-v3-turbo",
        "english_only": False,
    },
    {
        "key": "parakeet",
        "label": "Parakeet TDT 0.6B (영어)",
        "engine": "parakeet",
        "repo": "mlx-community/parakeet-tdt-0.6b-v2",
        "english_only": True,
    },
]
MODELS_BY_KEY = {m["key"]: m for m in MODELS}

# Profiles bundle a language and a model so the user picks one thing in the UI
# and can't accidentally pair (e.g.) Korean with Parakeet.
PROFILES = [
    {"key": "ko-whisper",  "label": "한국어 (Whisper Turbo)", "language": "ko", "model_key": "whisper-turbo"},
    {"key": "en-parakeet", "label": "English (Parakeet)",      "language": "en", "model_key": "parakeet"},
]
PROFILES_BY_KEY = {p["key"]: p for p in PROFILES}

class VoiceRecorderApp(rumps.App):
    def __init__(self):
        super().__init__("🎤", quit_button=None)

        # 설정 로드
        self.config_path = Path.home() / ".config" / "voice-recorder" / "config.json"
        self.load_config()
        self.title = f"🎤{LANG_BADGE.get(self.config['language'], self.config['language'].upper())}"
        # 오디오 설정
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000  # preferred; falls back to device default if unsupported
        self.CHUNK = 1024
        self.actual_rate = self.RATE  # rate the current stream is actually open at

        self.is_recording = False
        self.frames = []
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.record_thread = None

        # UI 작업 큐(메인 루프에서만 UI 변경/notification 실행)
        self._uiq: "queue.Queue[callable]" = queue.Queue()

        # 핫키 이벤트 (pynput 스레드 -> event set -> 메인 타이머 처리)
        self._toggle_event = threading.Event()
        self._lang_event = threading.Event()
        self._cancel_event = threading.Event()

        # 타이머: UI 큐 drain + 이벤트 처리
        self._ui_timer = rumps.Timer(self._drain_mainloop, 0.05)
        self._ui_timer.start()

        # 단축키 리스너
        self.hotkey_listener = None
        self.setup_hotkey()

        # 메뉴 구성
        self.build_menu()

        # 엔진/모델 상태 — 모든 MLX 작업은 단일 워커 스레드에서 실행
        # (MLX의 GPU stream은 thread-local이므로 cached parakeet model을
        # 로드한 스레드 외 다른 스레드에서 호출하면 RuntimeError 발생)
        self._parakeet_models = {}  # repo -> loaded model (cached)
        self._engine_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engine")

        # 모델 워밍업 (cold-start 방지)
        self._warmup_done = threading.Event()
        self._engine_executor.submit(self._warmup_model)

    def _active_model(self):
        return MODELS_BY_KEY[self.config["model_key"]]

    def _active_profile(self):
        """Find the profile matching the current (language, model_key) pair, else fall back to PROFILES[0]."""
        lang = self.config.get("language")
        mk = self.config.get("model_key")
        for p in PROFILES:
            if p["language"] == lang and p["model_key"] == mk:
                return p
        return PROFILES[0]

    def _refresh_title(self):
        """타이틀 배지 갱신 (현재 언어 + 녹음/전사 상태 반영)"""
        badge = LANG_BADGE.get(self.config["language"], self.config["language"].upper())
        cur = str(self.title)
        if cur.startswith("🔴"):
            self.title = f"🔴{badge}"
        elif cur.startswith("⏳"):
            self.title = f"⏳{badge}"
        else:
            self.title = f"🎤{badge}"

    def _get_parakeet(self, repo):
        """Lazily load + cache a parakeet model. MUST be called from the engine thread."""
        if repo not in self._parakeet_models:
            import parakeet_mlx
            print(f"[parakeet] loading {repo}…", flush=True)
            self._parakeet_models[repo] = parakeet_mlx.from_pretrained(repo)
            print("[parakeet] loaded", flush=True)
        return self._parakeet_models[repo]

    def _warmup_model(self):
        """현재 엔진의 첫 transcribe 호출은 MLX 커널 JIT으로 인해 결과가 망가질 수 있음.
        저진폭 노이즈로 인코더+디코더 경로 모두 워밍업한다."""
        path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
            np.random.seed(0)
            samples = (np.random.randn(32000) * 800).astype(np.int16).tobytes()
            wf = wave.open(path, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples)
            wf.close()

            m = self._active_model()
            print(f"[warmup] {m['key']} starting…", flush=True)
            self._run_engine(m, path, language=self.config["language"])
            print(f"[warmup] {m['key']} done", flush=True)
        except Exception as e:
            print(f"[warmup] failed: {e}", flush=True)
        finally:
            self._warmup_done.set()
            if path:
                try:
                    os.unlink(path)
                except Exception:
                    pass

    def _run_engine(self, model_def, wav_path, language):
        """엔진 디스패치. 반환: 결과 텍스트(문자열)."""
        if model_def["engine"] == "whisper":
            lang = "en" if model_def["english_only"] else language
            result = mlx_whisper.transcribe(
                wav_path,
                path_or_hf_repo=model_def["repo"],
                language=lang,
                hallucination_silence_threshold=0.5,
            )
            return (result.get("text") or "").strip()
        if model_def["engine"] == "parakeet":
            model = self._get_parakeet(model_def["repo"])
            result = model.transcribe(wav_path)
            return (getattr(result, "text", "") or "").strip()
        raise ValueError(f"unknown engine: {model_def['engine']}")

    # ---------------------------
    # Config
    # ---------------------------
    def load_config(self):
        """설정 파일 로드"""
        default_config = {
            "lang_hotkey": "cmd+shift+space",
            "language": "ko",
            "model_key": "whisper-turbo",
        }

        cfg = {}
        try:
            if self.config_path.exists():
                with open(self.config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f) or {}
        except Exception:
            cfg = {}

        # 녹음 단축키는 right shift로 고정 (이전 설정 무시)
        cfg.pop("record_hotkey", None)
        cfg.pop("hotkey", None)
        # 구버전 호환: 'model' (repo 문자열) → 'model_key'
        if "model_key" not in cfg and "model" in cfg:
            for m in MODELS:
                if m["repo"] == cfg["model"]:
                    cfg["model_key"] = m["key"]
                    break
        cfg.pop("model", None)

        self.config = {**default_config, **cfg}
        self.config["record_hotkey"] = "rshift"

        if self.config.get("model_key") not in MODELS_BY_KEY:
            self.config["model_key"] = default_config["model_key"]
        if not self.config.get("lang_hotkey"):
            self.config["lang_hotkey"] = default_config["lang_hotkey"]

    def save_config(self):
        """설정 파일 저장"""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    # ---------------------------
    # UI helpers (main thread via queue)
    # ---------------------------
    def _ui(self, fn):
        """메인 루프에서 실행할 UI 작업 등록"""
        self._uiq.put(fn)

    def _notify(self, title, subtitle, message):
        """notification도 메인 루프에서 실행"""
        def _do():
            try:
                rumps.notification(title, subtitle, message)
            except Exception:
                # 알림 실패는 치명적이지 않음
                pass
        self._ui(_do)

    def _drain_mainloop(self, _):
        """메인 루프에서: 이벤트 처리 + UI 큐 실행"""
        # 1) 핫키 이벤트 처리
        if self._toggle_event.is_set():
            self._toggle_event.clear()
            self.toggle_recording(None)

        if self._lang_event.is_set():
            self._lang_event.clear()
            self.cycle_profile()

        if self._cancel_event.is_set():
            self._cancel_event.clear()
            self.cancel_recording()

        # 2) UI 큐 drain
        for _ in range(50):  # 한 tick에 과도 실행 방지
            try:
                fn = self._uiq.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception:
                # 디버그용: 여기서 죽으면 앱이 조용히 종료될 수 있어서 방어
                traceback.print_exc()

    # ---------------------------
    # Menu
    # ---------------------------
    def build_menu(self):
        """메뉴 구성"""
        self.menu.clear()

        record_hk = self.config.get("record_hotkey", "")
        lang_hk = self.config.get("lang_hotkey", "")
        active = self._active_profile()

        # 녹음 상태
        status = "🔴 녹음 중지" if self.is_recording else "녹음 시작"
        self.status_item = rumps.MenuItem(
            f"{status} ({self.format_hotkey(record_hk)})",
            callback=self.toggle_recording
        )
        self.menu.add(self.status_item)

        # 프로필 스위치 안내 (고정 단축키)
        self.menu.add(rumps.MenuItem(
            f"프로필 전환: {self.format_hotkey(lang_hk)}  (현재: {active['label']})",
            callback=None
        ))

        self.menu.add(rumps.separator)

        # 프로필 선택 (언어+모델 묶음)
        profile_menu = rumps.MenuItem("프로필")
        for p in PROFILES:
            item = rumps.MenuItem(
                f"{'✓ ' if active['key'] == p['key'] else '   '}{p['label']}",
                callback=lambda sender, k=p["key"]: self.set_profile(k),
            )
            profile_menu.add(item)
        self.menu.add(profile_menu)

        self.menu.add(rumps.separator)
        self.menu.add(rumps.MenuItem("종료", callback=self.quit_app))

    def format_hotkey(self, hotkey: str) -> str:
        """단축키를 보기 좋게 포맷"""
        if not hotkey:
            return "-"
        # Order matters: rshift/lshift before shift so they aren't partially replaced.
        replacements = [
            ("rshift", "Right ⇧"),
            ("lshift", "Left ⇧"),
            ("cmd", "⌘"), ("shift", "⇧"), ("alt", "⌥"),
            ("ctrl", "⌃"), ("space", "Space"), ("+", ""),
        ]
        result = hotkey.lower()
        for k, v in replacements:
            result = result.replace(k, v)
        return result

    # ---------------------------
    # Hotkey parsing/normalization
    # ---------------------------
    def _key_tokens(self, key):
        """Tokens contributed by a physical pynput key. A right-shift contributes
        both 'shift' and 'rshift' so that hotkeys configured with either token match."""
        if key == keyboard.Key.shift_l:
            return {"shift", "lshift"}
        if key == keyboard.Key.shift_r:
            return {"shift", "rshift"}
        if key == keyboard.Key.shift:
            return {"shift"}
        if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            return {"ctrl"}
        if key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
            return {"alt"}
        if key in (keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r):
            return {"cmd"}
        if key == keyboard.Key.space:
            return {"space"}
        if isinstance(key, keyboard.KeyCode) and key.char:
            return {"char:" + key.char.lower()}
        return set()

    _MOD_TOKENS = {"cmd", "shift", "alt", "ctrl", "space", "rshift", "lshift"}

    def parse_hotkey(self, hotkey: str):
        """단축키 문자열을 토큰 set으로 변환 (e.g. 'ctrl+shift+m' -> {'ctrl','shift','char:m'})"""
        out = set()
        for part in (hotkey or "").lower().split("+"):
            part = part.strip()
            if not part:
                continue
            if part in self._MOD_TOKENS:
                out.add(part)
            elif len(part) == 1:
                out.add("char:" + part)
        return out

    def setup_hotkey(self):
        """글로벌 단축키 설정 (녹음 토글 + 언어 전환)"""
        if self.hotkey_listener:
            self.hotkey_listener.stop()

        record_keys = self.parse_hotkey(self.config.get("record_hotkey", "ctrl+shift+m"))
        lang_keys = self.parse_hotkey(self.config.get("lang_hotkey", "cmd+shift+space"))

        physical = set()  # pynput keys currently held
        fired_record = False
        fired_lang = False

        def current_tokens():
            t = set()
            for k in physical:
                t |= self._key_tokens(k)
            return t

        def on_press(key):
            nonlocal fired_record, fired_lang
            # Esc cancels a recording in progress (drop frames, no transcription)
            if key == keyboard.Key.esc and self.is_recording:
                self._cancel_event.set()
                return
            physical.add(key)
            cur = current_tokens()
            if (not fired_record) and record_keys and record_keys.issubset(cur):
                fired_record = True
                self._toggle_event.set()
            if (not fired_lang) and lang_keys and lang_keys.issubset(cur):
                fired_lang = True
                self._lang_event.set()

        def on_release(key):
            nonlocal fired_record, fired_lang
            physical.discard(key)
            cur = current_tokens()
            if not record_keys.issubset(cur):
                fired_record = False
            if not lang_keys.issubset(cur):
                fired_lang = False

        self.hotkey_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.hotkey_listener.start()

    # ---------------------------
    # Settings actions
    # ---------------------------
    def set_profile(self, key: str):
        """프로필(언어+모델) 변경. 모델이 바뀌면 워밍업 다시 수행."""
        if key not in PROFILES_BY_KEY:
            return
        p = PROFILES_BY_KEY[key]
        if p["language"] == self.config.get("language") and p["model_key"] == self.config.get("model_key"):
            return
        model_changed = p["model_key"] != self.config.get("model_key")
        self.config["language"] = p["language"]
        self.config["model_key"] = p["model_key"]
        self.save_config()
        self._refresh_title()
        self.build_menu()
        self._notify("음성 인식", "", f"프로필: {p['label']}")
        if model_changed:
            self._warmup_done.clear()
            self._engine_executor.submit(self._warmup_model)

    def cycle_profile(self):
        """프로필 순환 전환 (cmd+shift+space)"""
        cur_key = self._active_profile()["key"]
        keys = [p["key"] for p in PROFILES]
        try:
            nxt = keys[(keys.index(cur_key) + 1) % len(keys)]
        except ValueError:
            nxt = keys[0]
        self.set_profile(nxt)
    # ---------------------------
    # Recording
    # ---------------------------
    def toggle_recording(self, sender):
        """녹음 토글"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """녹음 시작"""
        if self.is_recording:
            return

        self.is_recording = True
        self.frames = []

        # UI 업데이트는 메인루프에서
        self.title = f"🔴{LANG_BADGE.get(self.config['language'], self.config['language'].upper())}"
        self.build_menu()

        def _device_default_rate():
            try:
                info = self.audio.get_default_input_device_info()
                return int(info.get("defaultSampleRate") or self.RATE)
            except Exception:
                return self.RATE

        def _open_stream(rate):
            return self.audio.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=rate,
                input=True,
                frames_per_buffer=self.CHUNK
            ), rate

        # Try preferred 16k, then device default, recreating PyAudio if CoreAudio is stale.
        candidates = [self.RATE]
        dev_rate = _device_default_rate()
        if dev_rate != self.RATE:
            candidates.append(dev_rate)

        opened = None
        last_err = None
        for _ in range(2):  # second pass after recreating PyAudio
            for rate in candidates:
                try:
                    opened = _open_stream(rate)
                    break
                except Exception as e:
                    last_err = e
                    print(f"[audio] open rate={rate} failed: {e}", flush=True)
            if opened:
                break
            print("[audio] recreating PyAudio to clear stale CoreAudio state", flush=True)
            try:
                self.audio.terminate()
            except Exception:
                pass
            self.audio = pyaudio.PyAudio()
            # Re-query in case the default device changed
            candidates = [self.RATE]
            dev_rate = _device_default_rate()
            if dev_rate != self.RATE:
                candidates.append(dev_rate)

        if not opened:
            self.is_recording = False
            self.title = f"🎤{LANG_BADGE.get(self.config['language'], self.config['language'].upper())}"
            self.build_menu()
            print(f"[audio] giving up: {last_err}", flush=True)
            self._notify("오디오 오류", "", str(last_err)[:120])
            return

        self.stream, self.actual_rate = opened
        print(f"[audio] stream opened at {self.actual_rate} Hz", flush=True)

        def record():
            while self.is_recording:
                try:
                    data = self.stream.read(self.CHUNK, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception as e:
                    # 백그라운드 스레드 -> 메인루프 알림
                    self._notify("오디오 오류", "", str(e)[:120])
                    break

        self.record_thread = threading.Thread(target=record, daemon=True)
        self.record_thread.start()

    def _close_stream(self):
        """녹음 스트림 정리"""
        if self.record_thread:
            self.record_thread.join(timeout=1)
        if self.stream:
            try:
                self.stream.stop_stream()
            except Exception:
                pass
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def stop_recording(self):
        """녹음 중지 및 전사"""
        if not self.is_recording:
            return

        self.is_recording = False
        self.title = f"⏳{LANG_BADGE.get(self.config['language'], self.config['language'].upper())}"
        self.build_menu()

        self._close_stream()

        if not self.frames:
            self.title = f"🎤{LANG_BADGE.get(self.config['language'], self.config['language'].upper())}"
            self.build_menu()
            return

        frames_snapshot = self.frames[:]  # 전사 스레드에 안전하게 전달
        rate_snapshot = self.actual_rate
        self.frames = []

        threading.Thread(target=self.transcribe_and_paste, args=(frames_snapshot, rate_snapshot), daemon=True).start()

    def cancel_recording(self):
        """Esc로 녹음 취소: 프레임 폐기, 전사하지 않음"""
        if not self.is_recording:
            return
        self.is_recording = False
        self._close_stream()
        self.frames = []
        self.title = f"🎤{LANG_BADGE.get(self.config['language'], self.config['language'].upper())}"
        self.build_menu()
        print("[record] cancelled (Esc)", flush=True)

    # ---------------------------
    # Transcription
    # ---------------------------
    def transcribe_and_paste(self, frames_snapshot, rate):
        """전사 및 붙여넣기 (백그라운드)"""
        temp_path = None
        try:
            print(f"[transcribe] frames={len(frames_snapshot)} bytes={sum(len(f) for f in frames_snapshot)}", flush=True)
            # WAV 파일로 저장
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            wf = wave.open(temp_path, "wb")
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(self.FORMAT))
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames_snapshot))
            wf.close()
            m = self._active_model()
            print(f"[transcribe] wav={temp_path} rate={rate} lang={self.config['language']} model={m['key']}", flush=True)

            if not self._warmup_done.is_set():
                print("[transcribe] waiting for warmup…", flush=True)
                self._warmup_done.wait(timeout=120)

            # 무음/짧은 트리거 가드 (whisper 환각 방지)
            samples = np.frombuffer(b"".join(frames_snapshot), dtype=np.int16)
            duration = samples.size / float(rate) if rate else 0.0
            rms = float(np.sqrt(np.mean(samples.astype(np.float32) ** 2))) if samples.size else 0.0
            # Loudest 100ms — catches the case where most of the buffer is quiet but
            # a stray click/breath bumps RMS above the floor.
            win = max(1, int(0.1 * rate))
            peak_rms = 0.0
            if samples.size >= win:
                sq = samples.astype(np.float32) ** 2
                # cheap moving-average via cumulative sum
                cs = np.cumsum(sq)
                window_means = (cs[win - 1:] - np.concatenate(([0.0], cs[:-win]))) / win
                peak_rms = float(np.sqrt(window_means.max()))
            print(f"[transcribe] dur={duration:.2f}s rms={rms:.0f} peak100ms={peak_rms:.0f}", flush=True)
            if duration < 0.5:
                print("[transcribe] skipped (too short)", flush=True)
                return
            if rms < 120 or peak_rms < 350:
                print("[transcribe] skipped (silence)", flush=True)
                return

            # MLX 작업은 단일 엔진 스레드에서 수행 (thread-local stream 제약)
            text = self._engine_executor.submit(
                self._run_engine, m, temp_path, self.config["language"]
            ).result()
            print(f"[transcribe] result={text!r}", flush=True)

            # Output-side hallucination filter (last line of defense)
            HALLUCINATIONS = {
                "v", "you", ".", "thank you", "thank you.",
                "thanks for watching", "thanks for watching.",
                "thanks for watching!", "bye", "bye.",
            }
            if text.lower().strip(" .!?,") in HALLUCINATIONS:
                print(f"[transcribe] skipped (hallucination: {text!r})", flush=True)
                return

            if text:
                pyperclip.copy(text)
                print("[transcribe] copied to clipboard", flush=True)

                # 붙여넣기 — pyautogui.hotkey는 macOS에서 modifier가 누락되어
                # 'v'만 입력되는 버그가 있어 osascript(System Events)로 대체.
                # bundled .app에서는 stdout이 /dev/null이라, 결과를 파일로도 남긴다.
                def do_paste():
                    log_path = Path.home() / "Library" / "Logs" / "voice-transcriber.log"
                    try:
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    try:
                        r = subprocess.run(
                            ["/usr/bin/osascript", "-e",
                             'tell application "System Events" to keystroke "v" using command down'],
                            capture_output=True, text=True, check=False, timeout=3,
                        )
                        try:
                            with log_path.open("a") as f:
                                from datetime import datetime
                                f.write(f"{datetime.now().isoformat()} paste rc={r.returncode} "
                                        f"stdout={r.stdout!r} stderr={r.stderr!r}\n")
                        except Exception:
                            pass
                        if r.returncode == 0:
                            print("[transcribe] pasted", flush=True)
                        else:
                            print(f"[transcribe] paste rc={r.returncode}: {r.stderr.strip()}", flush=True)
                            self._notify("붙여넣기 오류", "",
                                         (r.stderr or f"rc={r.returncode}").strip()[:120])
                    except Exception as e:
                        try:
                            with log_path.open("a") as f:
                                from datetime import datetime
                                f.write(f"{datetime.now().isoformat()} paste EXC {type(e).__name__}: {e}\n")
                        except Exception:
                            pass
                        print(f"[transcribe] paste error: {e}", flush=True)
                        self._notify("붙여넣기 오류", "", str(e)[:120])

                # 약간 딜레이 후 메인루프에서 수행
                time.sleep(0.1)
                self._ui(do_paste)

                self._notify("음성 인식 완료", "", text[:50] + ("..." if len(text) > 50 else ""))
            else:
                print("[transcribe] empty result", flush=True)
                self._notify("음성 인식", "", "인식된 텍스트가 없습니다.")

        except Exception as e:
            print(f"[transcribe] ERROR: {e}", flush=True)
            traceback.print_exc()
            self._notify("오류", "", str(e)[:160])

        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

            # UI 복귀
            self._ui(lambda: setattr(self, "title", f"🎤{LANG_BADGE.get(self.config['language'], self.config['language'].upper())}"))
            self._ui(self.build_menu)

    # ---------------------------
    # Quit
    # ---------------------------
    def quit_app(self, sender):
        """앱 종료"""
        try:
            if self.hotkey_listener:
                self.hotkey_listener.stop()
        except Exception:
            pass

        try:
            if self._ui_timer:
                self._ui_timer.stop()
        except Exception:
            pass

        try:
            if self.stream:
                self.stream.close()
        except Exception:
            pass

        try:
            self.audio.terminate()
        except Exception:
            pass

        try:
            self._engine_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

        rumps.quit_application()


if __name__ == "__main__":
    app = VoiceRecorderApp()
    app.run()
