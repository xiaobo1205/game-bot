"""Audio monitoring for detecting sound spikes (e.g., WoW bobber splash).

Uses sounddevice with WASAPI loopback to capture system audio output
without needing a virtual audio cable.
"""

import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd


def list_loopback_devices() -> list[dict]:
    """List available WASAPI loopback devices for capturing system audio."""
    devices = []
    for i, dev in enumerate(sd.query_devices()):
        # WASAPI loopback devices typically have "Loopback" in the name
        # or are output devices that can be opened as loopback
        if dev["max_input_channels"] > 0 or "loopback" in dev["name"].lower():
            devices.append({"index": i, "name": dev["name"], "channels": dev["max_input_channels"],
                            "sample_rate": dev["default_samplerate"]})
    return devices


def find_loopback_device() -> int | None:
    """Auto-detect the best audio input device for capturing system sound.

    Prefers Stereo Mix (captures all system audio output as an input device).
    Falls back to any input device.

    Returns the device index or None if not found.
    """
    devices = sd.query_devices()

    # Prefer "Stereo Mix" — it captures system audio as an input device
    for i, dev in enumerate(devices):
        if "stereo mix" in dev["name"].lower() and dev["max_input_channels"] > 0:
            print(f"  Auto-detected Stereo Mix: [{i}] {dev['name']}")
            return i

    # Fallback: any input device with "loopback" or "mix" in name
    for i, dev in enumerate(devices):
        name = dev["name"].lower()
        if dev["max_input_channels"] > 0 and ("loopback" in name or "mix" in name):
            print(f"  Auto-detected loopback device: [{i}] {dev['name']}")
            return i

    # Last resort: any input device
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            return i

    return None


class AudioMonitor:
    """Monitors system audio for volume spikes.

    Runs a background thread that continuously samples audio, maintains
    a rolling baseline of ambient volume, and fires a callback when
    the volume exceeds baseline * threshold_multiplier.
    """

    def __init__(
        self,
        threshold_multiplier: float = 3.0,
        cooldown: float = 3.0,
        device: int | None = None,
        block_duration: float = 0.05,
        baseline_window: int = 100,
        debug: bool = False,
    ):
        """
        Args:
            threshold_multiplier: Volume must exceed baseline * this to trigger.
            cooldown: Seconds to wait after a trigger before allowing another.
            device: Audio device index. None = auto-detect loopback.
            block_duration: Audio chunk size in seconds (~50ms).
            baseline_window: Number of chunks to average for baseline.
        """
        self.threshold_multiplier = threshold_multiplier
        self.cooldown = cooldown
        self.block_duration = block_duration
        self.baseline_window = baseline_window

        if device is None:
            self.device = find_loopback_device()
        else:
            self.device = device

        self._callback = None
        self._running = False
        self._thread = None
        self._last_trigger = 0.0
        self._rms_history = deque(maxlen=baseline_window)
        self._enabled = False
        self._debug = debug
        self._debug_counter = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        if value:
            # Reset baseline and cooldown when re-enabled
            self._rms_history.clear()
            self._last_trigger = 0.0

    def on_trigger(self, callback) -> None:
        """Set the callback to fire when a splash is detected.

        callback receives (rms, baseline) as arguments.
        """
        self._callback = callback

    def start(self) -> None:
        """Start monitoring audio in a background thread."""
        if self.device is None:
            print("WARNING: No loopback audio device found. Sound detection disabled.")
            print("Run with --list-devices to see available devices.")
            return

        dev_info = sd.query_devices(self.device)
        print(f"Audio device: {dev_info['name']}")
        print(f"  Sample rate: {dev_info['default_samplerate']}Hz")
        print(f"  Threshold: {self.threshold_multiplier}x ambient, Cooldown: {self.cooldown}s")

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the audio monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _compute_rms(self, audio_data: np.ndarray) -> float:
        """Compute root mean square of audio data."""
        return float(np.sqrt(np.mean(audio_data.astype(np.float64) ** 2)))

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """Called by sounddevice for each audio block (runs in audio thread)."""
        if status:
            if self._debug:
                print(f"  [AUDIO] Stream status: {status}")
            return

        rms = self._compute_rms(indata)
        self._rms_history.append(rms)

        # Periodic debug: print RMS every ~2 seconds (every 40 blocks at 50ms each)
        self._debug_counter += 1
        if self._debug and self._debug_counter % 40 == 0:
            baseline = float(np.median(list(self._rms_history))) if len(self._rms_history) >= 5 else 0.0
            ratio = rms / baseline if baseline > 1e-6 else 0.0
            print(f"  [AUDIO] rms={rms:.6f} baseline={baseline:.6f} ratio={ratio:.1f}x "
                  f"enabled={self._enabled} samples={len(self._rms_history)}")

        if not self._enabled:
            return

        # Need enough samples for a baseline
        if len(self._rms_history) < 20:
            return

        baseline = float(np.median(list(self._rms_history)))
        if baseline < 1e-6:
            baseline = 0.001

        now = time.time()
        if rms > baseline * self.threshold_multiplier:
            if now - self._last_trigger > self.cooldown:
                self._last_trigger = now
                if self._callback:
                    self._callback(rms, baseline)

    def _monitor_loop(self) -> None:
        """Background thread: open callback-based audio stream and keep it alive."""
        dev_info = sd.query_devices(self.device)
        sample_rate = int(dev_info["default_samplerate"])
        channels = dev_info.get("max_input_channels", 0)
        block_size = int(sample_rate * self.block_duration)

        if channels == 0:
            print(f"  ERROR: Device '{dev_info['name']}' is output-only. "
                  f"Use an input device like Stereo Mix. Run --list-devices.")
            self._running = False
            return

        try:
            with sd.InputStream(
                device=self.device,
                samplerate=sample_rate,
                channels=channels,
                blocksize=block_size,
                dtype="float32",
                callback=self._audio_callback,
            ):
                print(f"  Audio stream opened successfully on device {self.device}")
                # Keep thread alive while stream runs via callback
                while self._running:
                    time.sleep(0.1)

        except Exception as e:
            print(f"Audio monitor error: {e}")
            print("  Sound detection will not work. Check --device or --list-devices.")
            self._running = False
