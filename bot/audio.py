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
    """Auto-detect the best WASAPI loopback device index.

    Returns the device index or None if not found.
    """
    hostapis = sd.query_hostapis()
    wasapi_index = None
    for i, api in enumerate(hostapis):
        if "wasapi" in api["name"].lower():
            wasapi_index = i
            break

    if wasapi_index is None:
        return None

    # Find the default output device's loopback
    devices = sd.query_devices()
    default_output = sd.default.device[1]
    if default_output is not None and default_output >= 0:
        dev = devices[default_output]
        # WASAPI loopback uses the output device as input
        if dev["hostapi"] == wasapi_index:
            return default_output

    # Fallback: find any WASAPI device with input channels
    for i, dev in enumerate(devices):
        if dev["hostapi"] == wasapi_index and dev["max_input_channels"] > 0:
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

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        if value:
            # Reset baseline when re-enabled
            self._rms_history.clear()

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

    def _monitor_loop(self) -> None:
        """Background thread: read audio chunks and check for spikes."""
        dev_info = sd.query_devices(self.device)
        sample_rate = int(dev_info["default_samplerate"])
        channels = max(1, dev_info.get("max_input_channels", 1))
        block_size = int(sample_rate * self.block_duration)

        try:
            with sd.InputStream(
                device=self.device,
                samplerate=sample_rate,
                channels=channels,
                blocksize=block_size,
                dtype="float32",
            ) as stream:
                while self._running:
                    data, overflowed = stream.read(block_size)
                    if overflowed:
                        continue

                    rms = self._compute_rms(data)
                    self._rms_history.append(rms)

                    if not self._enabled:
                        continue

                    # Need enough samples for a baseline
                    if len(self._rms_history) < 20:
                        continue

                    baseline = float(np.median(list(self._rms_history)))
                    if baseline < 1e-6:
                        # Near-silence, use absolute threshold
                        baseline = 0.001

                    # Check for spike
                    now = time.time()
                    if rms > baseline * self.threshold_multiplier:
                        if now - self._last_trigger > self.cooldown:
                            self._last_trigger = now
                            if self._callback:
                                self._callback(rms, baseline)

        except Exception as e:
            print(f"Audio monitor error: {e}")
            self._running = False
