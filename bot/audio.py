"""Audio monitoring for detecting sound spikes (e.g., WoW bobber splash).

Uses pyaudiowpatch with WASAPI loopback to capture system audio output
directly — no Stereo Mix or virtual audio cable needed.
"""

import threading
import time
from collections import deque

import numpy as np
import pyaudiowpatch as pyaudio


def find_loopback_device(p: pyaudio.PyAudio, preferred_name: str | None = None) -> dict | None:
    """Auto-detect the best WASAPI loopback device.

    Prefers the default speakers' loopback. If preferred_name is given,
    tries to match that device name first.

    Returns device info dict or None.
    """
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        print("  WARNING: WASAPI host API not available.")
        return None

    # Collect all loopback devices
    loopback_devices = []
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get("isLoopbackDevice", False):
            loopback_devices.append(dev)

    if not loopback_devices:
        print("  WARNING: No WASAPI loopback devices found.")
        return None

    # Try preferred name first
    if preferred_name:
        for dev in loopback_devices:
            if preferred_name.lower() in dev["name"].lower():
                print(f"  Matched preferred device: [{dev['index']}] {dev['name']}")
                return dev

    # Try default output device's loopback
    default_output = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    default_name = default_output["name"].lower()
    for dev in loopback_devices:
        if default_name in dev["name"].lower():
            print(f"  Auto-detected loopback for default output: [{dev['index']}] {dev['name']}")
            return dev

    # Fallback: first loopback device
    dev = loopback_devices[0]
    print(f"  Fallback loopback device: [{dev['index']}] {dev['name']}")
    return dev


def list_all_devices() -> None:
    """Print all available audio devices with loopback info."""
    p = pyaudio.PyAudio()
    print("Available audio devices:")
    print("-" * 70)
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        ch_in = dev["maxInputChannels"]
        ch_out = dev["maxOutputChannels"]
        loopback = " [LOOPBACK]" if dev.get("isLoopbackDevice", False) else ""
        direction = ""
        if ch_in > 0:
            direction += "IN"
        if ch_out > 0:
            direction += "/OUT" if direction else "OUT"
        print(f"  [{i:2d}] {dev['name']:<50s} {direction}{loopback}")
    print("-" * 70)

    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        print(f"\nDefault WASAPI output: [{default_out['index']}] {default_out['name']}")
    except OSError:
        print("\nWASAPI not available.")

    p.terminate()


class AudioMonitor:
    """Monitors system audio for volume spikes via WASAPI loopback.

    Uses pyaudiowpatch to capture audio from any output device's loopback
    without needing Stereo Mix or a virtual audio cable.
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
        self.threshold_multiplier = threshold_multiplier
        self.cooldown = cooldown
        self.block_duration = block_duration
        self.baseline_window = baseline_window

        self._p = pyaudio.PyAudio()

        # Find the loopback device
        if device is not None:
            self._device_info = self._p.get_device_info_by_index(device)
        else:
            self._device_info = find_loopback_device(self._p)

        self._callback_fn = None
        self._running = False
        self._thread = None
        self._last_trigger = 0.0
        self._rms_history: deque[float] = deque(maxlen=baseline_window)
        self._enabled = False
        self._suppress_until: float = 0.0  # ignore triggers until this epoch
        self._debug = debug
        self._debug_counter = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value
        if value:
            # Reset cooldown but keep baseline history — clearing it causes
            # false triggers because the new baseline is built from too few
            # near-silence samples
            self._last_trigger = 0.0

    def suppress(self, seconds: float) -> None:
        """Suppress triggers for the next N seconds."""
        self._suppress_until = time.time() + seconds

    def on_trigger(self, callback) -> None:
        """Set the callback to fire when a splash is detected."""
        self._callback_fn = callback

    def start(self) -> None:
        """Start monitoring audio in a background thread."""
        if self._device_info is None:
            print("WARNING: No loopback audio device found. Sound detection disabled.")
            print("Run with --list-devices to see available devices.")
            return

        dev = self._device_info
        print(f"Audio device: {dev['name']}")
        print(f"  Sample rate: {dev['defaultSampleRate']}Hz, Channels: {dev['maxInputChannels']}")
        print(f"  Loopback: {dev.get('isLoopbackDevice', False)}")
        print(f"  Threshold: {self.threshold_multiplier}x ambient, Cooldown: {self.cooldown}s")

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the audio monitor."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._p.terminate()

    def _compute_rms(self, audio_data: np.ndarray) -> float:
        """Compute root mean square of audio data."""
        return float(np.sqrt(np.mean(audio_data.astype(np.float64) ** 2)))

    def _process_block(self, data: bytes, channels: int) -> None:
        """Process one audio block: compute RMS, check for spike, fire callback."""
        audio = np.frombuffer(data, dtype=np.float32)
        if channels > 1:
            audio = audio.reshape(-1, channels)
            audio = np.mean(audio, axis=1)  # mix to mono

        rms = self._compute_rms(audio)
        self._rms_history.append(rms)

        # Periodic debug output
        self._debug_counter += 1
        if self._debug and self._debug_counter % 40 == 0:
            baseline = float(np.median(list(self._rms_history))) if len(self._rms_history) >= 5 else 0.0
            ratio = rms / baseline if baseline > 1e-6 else 0.0
            print(f"  [AUDIO] rms={rms:.6f} baseline={baseline:.6f} ratio={ratio:.1f}x "
                  f"enabled={self._enabled} samples={len(self._rms_history)}")

        if not self._enabled:
            return

        if len(self._rms_history) < 60:
            return

        baseline = float(np.median(list(self._rms_history)))
        if baseline < 1e-6:
            baseline = 0.001

        now = time.time()

        # Suppressed period — ignore all triggers
        if now < self._suppress_until:
            return

        # Require BOTH relative spike (vs baseline) AND absolute minimum RMS
        # to avoid false triggers from mouse clicks / keyboard sounds when
        # the ambient level is near-silence. The splash sound in WoW is
        # loud enough to produce RMS > 0.05.
        min_abs_rms = 0.05
        if rms > baseline * self.threshold_multiplier and rms > min_abs_rms:
            if now - self._last_trigger > self.cooldown:
                self._last_trigger = now
                if self._callback_fn:
                    self._callback_fn(rms, baseline)

    def _monitor_loop(self) -> None:
        """Background thread: open WASAPI loopback stream and read audio."""
        dev = self._device_info
        sample_rate = int(dev["defaultSampleRate"])
        channels = dev["maxInputChannels"]
        block_size = int(sample_rate * self.block_duration)

        if channels == 0:
            print(f"  ERROR: Device '{dev['name']}' has no input channels.")
            self._running = False
            return

        try:
            stream = self._p.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=dev["index"],
                frames_per_buffer=block_size,
            )
            print(f"  WASAPI loopback stream opened on [{dev['index']}] {dev['name']}")

            while self._running:
                try:
                    data = stream.read(block_size, exception_on_overflow=False)
                    self._process_block(data, channels)
                except OSError:
                    # Stream hiccup, skip this block
                    pass

            stream.stop_stream()
            stream.close()

        except Exception as e:
            print(f"Audio monitor error: {e}")
            print("  Sound detection will not work. Check --device or --list-devices.")
            self._running = False
