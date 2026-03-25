"""WoW fishing bot: sound-triggered catch with visual bobber location.

Sound detects the splash moment. Template matching locates the bobber on screen
shortly after activation (not after the splash), and caches the position.
"""

import os
import random
import time
import threading
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from bot.screen import ScreenCapture
from bot.audio import AudioMonitor
from bot.hotkeys import HotkeyListener
from bot.vision import find_template_multiscale, find_bobber_by_color
from bot.input import move_human, click, press

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def find_latest_template(template_dir: str) -> str | None:
    """Find the most recently modified image file in a directory."""
    path = Path(template_dir)
    if not path.is_dir():
        return None
    images = [f for f in path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        return None
    latest = max(images, key=lambda f: f.stat().st_mtime)
    return str(latest)


class State(Enum):
    IDLE = "idle"
    LOCATING = "locating"
    LISTENING = "listening"
    CATCHING = "catching"
    LOOTING = "looting"


class FishingBot:
    """Sound-triggered fishing bot with visual bobber location and auto-loop.

    Flow (auto-loop after F6):
        1. F6 pressed → cast ('o') → wait → locate bobber → listen → catch → loot
        2. Loops endlessly until F7 is pressed

    States:
        IDLE      — waiting for F6 activation
        LOCATING  — screenshot taken, searching for bobber
        LISTENING — bobber found, waiting for audio trigger
        CATCHING  — moving mouse + right-clicking
        LOOTING   — pressing loot key
    """

    def __init__(
        self,
        template_dir: str,
        threshold: float = 0.6,
        volume_multiplier: float = 3.0,
        cooldown: float = 3.0,
        loot_key: str = "1",
        cast_delay: float = 2.0,
        monitor: int = 1,
        audio_device: int | None = None,
        start_key: str = "f6",
        stop_key: str = "f7",
        locate_delay: float = 1.0,
        debug: bool = False,
        roi: dict | None = None,
        pole_pos: dict | None = None,
        bauble_interval: float = 600.0,
    ):
        # Vision: template directory for bobber location
        self.debug = debug
        self.roi = roi  # {"top": N, "left": N, "width": N, "height": N} or None
        self.pole_pos = pole_pos  # {"x": N, "y": N} or None
        self.bauble_interval = bauble_interval  # seconds, 0 = disabled
        self.template_dir = template_dir
        if not Path(template_dir).is_dir():
            raise FileNotFoundError(f"Template directory not found: {template_dir}")
        self.template = None
        self._template_path = None
        self.threshold = threshold
        self.locate_delay = locate_delay

        # Screen capture
        self.screen = ScreenCapture(monitor=monitor)

        # Audio monitor
        self.audio = AudioMonitor(
            threshold_multiplier=volume_multiplier,
            cooldown=cooldown,
            device=audio_device,
        )
        self.audio.on_trigger(self._on_splash)

        # Hotkeys
        self.hotkeys = HotkeyListener(start_key=start_key, stop_key=stop_key)
        self.hotkeys.on_start(self._activate)
        self.hotkeys.on_stop(self._deactivate)

        # State
        self.loot_key = loot_key
        self.cast_key = "o"  # hardcoded fishing hotkey
        self.cast_delay = cast_delay
        self.state = State.IDLE
        self.catch_count = 0
        self._running = False
        self._looping = False
        self._splash_event = threading.Event()
        self._activate_event = threading.Event()
        self._bobber_pos: tuple[int, int] | None = None
        self._last_bauble_time: float = 0.0  # epoch when baubles were last applied

    def _activate(self) -> None:
        self._activate_event.set()

    def _deactivate(self) -> None:
        self._looping = False
        self.audio.enabled = False
        self._bobber_pos = None
        self._splash_event.set()  # unblock any waiting splash listener
        self.state = State.IDLE
        print("Bot PAUSED (F7) — loop stopped")

    def _on_splash(self, rms: float, baseline: float) -> None:
        """Called by audio monitor when a volume spike is detected."""
        ratio = rms / baseline if baseline > 0 else 0
        print(f"Splash detected! (volume {ratio:.1f}x baseline)")
        self._splash_event.set()

    def run(self) -> None:
        """Start the bot. Locates bobber on activation, then listens for splash."""
        self._running = True
        self.hotkeys.register()
        self.audio.start()

        print(f"Template dir: {self.template_dir} (loads latest on each cycle)")
        print(f"Match threshold: {self.threshold} (multi-scale 0.5x–1.5x)")
        if self.roi:
            r = self.roi
            print(f"ROI: ({r['left']}, {r['top']}) {r['width']}x{r['height']}px")
        else:
            print("ROI: full screen (run --setup to set one)")
        if self.pole_pos and self.bauble_interval > 0:
            print(f"Bauble: every {self.bauble_interval / 60:.0f}min → "
                  f"pole at ({self.pole_pos['x']}, {self.pole_pos['y']})")
        elif not self.pole_pos:
            print("Bauble: disabled (no pole position — run --setup)")
        else:
            print("Bauble: disabled (interval=0)")
        print(f"Cast key: '{self.cast_key}' | Loot key: '{self.loot_key}'")
        print(f"Cast delay: {self.cast_delay}s | Locate delay: {self.locate_delay}s")
        print(f"Press {self.hotkeys.start_key.upper()} to start, "
              f"{self.hotkeys.stop_key.upper()} to stop, Ctrl+C to quit.")

        try:
            while self._running:
                # Wait for F6 activation
                if self._activate_event.wait(timeout=0.1):
                    self._activate_event.clear()
                    self._looping = True
                    self._loop()
        except KeyboardInterrupt:
            print("\nBot quit.")
        finally:
            self._running = False
            self._looping = False
            self.audio.stop()
            self.hotkeys.unregister()

    def _needs_bauble(self) -> bool:
        """Check if it's time to apply a bauble."""
        if not self.pole_pos or self.bauble_interval <= 0:
            return False
        if self._last_bauble_time == 0.0:
            return False  # skip first run, start the timer
        return (time.time() - self._last_bauble_time) >= self.bauble_interval

    def _apply_bauble(self) -> None:
        """Apply bauble to fishing pole: press 'i' (action bar), click pole on screen."""
        if not self.pole_pos:
            return

        px, py = self.pole_pos["x"], self.pole_pos["y"]
        print(f"\n  ** APPLYING BAUBLE **")
        print(f"  Selecting bauble (pressing 'i')...")
        press("i")
        time.sleep(random.uniform(0.5, 0.8))

        print(f"  Moving to fishing pole at ({px}, {py})...")
        move_human(px, py)
        time.sleep(random.uniform(0.2, 0.4))

        print(f"  Left-clicking fishing pole...")
        click(px, py, button="left")
        time.sleep(random.uniform(0.8, 1.2))

        self._last_bauble_time = time.time()
        print(f"  Bauble applied. Next in {self.bauble_interval / 60:.0f}min.")

    def _loop(self) -> None:
        """Auto-loop: cast → locate → listen → catch → loot → repeat until F7."""
        # Start the bauble timer on first loop activation
        if self._last_bauble_time == 0.0:
            self._last_bauble_time = time.time()
        cycle = 0
        while self._running and self._looping:
            cycle += 1
            print(f"\n{'='*40}")
            print(f"  CYCLE #{cycle}")
            print(f"{'='*40}")

            # Check if bauble needs to be applied
            if self._needs_bauble():
                self._apply_bauble()
                if not self._looping:
                    break

            # Cast: press the fishing key
            print(f"  Casting (pressing '{self.cast_key}')...")
            press(self.cast_key)
            # Wait for cast animation + bobber to land
            delay = self.cast_delay + random.uniform(-0.3, 0.5)
            print(f"  Waiting {delay:.1f}s for bobber to land...")
            time.sleep(delay)

            if not self._looping:
                break

            self._run_cycle()

            if not self._looping:
                break

            # Small pause between cycles
            pause = random.uniform(0.5, 1.5)
            print(f"  Pausing {pause:.1f}s before next cast...")
            time.sleep(pause)

        print(f"\nLoop ended after {cycle} cycle(s).")

    def stop(self) -> None:
        self._running = False

    def _load_latest_template(self) -> bool:
        """Load the most recently modified template from the template directory."""
        latest = find_latest_template(self.template_dir)
        if latest is None:
            print(f"  WARNING: No image files found in {self.template_dir}")
            return False
        if latest != self._template_path:
            self.template = cv2.imread(latest)
            if self.template is None:
                print(f"  WARNING: Could not read image: {latest}")
                return False
            self._template_path = latest
            h, w = self.template.shape[:2]
            print(f"  Loaded template: {Path(latest).name} ({w}x{h}px)")
        return True

    def _locate_bobber(self) -> tuple[int, int] | None:
        """Try to locate bobber on screen, retrying up to 3 times.

        If an ROI is configured, crops the screenshot to that region first,
        then offsets the match coordinates back to full-screen.
        """
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            if not self._looping:
                return None

            if attempt > 1:
                print(f"  Retry {attempt}/{max_retries} — waiting 1s...")
                time.sleep(1.0)

            frame = self.screen.grab()

            # Crop to ROI if configured
            offset_x, offset_y = 0, 0
            if self.roi:
                t = self.roi["top"]
                l = self.roi["left"]
                w = self.roi["width"]
                h = self.roi["height"]
                frame = frame[t:t + h, l:l + w]
                offset_x, offset_y = l, t
                if self.debug and attempt == 1:
                    print(f"  [DEBUG] ROI: ({l}, {t}) {w}x{h}px")

            # Primary: multi-scale template matching
            matches = find_template_multiscale(
                frame, self.template, self.threshold, debug=self.debug
            )

            if matches:
                mx, my = matches[0]
                print(f"  Found via template matching.")
                return (mx + offset_x, my + offset_y)

            # Fallback: color-based detection (red/blue feathers + white body)
            if self.debug:
                print(f"  Template failed, trying color detection...")
            color_matches = find_bobber_by_color(frame, debug=self.debug)

            if color_matches:
                mx, my = color_matches[0]
                print(f"  Found via color detection.")
                return (mx + offset_x, my + offset_y)

            print(f"  Attempt {attempt}/{max_retries}: bobber not found.")

        print("  WARNING: Could not locate bobber after 3 attempts. Press F7 to stop or F6 to retry.")
        return None

    def _run_cycle(self) -> None:
        """One full fishing cycle: locate bobber → move mouse → listen for splash → catch."""
        # Step 0: Load latest template
        self.state = State.LOCATING
        if not self._load_latest_template():
            self._looping = False
            self.state = State.IDLE
            return

        # Step 1: Wait, then locate bobber (with retries)
        print(f"  Locating bobber in {self.locate_delay}s...")
        time.sleep(self.locate_delay)

        pos = self._locate_bobber()
        if pos is None:
            # Bobber not found — skip this cycle, re-cast on next loop iteration
            print("  Skipping cycle, will re-cast...")
            self.audio.enabled = False
            self.state = State.IDLE
            return

        self._bobber_pos = pos
        x, y = self._bobber_pos
        print(f"  Bobber found at ({x}, {y})")

        # Step 2: Move mouse to bobber immediately
        print(f"  Moving mouse to bobber...")
        move_human(x, y)

        # Step 3: Enable audio and listen for splash
        self.state = State.LISTENING
        self.audio.enabled = True
        listen_start = time.time()
        listen_timeout = 30.0  # WoW bobber despawns after ~30s
        print(f"  Listening for splash (timeout {listen_timeout}s)...")

        # Wait for splash, F7, or timeout
        while self._running and self._looping and self.state == State.LISTENING:
            elapsed = time.time() - listen_start
            if elapsed >= listen_timeout:
                print("  Bobber expired (no splash detected). Re-casting...")
                self.audio.enabled = False
                self._bobber_pos = None
                self.state = State.IDLE
                return  # back to _loop() for next cycle

            if self._splash_event.wait(timeout=0.1):
                self._splash_event.clear()
                if self.state == State.LISTENING and self._looping:
                    self._handle_catch()
                    return  # back to _loop() for next cycle

    def _handle_catch(self) -> None:
        """Catch sequence: right-click bobber (mouse is already there) → wait for loot."""
        if self._bobber_pos is None:
            print("  WARNING: No bobber position cached. Skipping.")
            self.state = State.IDLE
            return

        x, y = self._bobber_pos
        self.catch_count += 1
        print(f"[#{self.catch_count}] Catching!")

        # Brief human reaction delay after hearing the splash
        self.state = State.CATCHING
        reaction_delay = random.uniform(0.3, 0.7)
        print(f"  Reaction delay: {reaction_delay:.2f}s")
        time.sleep(reaction_delay)

        # Right-click the bobber (mouse already positioned there)
        click(x, y, button="right")
        print("  Right-clicked bobber — looting...")

        # Wait for loot to complete
        self.state = State.LOOTING
        time.sleep(1.0)

        # Done — disable audio until next cycle re-enables it
        self.audio.enabled = False
        self._bobber_pos = None
        self.state = State.IDLE
        print("  Catch complete.")
