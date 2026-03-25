"""WoW fishing bot: sound-triggered catch with visual bobber location.

Sound detects the splash moment. Template matching locates the bobber on screen.
"""

import random
import time
import threading
from enum import Enum

import cv2
import numpy as np

from bot.screen import ScreenCapture
from bot.audio import AudioMonitor
from bot.hotkeys import HotkeyListener
from bot.vision import find_template
from bot.input import move_human, click, press


class State(Enum):
    IDLE = "idle"
    LOCATING = "locating"
    CATCHING = "catching"
    LOOTING = "looting"


class FishingBot:
    """Sound-triggered fishing bot with visual bobber location.

    Flow:
        1. Audio monitor listens for splash sound (volume spike)
        2. On trigger → screenshot → find_template() to locate bobber
        3. Move mouse to bobber → right-click → loot key
        4. Return to idle

    States:
        IDLE     — waiting for audio trigger
        LOCATING — screenshot taken, searching for bobber
        CATCHING — moving mouse + right-clicking
        LOOTING  — pressing loot key
    """

    def __init__(
        self,
        template_path: str,
        threshold: float = 0.7,
        volume_multiplier: float = 3.0,
        cooldown: float = 3.0,
        loot_key: str = "1",
        monitor: int = 1,
        audio_device: int | None = None,
        start_key: str = "f6",
        stop_key: str = "f7",
    ):
        # Vision: template for bobber location
        self.template = cv2.imread(template_path)
        if self.template is None:
            raise FileNotFoundError(f"Could not load template image: {template_path}")
        self.threshold = threshold

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
        self.state = State.IDLE
        self.catch_count = 0
        self._running = False
        self._splash_event = threading.Event()

    def _activate(self) -> None:
        self.audio.enabled = True
        self.state = State.IDLE
        print("Bot ACTIVE — listening for splash...")

    def _deactivate(self) -> None:
        self.audio.enabled = False
        print("Bot PAUSED")

    def _on_splash(self, rms: float, baseline: float) -> None:
        """Called by audio monitor when a volume spike is detected."""
        ratio = rms / baseline if baseline > 0 else 0
        print(f"Splash detected! (volume {ratio:.1f}x baseline)")
        self._splash_event.set()

    def run(self) -> None:
        """Start the bot. Idles until sound trigger fires."""
        self._running = True
        self.hotkeys.register()
        self.audio.start()

        h, w = self.template.shape[:2]
        print(f"Template loaded ({w}x{h}px), match threshold={self.threshold}")
        print(f"Loot key: {self.loot_key}")
        print(f"Press {self.hotkeys.start_key.upper()} to start, "
              f"{self.hotkeys.stop_key.upper()} to stop, Ctrl+C to quit.")

        try:
            while self._running:
                # Wait for splash event (check every 100ms for ctrl+c)
                if self._splash_event.wait(timeout=0.1):
                    self._splash_event.clear()
                    if self.audio.enabled:
                        self._handle_catch()
        except KeyboardInterrupt:
            print("\nBot quit.")
        finally:
            self._running = False
            self.audio.stop()
            self.hotkeys.unregister()

    def stop(self) -> None:
        self._running = False

    def _handle_catch(self) -> None:
        """Full catch sequence: locate bobber → move → click → loot."""
        self.state = State.LOCATING

        # Take screenshot and find bobber
        frame = self.screen.grab()
        matches = find_template(frame, self.template, self.threshold)

        if not matches:
            print("  WARNING: Could not locate bobber on screen. Skipping.")
            self.state = State.IDLE
            return

        x, y = matches[0]
        self.catch_count += 1
        print(f"[#{self.catch_count}] Bobber at ({x}, {y}) — catching!")

        # Move mouse to bobber with human-like path
        self.state = State.CATCHING
        move_human(x, y)

        # Small pause before clicking
        time.sleep(random.uniform(0.1, 0.3))

        # Right-click the bobber
        click(x, y, button="right")
        print("  Right-clicked bobber")

        # Wait for loot window
        self.state = State.LOOTING
        time.sleep(random.uniform(0.8, 1.5))

        # Press loot key
        press(self.loot_key)
        print(f"  Pressed '{self.loot_key}' to loot")

        # Cooldown
        time.sleep(random.uniform(0.3, 0.6))
        self.state = State.IDLE
        print("  Resuming listen...")
