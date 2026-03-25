"""WoW fishing bot: sound-triggered catch with visual bobber location.

Sound detects the splash moment. Template matching locates the bobber on screen
shortly after activation (not after the splash), and caches the position.
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
    LISTENING = "listening"
    CATCHING = "catching"
    LOOTING = "looting"


class FishingBot:
    """Sound-triggered fishing bot with visual bobber location.

    Flow:
        1. F6 pressed → wait 1s → screenshot → find bobber → cache position
        2. Audio monitor listens for splash sound (volume spike)
        3. On splash → move mouse to cached bobber position → right-click → loot
        4. Return to idle

    States:
        IDLE      — waiting for F6 activation
        LOCATING  — screenshot taken, searching for bobber
        LISTENING — bobber found, waiting for audio trigger
        CATCHING  — moving mouse + right-clicking
        LOOTING   — pressing loot key
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
        locate_delay: float = 1.0,
    ):
        # Vision: template for bobber location
        self.template = cv2.imread(template_path)
        if self.template is None:
            raise FileNotFoundError(f"Could not load template image: {template_path}")
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
        self.state = State.IDLE
        self.catch_count = 0
        self._running = False
        self._splash_event = threading.Event()
        self._activate_event = threading.Event()
        self._bobber_pos: tuple[int, int] | None = None

    def _activate(self) -> None:
        self._activate_event.set()

    def _deactivate(self) -> None:
        self.audio.enabled = False
        self._bobber_pos = None
        self.state = State.IDLE
        print("Bot PAUSED")

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

        h, w = self.template.shape[:2]
        print(f"Template loaded ({w}x{h}px), match threshold={self.threshold}")
        print(f"Loot key: {self.loot_key}")
        print(f"Locate delay: {self.locate_delay}s after activation")
        print(f"Press {self.hotkeys.start_key.upper()} to start, "
              f"{self.hotkeys.stop_key.upper()} to stop, Ctrl+C to quit.")

        try:
            while self._running:
                # Wait for F6 activation
                if self._activate_event.wait(timeout=0.1):
                    self._activate_event.clear()
                    self._run_cycle()
        except KeyboardInterrupt:
            print("\nBot quit.")
        finally:
            self._running = False
            self.audio.stop()
            self.hotkeys.unregister()

    def stop(self) -> None:
        self._running = False

    def _run_cycle(self) -> None:
        """One full fishing cycle: locate bobber → listen for splash → catch."""
        # Step 1: Wait, then locate bobber
        self.state = State.LOCATING
        print(f"Bot ACTIVE - locating bobber in {self.locate_delay}s...")
        time.sleep(self.locate_delay)

        frame = self.screen.grab()
        matches = find_template(frame, self.template, self.threshold)

        if not matches:
            print("  WARNING: Could not locate bobber on screen. Press F6 to retry.")
            self.state = State.IDLE
            return

        self._bobber_pos = matches[0]
        x, y = self._bobber_pos
        print(f"  Bobber found at ({x}, {y})")

        # Step 2: Enable audio and listen for splash
        self.state = State.LISTENING
        self.audio.enabled = True
        print("  Listening for splash...")

        # Wait for splash (or F7 to deactivate)
        while self._running and self.state == State.LISTENING:
            if self._splash_event.wait(timeout=0.1):
                self._splash_event.clear()
                if self.state == State.LISTENING:
                    self._handle_catch()

    def _handle_catch(self) -> None:
        """Catch sequence: move to cached bobber position → click → loot."""
        if self._bobber_pos is None:
            print("  WARNING: No bobber position cached. Skipping.")
            self.state = State.IDLE
            return

        x, y = self._bobber_pos
        self.catch_count += 1
        print(f"[#{self.catch_count}] Moving to bobber at ({x}, {y})")

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

        # Done — back to idle (user presses F6 for next cast)
        time.sleep(random.uniform(0.3, 0.6))
        self.audio.enabled = False
        self._bobber_pos = None
        self.state = State.IDLE
        print("  Done. Press F6 after next cast.")
