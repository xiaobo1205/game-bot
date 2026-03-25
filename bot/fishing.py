"""WoW fishing bot: detect bobber splash via template matching, then loot."""

import random
import time
from enum import Enum

import cv2
import numpy as np

from bot.loop import GameBot
from bot.vision import find_template
from bot.input import move_human, click, press


class State(Enum):
    WATCHING = "watching"
    CATCHING = "catching"
    LOOTING = "looting"
    COOLDOWN = "cooldown"


class FishingBot(GameBot):
    """Watches for a bobber splash, moves mouse to it, right-clicks, and loots.

    States:
        WATCHING  — scanning each frame for template match
        CATCHING  — matched! moving mouse + right-clicking
        LOOTING   — pressing loot key after a pause
        COOLDOWN  — brief pause before resuming watch
    """

    def __init__(
        self,
        template_path: str,
        threshold: float = 0.75,
        tick_rate: float = 1.0,
        loot_key: str = "1",
        monitor: int = 1,
        start_key: str = "f6",
        stop_key: str = "f7",
    ):
        super().__init__(tick_rate=tick_rate, monitor=monitor, start_key=start_key, stop_key=stop_key)
        self.template = cv2.imread(template_path)
        if self.template is None:
            raise FileNotFoundError(f"Could not load template image: {template_path}")
        self.threshold = threshold
        self.loot_key = loot_key
        self.state = State.WATCHING
        self.catch_count = 0

    def on_start(self) -> None:
        h, w = self.template.shape[:2]
        print(f"Template loaded ({w}x{h}px), threshold={self.threshold}")
        print(f"Loot key: {self.loot_key}")
        print(f"Scan interval: {self.tick_rate}s")

    def on_frame(self, frame: np.ndarray) -> None:
        if self.state != State.WATCHING:
            return

        matches = find_template(frame, self.template, self.threshold)
        if not matches:
            return

        # Take the first (best) match
        x, y = matches[0]
        self.catch_count += 1
        print(f"[#{self.catch_count}] Bobber detected at ({x}, {y}) — catching!")

        self.state = State.CATCHING
        self._catch(x, y)

    def _catch(self, x: int, y: int) -> None:
        """Move to bobber, right-click, loot, then return to watching."""
        # Move mouse with human-like curve
        move_human(x, y)

        # Small pause before clicking
        time.sleep(random.uniform(0.1, 0.3))

        # Right-click the bobber
        click(x, y, button="right")
        print("  Right-clicked bobber")

        # Wait for loot window to appear
        self.state = State.LOOTING
        time.sleep(random.uniform(0.8, 1.5))

        # Press loot key
        press(self.loot_key)
        print(f"  Pressed '{self.loot_key}' to loot")

        # Cooldown before next scan
        self.state = State.COOLDOWN
        time.sleep(random.uniform(0.5, 1.0))

        self.state = State.WATCHING
        print("  Resuming watch...")
