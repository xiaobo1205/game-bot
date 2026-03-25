"""WoW fishing bot: detect bobber splash via color-based HSV matching, then loot."""

import random
import time
from enum import Enum

import cv2
import numpy as np

from bot.loop import GameBot
from bot.vision import extract_hsv_range, find_color_regions
from bot.input import move_human, click, press


class State(Enum):
    WATCHING = "watching"
    CATCHING = "catching"
    LOOTING = "looting"
    COOLDOWN = "cooldown"


class FishingBot(GameBot):
    """Watches for a bobber splash using color detection, then right-clicks and loots.

    Uses the template image to auto-calibrate an HSV color range, then scans
    each frame for matching color regions. More robust than pixel-exact template
    matching — tolerates resolution, lighting, and UI scale differences.

    States:
        WATCHING  — scanning each frame for color match
        CATCHING  — matched! moving mouse + right-clicking
        LOOTING   — pressing loot key after a pause
        COOLDOWN  — brief pause before resuming watch
    """

    def __init__(
        self,
        template_path: str,
        min_area: int = 200,
        tick_rate: float = 1.0,
        loot_key: str = "1",
        monitor: int = 1,
        start_key: str = "f6",
        stop_key: str = "f7",
    ):
        super().__init__(tick_rate=tick_rate, monitor=monitor, start_key=start_key, stop_key=stop_key)
        template = cv2.imread(template_path)
        if template is None:
            raise FileNotFoundError(f"Could not load template image: {template_path}")
        self.lower_hsv, self.upper_hsv = extract_hsv_range(template)
        self.min_area = min_area
        self.loot_key = loot_key
        self.state = State.WATCHING
        self.catch_count = 0

    def on_start(self) -> None:
        print(f"HSV range: {self.lower_hsv} → {self.upper_hsv}")
        print(f"Min area: {self.min_area}px, Loot key: {self.loot_key}")
        print(f"Scan interval: {self.tick_rate}s")

    def on_frame(self, frame: np.ndarray) -> None:
        if self.state != State.WATCHING:
            return

        regions = find_color_regions(frame, self.lower_hsv, self.upper_hsv, min_area=self.min_area)
        if not regions:
            return

        # Pick the largest matching region (most likely the bobber splash)
        biggest = max(regions, key=lambda r: r["area"])
        x, y = biggest["x"], biggest["y"]
        self.catch_count += 1
        print(f"[#{self.catch_count}] Bobber detected at ({x}, {y}), area={biggest['area']} — catching!")

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
