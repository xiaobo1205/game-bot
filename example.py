"""Example: a bot that clicks on red objects on screen."""

import cv2
import numpy as np
from bot.loop import GameBot
from bot.vision import find_color_regions
from bot.input import click


class RedClickerBot(GameBot):
    """Finds red-colored regions on screen and clicks the largest one."""

    def on_frame(self, frame: np.ndarray) -> None:
        # Detect red in HSV space (two ranges because red wraps around H=0)
        regions_low = find_color_regions(frame, (0, 120, 70), (10, 255, 255), min_area=200)
        regions_high = find_color_regions(frame, (170, 120, 70), (180, 255, 255), min_area=200)
        regions = regions_low + regions_high

        if regions:
            # Click the largest red region
            biggest = max(regions, key=lambda r: r["area"])
            print(f"Found red at ({biggest['x']}, {biggest['y']}), area={biggest['area']}")
            click(biggest["x"], biggest["y"])


if __name__ == "__main__":
    bot = RedClickerBot(tick_rate=0.5)
    bot.run()
