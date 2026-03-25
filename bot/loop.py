"""Main game bot loop: screenshot → analyze → act."""

import time
from bot.screen import ScreenCapture
from bot.vision import find_template, find_color_regions
from bot.input import click, press, move


class GameBot:
    """Base game bot with a configurable tick loop.

    Subclass this and override `on_frame()` to implement game-specific logic.
    """

    def __init__(self, tick_rate: float = 0.1, monitor: int = 1, region: dict | None = None):
        """
        Args:
            tick_rate: Seconds between each frame capture.
            monitor: Monitor index.
            region: Optional screen region to capture.
        """
        self.tick_rate = tick_rate
        self.screen = ScreenCapture(monitor=monitor, region=region)
        self.running = False

    def on_frame(self, frame) -> None:
        """Called each tick with the current screenshot. Override in subclass."""
        pass

    def on_start(self) -> None:
        """Called when the bot starts. Override for setup."""
        pass

    def on_stop(self) -> None:
        """Called when the bot stops. Override for cleanup."""
        pass

    def run(self) -> None:
        """Start the bot loop. Press Ctrl+C to stop."""
        self.running = True
        self.on_start()
        print(f"Bot started (tick_rate={self.tick_rate}s). Press Ctrl+C to stop.")
        try:
            while self.running:
                start = time.perf_counter()
                frame = self.screen.grab()
                self.on_frame(frame)
                elapsed = time.perf_counter() - start
                sleep_time = max(0, self.tick_rate - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nBot stopped by user.")
        finally:
            self.running = False
            self.on_stop()

    def stop(self) -> None:
        """Signal the bot to stop."""
        self.running = False
