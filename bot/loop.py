"""Main game bot loop: screenshot → analyze → act."""

import time
from bot.screen import ScreenCapture
from bot.hotkeys import HotkeyListener


class GameBot:
    """Base game bot with a configurable tick loop.

    Subclass this and override `on_frame()` to implement game-specific logic.
    Controlled by F6 (start) / F7 (stop) hotkeys.
    """

    def __init__(self, tick_rate: float = 0.1, monitor: int = 1, region: dict | None = None,
                 start_key: str = "f6", stop_key: str = "f7"):
        """
        Args:
            tick_rate: Seconds between each frame capture.
            monitor: Monitor index.
            region: Optional screen region to capture.
            start_key: Hotkey to activate the bot.
            stop_key: Hotkey to pause the bot.
        """
        self.tick_rate = tick_rate
        self.screen = ScreenCapture(monitor=monitor, region=region)
        self.running = False
        self.active = False
        self.hotkeys = HotkeyListener(start_key=start_key, stop_key=stop_key)
        self.hotkeys.on_start(self._activate)
        self.hotkeys.on_stop(self._deactivate)

    def _activate(self) -> None:
        self.active = True

    def _deactivate(self) -> None:
        self.active = False

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
        """Start the bot loop. Press F6 to activate, F7 to pause, Ctrl+C to quit."""
        self.running = True
        self.hotkeys.register()
        self.on_start()
        print(f"Bot ready. Press {self.hotkeys.start_key.upper()} to start, "
              f"{self.hotkeys.stop_key.upper()} to stop, Ctrl+C to quit.")
        try:
            while self.running:
                if not self.active:
                    time.sleep(0.1)
                    continue
                start = time.perf_counter()
                frame = self.screen.grab()
                self.on_frame(frame)
                elapsed = time.perf_counter() - start
                sleep_time = max(0, self.tick_rate - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\nBot quit.")
        finally:
            self.running = False
            self.active = False
            self.hotkeys.unregister()
            self.on_stop()

    def stop(self) -> None:
        """Signal the bot to stop entirely."""
        self.running = False
