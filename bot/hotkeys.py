"""Global hotkey listener for controlling the bot while a game is focused."""

import threading
import keyboard


class HotkeyListener:
    """Listens for global hotkeys in a background thread."""

    def __init__(self, start_key: str = "f6", stop_key: str = "f7"):
        self.start_key = start_key
        self.stop_key = stop_key
        self._on_start = None
        self._on_stop = None
        self._active = False

    def on_start(self, callback) -> None:
        self._on_start = callback

    def on_stop(self, callback) -> None:
        self._on_stop = callback

    def _handle_start(self) -> None:
        if not self._active and self._on_start:
            self._active = True
            print(f"[Hotkey] {self.start_key.upper()} pressed — bot ACTIVE")
            self._on_start()

    def _handle_stop(self) -> None:
        if self._active and self._on_stop:
            self._active = False
            print(f"[Hotkey] {self.stop_key.upper()} pressed — bot PAUSED")
            self._on_stop()

    @property
    def is_active(self) -> bool:
        return self._active

    def register(self) -> None:
        """Register hotkeys. Call this from the main thread."""
        keyboard.on_press_key(self.start_key, lambda _: self._handle_start())
        keyboard.on_press_key(self.stop_key, lambda _: self._handle_stop())

    def unregister(self) -> None:
        """Remove all hotkey hooks."""
        keyboard.unhook_all()
