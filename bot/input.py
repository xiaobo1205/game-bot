"""Input simulation for mouse and keyboard actions."""

import time
import pyautogui

# Disable pyautogui's built-in pause and failsafe for game use
pyautogui.PAUSE = 0.02
pyautogui.FAILSAFE = True  # Move mouse to top-left corner to abort

try:
    import pydirectinput
    pydirectinput.PAUSE = 0.02
    _USE_DIRECT = True
except ImportError:
    _USE_DIRECT = False


def click(x: int, y: int, button: str = "left", direct: bool = True) -> None:
    """Click at screen coordinates.

    Args:
        x, y: Screen coordinates.
        button: 'left', 'right', or 'middle'.
        direct: Use DirectInput (needed for most games). Falls back to pyautogui if unavailable.
    """
    if direct and _USE_DIRECT:
        pydirectinput.click(x, y, button=button)
    else:
        pyautogui.click(x, y, button=button)


def move(x: int, y: int, duration: float = 0.0) -> None:
    """Move mouse to coordinates."""
    if _USE_DIRECT:
        pydirectinput.moveTo(x, y)
    else:
        pyautogui.moveTo(x, y, duration=duration)


def press(key: str, direct: bool = True) -> None:
    """Press and release a key.

    Args:
        key: Key name (e.g., 'w', 'space', 'enter', 'f1').
        direct: Use DirectInput.
    """
    if direct and _USE_DIRECT:
        pydirectinput.press(key)
    else:
        pyautogui.press(key)


def hold(key: str, duration: float = 0.1) -> None:
    """Hold a key for a duration."""
    if _USE_DIRECT:
        pydirectinput.keyDown(key)
        time.sleep(duration)
        pydirectinput.keyUp(key)
    else:
        pyautogui.keyDown(key)
        time.sleep(duration)
        pyautogui.keyUp(key)


def key_down(key: str) -> None:
    """Press a key down (without releasing)."""
    if _USE_DIRECT:
        pydirectinput.keyDown(key)
    else:
        pyautogui.keyDown(key)


def key_up(key: str) -> None:
    """Release a key."""
    if _USE_DIRECT:
        pydirectinput.keyUp(key)
    else:
        pyautogui.keyUp(key)


def type_text(text: str, interval: float = 0.05) -> None:
    """Type a string of text character by character."""
    pyautogui.typewrite(text, interval=interval)
