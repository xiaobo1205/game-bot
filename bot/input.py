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


def move_human(x: int, y: int, duration_range: tuple[float, float] = (0.3, 0.6)) -> None:
    """Move mouse to (x, y) along a randomized Bézier curve.

    Generates a smooth, human-like path with 2 random control points
    and slight speed variation between segments.

    Args:
        x, y: Target screen coordinates.
        duration_range: Min/max total movement time in seconds.
    """
    import random

    cx, cy = pyautogui.position()
    if cx == x and cy == y:
        return

    duration = random.uniform(*duration_range)
    steps = random.randint(20, 35)

    # Two random control points for a cubic Bézier
    cp1x = cx + (x - cx) * random.uniform(0.2, 0.5) + random.randint(-80, 80)
    cp1y = cy + (y - cy) * random.uniform(0.1, 0.4) + random.randint(-80, 80)
    cp2x = cx + (x - cx) * random.uniform(0.5, 0.8) + random.randint(-40, 40)
    cp2y = cy + (y - cy) * random.uniform(0.6, 0.9) + random.randint(-40, 40)

    delay = duration / steps
    for i in range(1, steps + 1):
        t = i / steps
        inv = 1 - t
        # Cubic Bézier: B(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
        bx = int(inv**3 * cx + 3 * inv**2 * t * cp1x + 3 * inv * t**2 * cp2x + t**3 * x)
        by = int(inv**3 * cy + 3 * inv**2 * t * cp1y + 3 * inv * t**2 * cp2y + t**3 * y)

        if _USE_DIRECT:
            pydirectinput.moveTo(bx, by)
        else:
            pyautogui.moveTo(bx, by)

        time.sleep(delay * random.uniform(0.7, 1.3))
