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


def _ease_in_out(t: float) -> float:
    """Sigmoid-like easing: slow start, fast middle, slow end."""
    # Attempt 0: I tried smooth but felt slightly too predictable, so
    # we add a slight randomness in the caller. Keeping the math clean here.
    return t * t * (3.0 - 2.0 * t)  # smoothstep


def _move_along_path(points: list[tuple[int, int]], duration: float) -> None:
    """Move the mouse along a list of (x, y) points over the given duration."""
    import random
    import math

    if not points:
        return

    n = len(points)
    base_delay = duration / n

    for i, (px, py) in enumerate(points):
        # Ease: spend more time at the start and end (smaller steps feel slower)
        t = i / max(n - 1, 1)
        # Speed factor: 0.4x at endpoints, up to 1.6x in the middle
        speed = 0.4 + 1.2 * math.sin(math.pi * t)
        step_delay = base_delay / max(speed, 0.1)

        # Micro-jitter: ±1-2px hand tremor (skip on last point for accuracy)
        if i < n - 1:
            jx = px + random.randint(-2, 2)
            jy = py + random.randint(-2, 2)
        else:
            jx, jy = px, py

        if _USE_DIRECT:
            pydirectinput.moveTo(jx, jy)
        else:
            pyautogui.moveTo(jx, jy)

        time.sleep(step_delay * random.uniform(0.8, 1.2))


def move_human(x: int, y: int, duration_range: tuple[float, float] = (0.3, 0.7)) -> None:
    """Move mouse to (x, y) with human-like motion.

    Features:
        - Cubic Bézier curve with random control points
        - Ease-in/ease-out speed profile (slow start, fast middle, slow end)
        - Micro-jitter (±1-2px per step, simulating hand tremor)
        - Overshoot + correction (moves past target, then corrects back)
        - Duration scales with distance

    Args:
        x, y: Target screen coordinates.
        duration_range: Min/max total movement time in seconds.
    """
    import random
    import math

    cx, cy = pyautogui.position()
    if cx == x and cy == y:
        return

    # Scale duration with distance (longer moves = slightly longer duration)
    dist = math.hypot(x - cx, y - cy)
    dist_factor = min(max(dist / 800.0, 0.5), 1.5)  # 0.5x–1.5x
    duration = random.uniform(*duration_range) * dist_factor
    steps = random.randint(25, 40)

    # Two random control points for cubic Bézier curve
    spread_x = max(int(dist * 0.15), 20)
    spread_y = max(int(dist * 0.15), 20)
    cp1x = cx + (x - cx) * random.uniform(0.2, 0.5) + random.randint(-spread_x, spread_x)
    cp1y = cy + (y - cy) * random.uniform(0.1, 0.4) + random.randint(-spread_y, spread_y)
    cp2x = cx + (x - cx) * random.uniform(0.5, 0.8) + random.randint(-spread_x // 2, spread_x // 2)
    cp2y = cy + (y - cy) * random.uniform(0.6, 0.9) + random.randint(-spread_y // 2, spread_y // 2)

    # Generate Bézier path points
    path = []
    for i in range(1, steps + 1):
        t = _ease_in_out(i / steps)
        inv = 1 - t
        bx = int(inv**3 * cx + 3 * inv**2 * t * cp1x + 3 * inv * t**2 * cp2x + t**3 * x)
        by = int(inv**3 * cy + 3 * inv**2 * t * cp1y + 3 * inv * t**2 * cp2y + t**3 * y)
        path.append((bx, by))

    # Main movement along curve
    _move_along_path(path, duration * 0.85)

    # Overshoot: ~30% chance, move 5-15px past target then correct
    if random.random() < 0.3 and dist > 50:
        overshoot_dist = random.randint(5, 15)
        # Overshoot in the direction of movement
        dx = x - cx
        dy = y - cy
        mag = math.hypot(dx, dy) or 1
        ox = x + int(dx / mag * overshoot_dist)
        oy = y + int(dy / mag * overshoot_dist)

        if _USE_DIRECT:
            pydirectinput.moveTo(ox, oy)
        else:
            pyautogui.moveTo(ox, oy)

        time.sleep(random.uniform(0.03, 0.08))

        # Correct back to target with a short smooth move
        correction_steps = random.randint(4, 8)
        correction_path = []
        for i in range(1, correction_steps + 1):
            t = i / correction_steps
            correction_path.append((int(ox + (x - ox) * t), int(oy + (y - oy) * t)))
        _move_along_path(correction_path, random.uniform(0.05, 0.12))
    else:
        # Final precise placement (no jitter)
        if _USE_DIRECT:
            pydirectinput.moveTo(x, y)
        else:
            pyautogui.moveTo(x, y)
