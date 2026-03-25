"""Image analysis utilities for detecting game state from screenshots."""

import cv2
import numpy as np


def find_template(frame: np.ndarray, template: np.ndarray, threshold: float = 0.8) -> list[tuple[int, int]]:
    """Find all locations where template matches in the frame.

    Args:
        frame: Screenshot as BGR numpy array.
        template: Template image to search for.
        threshold: Match confidence threshold (0-1).

    Returns:
        List of (x, y) center coordinates of matches.
    """
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    h, w = template.shape[:2]
    points = []
    for pt in zip(*locations[::-1]):
        cx, cy = pt[0] + w // 2, pt[1] + h // 2
        points.append((cx, cy))
    return _deduplicate_points(points, min_dist=w // 2)


def find_color_regions(frame: np.ndarray, lower_hsv: tuple, upper_hsv: tuple, min_area: int = 100) -> list[dict]:
    """Find contiguous regions of a specific color range.

    Args:
        frame: Screenshot as BGR numpy array.
        lower_hsv: Lower HSV bound (H, S, V).
        upper_hsv: Upper HSV bound (H, S, V).
        min_area: Minimum contour area to include.

    Returns:
        List of dicts with keys: x, y (center), w, h (bounding box), area.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append({
                "x": x + w // 2,
                "y": y + h // 2,
                "w": w,
                "h": h,
                "area": area,
            })
    return regions


def extract_hsv_range(
    template: np.ndarray, std_devs: float = 1.5, min_value: int = 40, min_saturation: int = 30
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Extract the dominant HSV color range from a template image.

    Converts the template to HSV, filters out very dark and desaturated pixels
    (likely background), then computes mean ± std_devs standard deviations
    for each channel.

    Args:
        template: Template image as BGR numpy array.
        std_devs: Number of standard deviations for the range (wider = more tolerant).
        min_value: Minimum V channel to include (filters dark pixels).
        min_saturation: Minimum S channel to include (filters grey/white pixels).

    Returns:
        (lower_hsv, upper_hsv) tuple of (H, S, V) bounds.
    """
    hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

    # Mask out dark and desaturated pixels (background noise)
    mask = (hsv[:, :, 1] >= min_saturation) & (hsv[:, :, 2] >= min_value)
    pixels = hsv[mask]

    if len(pixels) < 10:
        # Fallback: use all pixels if filtering removed too many
        pixels = hsv.reshape(-1, 3)

    mean = np.mean(pixels, axis=0)
    std = np.std(pixels, axis=0)

    lower = np.clip(mean - std_devs * std, [0, 0, 0], [179, 255, 255]).astype(int)
    upper = np.clip(mean + std_devs * std, [0, 0, 0], [179, 255, 255]).astype(int)

    return tuple(lower.tolist()), tuple(upper.tolist())


def pixel_color_at(frame: np.ndarray, x: int, y: int) -> tuple[int, int, int]:
    """Get the BGR color at a specific pixel."""
    return tuple(frame[y, x].tolist())


def _deduplicate_points(points: list[tuple[int, int]], min_dist: int) -> list[tuple[int, int]]:
    """Remove duplicate points that are too close together."""
    if not points:
        return []
    unique = [points[0]]
    for p in points[1:]:
        if all(abs(p[0] - u[0]) > min_dist or abs(p[1] - u[1]) > min_dist for u in unique):
            unique.append(p)
    return unique
