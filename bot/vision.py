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
