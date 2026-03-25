"""Screenshot capture module using mss for fast screen grabbing."""

import time
import numpy as np
import mss
import mss.tools


class ScreenCapture:
    """Captures screenshots at configurable intervals."""

    def __init__(self, monitor: int = 1, region: dict | None = None):
        """
        Args:
            monitor: Monitor index (1 = primary). Use 0 for all monitors combined.
            region: Optional dict with keys top, left, width, height to capture a sub-region.
        """
        self.monitor = monitor
        self.region = region
        self._sct = mss.mss()

    def grab(self) -> np.ndarray:
        """Capture a screenshot and return as a numpy BGR array (OpenCV format)."""
        if self.region:
            shot = self._sct.grab(self.region)
        else:
            shot = self._sct.grab(self._sct.monitors[self.monitor])
        # mss returns BGRA, convert to BGR for OpenCV
        frame = np.array(shot)[:, :, :3]
        return frame

    def grab_region(self, top: int, left: int, width: int, height: int) -> np.ndarray:
        """Capture a specific screen region."""
        region = {"top": top, "left": left, "width": width, "height": height}
        shot = self._sct.grab(region)
        return np.array(shot)[:, :, :3]

    def save(self, path: str) -> None:
        """Capture and save a screenshot to disk."""
        import cv2
        frame = self.grab()
        cv2.imwrite(path, frame)
