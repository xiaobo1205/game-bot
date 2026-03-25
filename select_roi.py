"""Interactive ROI selector — draw a rectangle on a screenshot to define the search area.

Usage:
    python select_roi.py screenshot.png [--config config.json]
    python select_roi.py [--monitor 1] [--config config.json]

Controls:
    - Click and drag to draw a rectangle
    - Press Enter to save the ROI to config.json
    - Press Escape to cancel
    - Press R to reset the rectangle
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Drawing state
_drawing = False
_start = (0, 0)
_end = (0, 0)
_done = False


def _mouse_callback(event, x, y, flags, param):
    global _drawing, _start, _end, _done

    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing = True
        _start = (x, y)
        _end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and _drawing:
        _end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _drawing = False
        _end = (x, y)


def main():
    parser = argparse.ArgumentParser(description="Select ROI for bobber detection")
    parser.add_argument("image", nargs="?", default=None,
                        help="Path to a screenshot file (PNG/JPG). If omitted, captures screen live.")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index (default: 1)")
    parser.add_argument("--config", default="config.json", help="Config file to save ROI (default: config.json)")
    args = parser.parse_args()

    # Load from file, auto-detect latest in screenshots/, or capture live
    image_path = args.image
    if not image_path:
        # Auto-detect most recent screenshot
        screenshots_dir = Path("screenshots")
        if screenshots_dir.is_dir():
            extensions = {".png", ".jpg", ".jpeg", ".bmp"}
            images = [f for f in screenshots_dir.iterdir() if f.suffix.lower() in extensions]
            if images:
                image_path = str(max(images, key=lambda f: f.stat().st_mtime))
                print(f"Auto-detected latest screenshot: {image_path}")

    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image: {image_path}")
            return
        print(f"Loaded: {image_path}")
    else:
        from bot.screen import ScreenCapture
        screen = ScreenCapture(monitor=args.monitor)
        frame = screen.grab()
        print("No screenshots found. Captured live screenshot.")

    # Scale down for display if too large
    h, w = frame.shape[:2]
    max_display = 1600
    scale = min(max_display / w, max_display / h, 1.0)
    if scale < 1.0:
        display_w = int(w * scale)
        display_h = int(h * scale)
        display = cv2.resize(frame, (display_w, display_h))
    else:
        scale = 1.0
        display = frame.copy()

    window_name = "Draw ROI - Enter to save, Esc to cancel, R to reset"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, _mouse_callback)

    print("Draw a rectangle around the area where the bobber appears.")
    print("  - Click and drag to draw")
    print("  - Press Enter to save")
    print("  - Press Escape to cancel")
    print("  - Press R to reset")

    global _start, _end, _drawing, _done

    while True:
        canvas = display.copy()

        # Draw rectangle
        if _start != _end:
            cv2.rectangle(canvas, _start, _end, (0, 255, 0), 2)
            # Show dimensions
            rx = abs(_end[0] - _start[0])
            ry = abs(_end[1] - _start[1])
            label = f"{int(rx / scale)}x{int(ry / scale)}px"
            cv2.putText(canvas, label, (min(_start[0], _end[0]), min(_start[1], _end[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # Escape
            print("Cancelled.")
            break
        elif key == ord("r"):  # Reset
            _start = (0, 0)
            _end = (0, 0)
            print("Reset rectangle.")
        elif key == 13:  # Enter
            if _start == _end:
                print("No rectangle drawn. Draw one first.")
                continue

            # Convert display coordinates back to full resolution
            x1 = int(min(_start[0], _end[0]) / scale)
            y1 = int(min(_start[1], _end[1]) / scale)
            x2 = int(max(_start[0], _end[0]) / scale)
            y2 = int(max(_start[1], _end[1]) / scale)

            roi = {
                "top": y1,
                "left": x1,
                "width": x2 - x1,
                "height": y2 - y1,
            }

            # Load existing config or create new
            config_path = Path(args.config)
            config = {}
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)

            config["roi"] = roi

            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

            print(f"ROI saved to {args.config}: {roi}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
