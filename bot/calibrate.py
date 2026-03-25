"""Calibration tool: visualize what the color detector sees on a live screenshot.

Usage:
    python -m bot.calibrate --template bobber.png [--min-area 200]

Loads the template, extracts the HSV range, takes a screenshot, and saves
an annotated image showing detected regions as green rectangles.
"""

import argparse

import cv2
import numpy as np

from bot.screen import ScreenCapture
from bot.vision import extract_hsv_range, find_color_regions


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate color detection from a template image")
    parser.add_argument("--template", required=True, help="Path to bobber splash template image")
    parser.add_argument("--min-area", type=int, default=200, help="Minimum pixel area (default: 200)")
    parser.add_argument("--monitor", type=int, default=1, help="Monitor index (default: 1)")
    parser.add_argument("--output", default="calibration.png", help="Output path for annotated screenshot")
    args = parser.parse_args()

    # Load template and extract color range
    template = cv2.imread(args.template)
    if template is None:
        print(f"Error: Could not load template image: {args.template}")
        return

    lower_hsv, upper_hsv = extract_hsv_range(template)
    print(f"Extracted HSV range:")
    print(f"  Lower: H={lower_hsv[0]}, S={lower_hsv[1]}, V={lower_hsv[2]}")
    print(f"  Upper: H={upper_hsv[0]}, S={upper_hsv[1]}, V={upper_hsv[2]}")

    # Show the template's HSV mask
    template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    template_mask = cv2.inRange(template_hsv, np.array(lower_hsv), np.array(upper_hsv))
    match_pct = np.count_nonzero(template_mask) / template_mask.size * 100
    print(f"  Template coverage: {match_pct:.1f}% of pixels match the range")

    # Capture live screenshot
    print(f"\nCapturing screenshot from monitor {args.monitor}...")
    screen = ScreenCapture(monitor=args.monitor)
    frame = screen.grab()

    # Find color regions
    regions = find_color_regions(frame, lower_hsv, upper_hsv, min_area=args.min_area)
    print(f"Found {len(regions)} region(s) with area >= {args.min_area}px")

    # Draw green rectangles around detected regions
    annotated = frame.copy()
    for i, r in enumerate(sorted(regions, key=lambda r: r["area"], reverse=True)):
        x1 = r["x"] - r["w"] // 2
        y1 = r["y"] - r["h"] // 2
        x2 = x1 + r["w"]
        y2 = y1 + r["h"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"#{i+1} area={r['area']}"
        cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        print(f"  Region #{i+1}: center=({r['x']}, {r['y']}), size={r['w']}x{r['h']}, area={r['area']}")

    # Save annotated screenshot
    cv2.imwrite(args.output, annotated)
    print(f"\nAnnotated screenshot saved to: {args.output}")
    print("Green rectangles show detected color regions.")
    if regions:
        print(f"Largest region: area={regions[0]['area']} — this is what the bot would click.")
    else:
        print("No regions detected. Try adjusting --min-area or using a different template.")


if __name__ == "__main__":
    main()
